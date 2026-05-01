# -*- coding: utf-8 -*-
"""Train our model (RevIN) on BB 900K — single GPU, optimized DataLoader.
Based on BB pretrain.py but without DDP.

Run on 5090: C:\\Python313\\python.exe scripts/train_bb900k_revin_v2.py
"""
import sys, os, time, torch, numpy as np, tomli
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent
os.environ['BUILDINGS_BENCH'] = str(Path(r'E:\BuildingsBench'))
sys.path.insert(0, str(BASE))

from buildings_bench import load_pretraining, utils
from src.models.transformer import model_factory

# Config — same as BB
with open(BASE / 'configs' / 'model' / 'TransformerWithGaussian-M-v3-3k.toml', 'rb') as f:
    cfg = tomli.load(f)

model = model_factory(cfg['model'])
device = torch.device('cuda')
model = model.to(device)
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params, RevIN={model.use_revin}')

# BB 900K data
transform_path = Path(os.environ['BUILDINGS_BENCH']) / 'metadata' / 'transforms'
print('Loading BB 900K train...', flush=True)
train_dataset = load_pretraining(
    'buildings-900k-train',
    apply_scaler_transform='boxcox',
    scaler_transform_path=transform_path,
)
print(f'Train: {len(train_dataset)} windows', flush=True)

# DataLoader — num_workers=4, pin_memory for speed
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,  # sequential access for speed (random enough with 900K buildings)
    num_workers=0,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn(),
    drop_last=True,
)

# BB original settings
batch_size = 64
pred_len = 24
train_tokens = 1_000_000_000
max_steps = train_tokens // (batch_size * pred_len)  # 651,042
warmup_steps = 10000
lr = 6e-5

print(f'max_steps={max_steps:,}, warmup={warmup_steps}, lr={lr}')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                               betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

import transformers
scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

try:
    scaler = torch.amp.GradScaler()
except AttributeError:
    scaler = torch.cuda.amp.GradScaler()

# Training
model.train()
step = 0
log_path = BASE / 'logs' / 'bb900k_revin_v2.log'
ckpt_dir = BASE / 'checkpoints'

print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
t0 = time.time()
epoch = 0

while step < max_steps:
    epoch += 1
    for batch in train_loader:
        if step >= max_steps:
            break
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = model(batch)
            targets = batch['load'][:, model.context_len:]
            loss = model.loss(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if step % 1000 == 0:
            elapsed = time.time() - t0
            seen = step * batch_size * pred_len
            lr_now = scheduler.get_last_lr()[0]
            msg = f'  step {step:,}/{max_steps:,} | loss={loss.item():.5f} | lr={lr_now:.2e} | tokens={seen:,} | {elapsed:.0f}s'
            print(msg, flush=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

        if step % 10000 == 0:
            ckpt = {'model_state_dict': model.state_dict(), 'step': step, 'epoch': epoch}
            torch.save(ckpt, ckpt_dir / f'bb900k_revin_step{step}.pt')

    print(f'  Epoch {epoch} done at step {step:,}', flush=True)

# Final save
ckpt = {'model_state_dict': model.state_dict(), 'step': step, 'epoch': epoch}
torch.save(ckpt, ckpt_dir / 'bb900k_revin_last.pt')

elapsed = time.time() - t0
msg = f'\nDone: {step:,} steps, {elapsed:.0f}s ({elapsed/3600:.1f}h)'
print(msg, flush=True)
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(msg + '\n')
