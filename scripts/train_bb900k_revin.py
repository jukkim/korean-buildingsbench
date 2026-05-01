# -*- coding: utf-8 -*-
"""Train our model (with RevIN) on BB 900K data.
Run on 5090: C:\\Python313\\python.exe scripts/train_bb900k_revin.py

Purpose: Fair comparison — same architecture + RevIN, BB data vs Korean data.
"""
import sys, os, time, torch, numpy as np, tomli
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent
os.environ['BUILDINGS_BENCH'] = str(Path(r'E:\BuildingsBench'))
sys.path.insert(0, str(BASE))

from buildings_bench import load_pretraining
from src.models.transformer import model_factory

# Config
with open(BASE / 'configs' / 'model' / 'TransformerWithGaussian-M-v3-3k.toml', 'rb') as f:
    cfg = tomli.load(f)

model = model_factory(cfg['model'])
device = torch.device('cuda')
model = model.to(device)
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params, RevIN={model.use_revin}')

# Load BB 900K
print('Loading BB 900K train...')
train_dataset = load_pretraining(
    'buildings-900k-train',
    apply_scaler_transform='boxcox',
    scaler_transform_path=Path(os.environ['BUILDINGS_BENCH']) / 'metadata' / 'transforms',
)
print(f'Train dataset: {len(train_dataset)} windows')

print('Loading BB 900K val...')
val_dataset = load_pretraining(
    'buildings-900k-val',
    apply_scaler_transform='boxcox',
    scaler_transform_path=Path(os.environ['BUILDINGS_BENCH']) / 'metadata' / 'transforms',
)
print(f'Val dataset: {len(val_dataset)} windows')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Optimizer (same as BB)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')

max_steps = 651042  # BB original: 1B tokens / (1 GPU * 64 batch * 24 pred_len)
warmup_steps = 10000  # BB original

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training
model.train()
step = 0
best_val_loss = float('inf')
log_path = BASE / 'logs' / 'bb900k_revin_on.log'

print(f'Training: max_steps={max_steps}, warmup={warmup_steps}')
print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')
t0 = time.time()

for epoch in range(100):
    if step >= max_steps:
        break
    for batch in train_loader:
        if step >= max_steps:
            break
        for k, v in batch.items():
            batch[k] = v.to(device)

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
            lr = scheduler.get_last_lr()[0]
            print(f'  step {step:,}/{max_steps:,} | loss={loss.item():.5f} | lr={lr:.2e} | {elapsed:.0f}s', flush=True)

    # Epoch end
    ep_num = epoch + 1
    print(f'  Epoch {ep_num} done at step {step}', flush=True)

    # Save checkpoint
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': ep_num,
    }
    ckpt_path = BASE / 'checkpoints' / 'bb900k_revin_on_last.pt'
    torch.save(ckpt, ckpt_path)

elapsed = time.time() - t0
print(f'\nTraining complete: {step} steps, {elapsed:.0f}s')
print(f'Checkpoint: {ckpt_path}')

# BB eval will be done separately with evaluate_bb.py
