# -*- coding: utf-8 -*-
"""Zero-shot evaluation on Korean convenience stores (100 real buildings).
Run with bb_repro: python -X utf8 scripts/eval_korean_stores.py
"""
import sys, os, torch, numpy as np, pandas as pd, tomli, time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.transformer import model_factory

BASE = Path(__file__).parent.parent
STORE_DIR = Path(r'H:/내 드라이브/과제수행(T&M)/작업 수행(3차년)/인증시험(녹색기후기술원)/인증시험(이상 탐지)/anomaly')
BB_DATA = BASE / 'external' / 'BuildingsBench_data'

# Load boxcox
import pickle
with open(BB_DATA / 'metadata' / 'transforms' / 'boxcox.pkl', 'rb') as f:
    boxcox = pickle.load(f)

# Load model
config_path = BASE / 'configs' / 'model' / 'TransformerWithGaussian-M-v3-3k.toml'
ckpt_path = BASE / 'checkpoints' / 'TransformerWithGaussian-M-v3-3k_ms_n50_s18000_revin_on_seed42_bb_best.pt'

with open(config_path, 'rb') as f:
    cfg = tomli.load(f)
model = model_factory(cfg['model'])
ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params, RevIN={model.use_revin}')

context_len = cfg['model']['context_len']  # 168
pred_len = cfg['model']['pred_len']  # 24
total_len = context_len + pred_len

# Load stores
stores = sorted([d.name for d in STORE_DIR.iterdir() if d.is_dir()])
print(f'Stores: {len(stores)}')

cvrmses = []
t0 = time.time()

for sid in stores:
    csv_path = STORE_DIR / sid / f'{sid}_hourly_labeled.csv'
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)
    # energy in Wh → kWh
    series = df['energy'].values.astype(np.float64) / 1000.0

    if len(series) < total_len:
        continue

    # Sliding windows (stride=24)
    starts = np.arange(0, len(series) - total_len + 1, pred_len)
    if len(starts) == 0:
        continue

    # Timestamps
    timestamps = pd.to_datetime(df['date'])
    doy = timestamps.dt.dayofyear.values.astype(np.float32)
    dow = timestamps.dt.dayofweek.values.astype(np.float32)
    hod = timestamps.dt.hour.values.astype(np.float32)
    # Normalize to [-1, 1]
    doy = (doy - 183) / 183
    dow = (dow - 3) / 3
    hod = (hod - 11.5) / 11.5

    sum_sq_err = 0.0
    sum_abs_gt = 0.0
    n_targets = 0
    batch_size = 64

    for chunk_start in range(0, len(starts), 256):
        chunk_starts = starts[chunk_start:chunk_start + 256]
        idx = chunk_starts[:, None] + np.arange(total_len)
        wins = np.clip(series[idx], 1e-6, None).astype(np.float32)
        wins_bc = boxcox.transform(wins.reshape(-1, 1)).reshape(wins.shape).astype(np.float32)
        doy_w = doy[idx]
        dow_w = dow[idx]
        hod_w = hod[idx]

        for i in range(0, len(chunk_starts), batch_size):
            bw = wins_bc[i:i + batch_size]
            B = len(bw)
            T = total_len
            load_t = torch.from_numpy(bw).unsqueeze(-1).float().to(device)
            zeros = torch.zeros(B, T, 1, device=device)
            batch_d = {
                'load': load_t,
                'latitude': zeros,
                'longitude': zeros,
                'building_type': torch.ones(B, T, 1, dtype=torch.long, device=device),
                'day_of_year': torch.from_numpy(doy_w[i:i + B]).unsqueeze(-1).to(device),
                'day_of_week': torch.from_numpy(dow_w[i:i + B]).unsqueeze(-1).to(device),
                'hour_of_day': torch.from_numpy(hod_w[i:i + B]).unsqueeze(-1).to(device),
            }
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                preds, _ = model.predict(batch_d)
            pred_bc = np.clip(preds.squeeze(-1).detach().cpu().numpy(), -5.0, 5.0)
            tgt_bc = wins_bc[i:i + B, context_len:]
            pred_orig = boxcox.inverse_transform(pred_bc.reshape(-1, 1)).reshape(pred_bc.shape)
            tgt_orig = boxcox.inverse_transform(tgt_bc.reshape(-1, 1)).reshape(tgt_bc.shape)
            se = (pred_orig - tgt_orig) ** 2
            sum_sq_err += float(np.nansum(se))
            sum_abs_gt += float(np.abs(tgt_orig).sum())
            n_targets += int(tgt_orig.size)

    if n_targets == 0:
        continue
    mean_gt = sum_abs_gt / n_targets
    if mean_gt < 1e-8:
        continue
    cvrmse_val = float(np.sqrt(sum_sq_err / n_targets) / mean_gt)
    if np.isfinite(cvrmse_val):
        cvrmses.append(cvrmse_val)

    if len(cvrmses) % 20 == 0:
        print(f'  [{len(cvrmses):3d}/{len(stores)}] median={np.median(cvrmses)*100:.2f}%', flush=True)

elapsed = time.time() - t0
print(f'\n=== Korean Convenience Store Zero-Shot ({elapsed:.0f}s) ===')
print(f'Stores: {len(cvrmses)}/{len(stores)}')
print(f'Median CVRMSE: {np.median(cvrmses)*100:.2f}%')
print(f'Mean CVRMSE: {np.mean(cvrmses)*100:.2f}%')
print(f'Min: {np.min(cvrmses)*100:.2f}%, Max: {np.max(cvrmses)*100:.2f}%')
