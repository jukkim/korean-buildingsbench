# -*- coding: utf-8 -*-
"""Evaluate BB checkpoint WITH RevIN (inference-only).
Run with bb_repro conda: C:\\Users\\User\\miniconda3\\envs\\bb_repro\\python.exe -X utf8

RevIN at inference: normalize context by its mean/std, predict, denormalize.
"""
import sys, os, torch, numpy as np, pickle, time, tomli
from pathlib import Path
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent
os.environ['BUILDINGS_BENCH'] = str(BASE / 'external' / 'BuildingsBench_data')

from buildings_bench.models import model_factory

BB_DATA = BASE / 'external' / 'BuildingsBench_data'
CKPT = BB_DATA / 'checkpoints' / 'Transformer_Gaussian_M.pt'
TRANSFORM_PATH = BB_DATA / 'metadata' / 'transforms'

# Load boxcox
with open(TRANSFORM_PATH / 'boxcox.pkl', 'rb') as f:
    boxcox = pickle.load(f)

# Load model
with open(BASE / 'external' / 'BuildingsBench' / 'buildings_bench' / 'configs' / 'TransformerWithGaussian-M.toml', 'rb') as f:
    cfg = tomli.load(f)
model, _, predict_fn = model_factory('TransformerWithGaussian-M', cfg['model'])
model.load_from_checkpoint(str(CKPT))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()

print(f'BB Model: {sum(p.numel() for p in model.parameters()):,} params')
print(f'torch={torch.__version__}, device={device}')

# Load BB buildings (same as train.py)
sys.path.insert(0, str(BASE))
from scripts.train import _load_bb_buildings, _BBBoxCox
bb_boxcox = _BBBoxCox()
bb_buildings = _load_bb_buildings(commercial_only=True)
print(f'Buildings: {len(bb_buildings)} commercial')

context_len, pred_len = 168, 24
total_len = context_len + pred_len
batch_size = 64

cvrmses_revin = []
cvrmses_norevin = []
t0 = time.time()

for bid, btype, series, doy_1d, dow_1d, hod_1d in bb_buildings:
    n = len(series)
    starts = np.arange(0, n - total_len + 1, pred_len)
    if len(starts) == 0:
        continue

    se_revin = 0.0
    se_norevin = 0.0
    sum_gt = 0.0
    nt = 0

    for cs in range(0, len(starts), 256):
        cstarts = starts[cs:cs + 256]
        idx = cstarts[:, None] + np.arange(total_len)
        wins = np.clip(series[idx], 1e-6, None).astype(np.float32)
        wins_bc = bb_boxcox.scaler.transform(wins.reshape(-1, 1)).reshape(wins.shape).astype(np.float32)

        for i in range(0, len(cstarts), batch_size):
            bw = wins_bc[i:i + batch_size]
            B = len(bw)
            T = total_len
            load_t = torch.from_numpy(bw).unsqueeze(-1).float().to(device)
            zeros = torch.zeros(B, T, 1, device=device)
            batch_d = {
                'load': load_t.clone(),
                'latitude': zeros,
                'longitude': zeros,
                'building_type': torch.ones(B, T, 1, dtype=torch.long, device=device),
                'day_of_year': torch.from_numpy(doy_1d[idx[i:i + B]]).unsqueeze(-1).to(device),
                'day_of_week': torch.from_numpy(dow_1d[idx[i:i + B]]).unsqueeze(-1).to(device),
                'hour_of_day': torch.from_numpy(hod_1d[idx[i:i + B]]).unsqueeze(-1).to(device),
            }

            tgt_bc = bw[..., context_len:]

            # === No RevIN (baseline) ===
            with torch.amp.autocast('cuda', enabled=device == 'cuda'):
                preds_no, _ = predict_fn(batch_d)
            pred_bc_no = preds_no.squeeze(-1).detach().cpu().numpy()

            # === With RevIN (inference-only) ===
            batch_revin = {k: v.clone() for k, v in batch_d.items()}
            ctx = batch_revin['load'][:, :context_len, :]  # (B, 168, 1)
            rv_mean = ctx.mean(dim=1, keepdim=True)  # (B, 1, 1)
            rv_std = ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
            batch_revin['load'] = (batch_revin['load'] - rv_mean) / rv_std

            with torch.amp.autocast('cuda', enabled=device == 'cuda'):
                preds_rv, _ = predict_fn(batch_revin)

            # Denormalize
            preds_rv = preds_rv * rv_std[:, :1, :] + rv_mean[:, :1, :]
            pred_bc_rv = preds_rv.squeeze(-1).detach().cpu().numpy()

            # Inverse transform
            pred_orig_no = bb_boxcox.scaler.inverse_transform(pred_bc_no.reshape(-1, 1)).reshape(pred_bc_no.shape)
            pred_orig_rv = bb_boxcox.scaler.inverse_transform(pred_bc_rv.reshape(-1, 1)).reshape(pred_bc_rv.shape)
            tgt_orig = bb_boxcox.scaler.inverse_transform(tgt_bc.reshape(-1, 1)).reshape(tgt_bc.shape)

            se_norevin += float(np.nansum(np.square(pred_orig_no - tgt_orig)))
            se_revin += float(np.nansum(np.square(pred_orig_rv - tgt_orig)))
            sum_gt += float(np.abs(tgt_orig).sum())
            nt += int(tgt_orig.size)

    if nt == 0:
        continue
    mg = sum_gt / nt
    if mg < 1e-8:
        continue

    cv_no = float(np.sqrt(se_norevin / nt) / mg)
    cv_rv = float(np.sqrt(se_revin / nt) / mg)

    if np.isfinite(cv_no):
        cvrmses_norevin.append(cv_no)
    if np.isfinite(cv_rv):
        cvrmses_revin.append(cv_rv)

    if len(cvrmses_norevin) % 100 == 0:
        print(f'  {len(cvrmses_norevin)} bldgs | no_revin={np.median(cvrmses_norevin)*100:.2f}% | revin={np.median(cvrmses_revin)*100:.2f}%')
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f'\n=== BB SOTA-M + RevIN ({elapsed:.0f}s) ===')
print(f'Without RevIN: {len(cvrmses_norevin)} bldgs, median={np.median(cvrmses_norevin)*100:.2f}%')
print(f'With RevIN:    {len(cvrmses_revin)} bldgs, median={np.median(cvrmses_revin)*100:.2f}%')
print(f'RevIN effect:  {(np.median(cvrmses_revin)-np.median(cvrmses_norevin))*100:+.2f}%p')
