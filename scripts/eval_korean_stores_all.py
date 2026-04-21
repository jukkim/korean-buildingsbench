# -*- coding: utf-8 -*-
"""Zero-shot evaluation on Korean convenience stores (100 + 125 = 225 buildings).
Evaluates BOTH our model AND BB SOTA-M checkpoint.
"""
import sys, os, torch, numpy as np, pandas as pd, tomli, time, pickle
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.transformer import model_factory

BASE = Path(__file__).parent.parent
BB_DATA = BASE / 'external' / 'BuildingsBench_data'

# Paths
STORE_100_DIR = Path(r'H:/내 드라이브/과제수행(T&M)/작업 수행(3차년)/인증시험(녹색기후기술원)/인증시험(이상 탐지)/anomaly')
STORE_120_DIR = Path(r'H:/내 드라이브/과제수행(T&M)/data/ARCHIVE_ELEC_BASE_PARSING_120')

# Load boxcox
with open(BB_DATA / 'metadata' / 'transforms' / 'boxcox.pkl', 'rb') as f:
    boxcox = pickle.load(f)

context_len, pred_len = 168, 24
total_len = context_len + pred_len


def load_store_100():
    """100개 점포: hourly labeled CSV (Wh)"""
    stores = {}
    for d in sorted(STORE_100_DIR.iterdir()):
        if not d.is_dir():
            continue
        csv = d / f'{d.name}_hourly_labeled.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df['energy'].mean() < 10:  # < 0.01 kWh avg → skip
            continue
        series = df['energy'].values.astype(np.float64) / 1000.0  # Wh → kWh
        ts = pd.to_datetime(df['date'])
        stores[f'GS100_{d.name}'] = (series, ts)
    return stores


def load_store_120():
    """120개 점포: 5min CSV → hourly (Wh)"""
    stores = {}
    for csv in sorted(STORE_120_DIR.glob('*.csv')):
        df = pd.read_csv(csv, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True).dt.tz_localize(None)
        df['hour'] = df['date'].dt.floor('h')
        hourly = df.groupby('hour')['ch1_5min'].sum()
        if len(hourly) < total_len:
            continue
        if hourly.mean() < 10:  # skip dead stores
            continue
        series = hourly.values.astype(np.float64) / 1000.0  # Wh → kWh
        ts = hourly.index
        stores[f'GS120_{csv.stem}'] = (series, ts)
    return stores


def eval_model_on_stores(model, stores, model_name, device='cuda'):
    """Evaluate a model on all stores."""
    cvrmses = []
    batch_size = 64

    for sid, (series, ts) in stores.items():
        if len(series) < total_len:
            continue
        starts = np.arange(0, len(series) - total_len + 1, pred_len)
        if len(starts) == 0:
            continue

        ts_idx = pd.DatetimeIndex(ts)
        doy = (ts_idx.dayofyear.values.astype(np.float32) - 183) / 183
        dow = (ts_idx.dayofweek.values.astype(np.float32) - 3) / 3
        hod = (ts_idx.hour.values.astype(np.float32) - 11.5) / 11.5

        sum_se = 0.0; sum_gt = 0.0; nt = 0
        idx = starts[:, None] + np.arange(total_len)
        wins = np.clip(series[idx], 1e-6, None).astype(np.float32)
        wins_bc = boxcox.transform(wins.reshape(-1, 1)).reshape(wins.shape).astype(np.float32)

        for i in range(0, len(starts), batch_size):
            bw = wins_bc[i:i + batch_size]; B = len(bw); T = total_len
            load_t = torch.from_numpy(bw).unsqueeze(-1).float().to(device)
            zeros = torch.zeros(B, T, 1, device=device)
            batch_d = {
                'load': load_t, 'latitude': zeros, 'longitude': zeros,
                'building_type': torch.ones(B, T, 1, dtype=torch.long, device=device),
                'day_of_year': torch.from_numpy(doy[idx[i:i + B]]).unsqueeze(-1).to(device),
                'day_of_week': torch.from_numpy(dow[idx[i:i + B]]).unsqueeze(-1).to(device),
                'hour_of_day': torch.from_numpy(hod[idx[i:i + B]]).unsqueeze(-1).to(device),
            }
            with torch.amp.autocast('cuda', enabled=device == 'cuda'):
                preds, _ = model.predict(batch_d)
            pred_bc = np.clip(preds.squeeze(-1).detach().cpu().numpy(), -5.0, 5.0)
            tgt_bc = wins_bc[i:i + B, context_len:]
            pred_orig = boxcox.inverse_transform(pred_bc.reshape(-1, 1)).reshape(pred_bc.shape)
            tgt_orig = boxcox.inverse_transform(tgt_bc.reshape(-1, 1)).reshape(tgt_bc.shape)
            sum_se += float(np.nansum((pred_orig - tgt_orig) ** 2))
            sum_gt += float(np.abs(tgt_orig).sum())
            nt += int(tgt_orig.size)

        if nt == 0:
            continue
        mg = sum_gt / nt
        if mg < 1e-8:
            continue
        cv = float(np.sqrt(sum_se / nt) / mg)
        if np.isfinite(cv) and cv < 10:  # skip > 1000%
            cvrmses.append({'store': sid, 'cvrmse': cv})

    cdf = pd.DataFrame(cvrmses)
    med = cdf['cvrmse'].median() * 100
    mean = cdf['cvrmse'].mean() * 100
    print(f'  {model_name}: {len(cdf)} stores, Median={med:.2f}%, Mean={mean:.2f}%')
    return cdf


# Load stores
print('Loading 100 stores...', flush=True)
stores_100 = load_store_100()
print(f'  Loaded: {len(stores_100)}')

print('Loading 120 stores...', flush=True)
stores_120 = load_store_120()
print(f'  Loaded: {len(stores_120)}')

all_stores = {**stores_100, **stores_120}
print(f'Total: {len(all_stores)} stores\n')

# Model 1: Our Korean-700 (RevIN ON)
print('=== Model 1: Korean-700 (RevIN ON) ===')
with open(BASE / 'configs' / 'model' / 'TransformerWithGaussian-M-v3-3k.toml', 'rb') as f:
    cfg = tomli.load(f)
our_model = model_factory(cfg['model'])
ckpt = torch.load(str(BASE / 'checkpoints' / 'TransformerWithGaussian-M-v3-3k_ms_n50_s18000_revin_on_seed42_bb_best.pt'),
                   map_location='cpu', weights_only=False)
our_model.load_state_dict(ckpt['model_state_dict'])
our_model = our_model.cuda().eval()

r1_100 = eval_model_on_stores(our_model, stores_100, 'Korean-700 (100 stores)')
r1_120 = eval_model_on_stores(our_model, stores_120, 'Korean-700 (120 stores)')
r1_all = eval_model_on_stores(our_model, all_stores, 'Korean-700 (all)')

del our_model
torch.cuda.empty_cache()

# Model 2: BB SOTA-M (no RevIN)
print('\n=== Model 2: BB SOTA-M ===')
os.environ['BUILDINGS_BENCH'] = str(BB_DATA)
from buildings_bench.models import model_factory as bb_model_factory
with open(BASE / 'external' / 'BuildingsBench' / 'buildings_bench' / 'configs' / 'TransformerWithGaussian-M.toml', 'rb') as f:
    bb_cfg = tomli.load(f)
bb_model, _, bb_predict = bb_model_factory('TransformerWithGaussian-M', bb_cfg['model'])
bb_model.load_from_checkpoint(str(BB_DATA / 'checkpoints' / 'Transformer_Gaussian_M.pt'))
bb_model = bb_model.cuda().eval()

# BB model uses different predict interface
class BBWrapper:
    def __init__(self, model, predict_fn):
        self.model = model
        self._predict = predict_fn
        self.context_len = model.context_len
        self.pred_len = model.pred_len
    def predict(self, batch):
        return self._predict(batch)
    def eval(self):
        self.model.eval()
        return self

bb_wrap = BBWrapper(bb_model, bb_predict)
r2_all = eval_model_on_stores(bb_wrap, all_stores, 'BB SOTA-M (all)')

print(f'\n=== Summary ===')
print(f'Korean-700 (all {len(all_stores)} stores): Median={r1_all["cvrmse"].median()*100:.2f}%')
print(f'BB SOTA-M (all {len(all_stores)} stores):  Median={r2_all["cvrmse"].median()*100:.2f}%')
