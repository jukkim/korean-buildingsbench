"""BB 공식 테스트셋으로 평가 — BB SOTA와 공정 비교

BB 프로토콜:
  1. 7개 데이터셋(BDG-2, Electricity, IDEAL, LCL, SMART, Sceaux, Borealis)
  2. OOV 건물 제외
  3. Per-building CVRMSE = sqrt(mean(SE)) / mean(|actual|)
  4. Aggregate: median across buildings
  5. Bootstrap 95% CI (50,000 reps)

주의: BB 공식 Box-Cox transform 사용 (lambda=-0.067, 우리 Korean과 다름)

Usage:
    python scripts/evaluate_bb.py \
        --checkpoint checkpoints/TransformerWithGaussian-M_office_best.pt \
        --config configs/model/TransformerWithGaussian-M.toml
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    # 절전 방지: eval 중 Windows 절전으로 프로세스 종료 방지 (관리자 권한 불필요)
    import ctypes
    ES_CONTINUOUS      = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import tomllib as tomli
except ModuleNotFoundError:
    import tomli
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.transformer import model_factory

PROJECT_DIR = Path(__file__).parent.parent
BB_DATA_DIR = PROJECT_DIR / 'external' / 'BuildingsBench_data'


# ============================================================
# BB 로딩 기준 (절대 변경 금지)
# ============================================================
#
# 로딩 방식: per-year 독립 로딩 (빈파일 스킵, 연도별 NaN>50% 제외)
# - BDG-2: 4개 사이트 모든 연도 로딩 (Panther 2017 포함)
#   Bear + Fox + Panther + Rat = 611
# - Electricity: 2011~2014 모든 연도 합집합, OOV 제외
#   = 344
# - 상업용 합계: 611 + 344 = 955
# - 주거용 합계: LCL(713) + IDEAL(219) + Borealis(15) + SMART(5) + Sceaux(1) = 953
# - 총합: 955 + 953 = 1,908
#
# BB 공식 기준: "eval over all real buildings for all available years" (BB README)
# zero_shot.py의 `if count == 10: break`는 데모용, 논문 결과는 전체 건물 기준
BB_FILTER = {
    "bdg2": {
        "rule": "per-year 독립 로딩, 빈파일 스킵, 연도별 NaN>50% 제외",
        "sites": ["Bear", "Fox", "Panther", "Rat"],
        "nan_threshold_per_year": 0.50,
        "expected_count": 611,
    },
    "electricity": {
        "rule": "2011~2014 모든 연도 합집합, OOV 제외",
        "nan_threshold_per_year": 0.50,
        "expected_count": 344,
    },
    "commercial_total": 955,
    "residential_total": 953,
    "total": 1908,
}


# ============================================================
# BB Box-Cox Transform (공식 pickle 사용)
# ============================================================

class BBBoxCoxTransform:
    """BB 공식 Box-Cox transform wrapper"""

    def __init__(self, transform_path):
        pkl_path = transform_path / 'boxcox.pkl'
        with open(pkl_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"  BB Box-Cox loaded: lambda={self.scaler.lambdas_[0]:.5f}")

    def transform(self, x):
        """numpy or tensor → transformed"""
        if isinstance(x, torch.Tensor):
            device = x.device
            shape = x.shape
            x_np = x.cpu().numpy().reshape(-1, 1)
            x_np = np.clip(x_np, 1e-6, None)  # BB uses 1e-6 offset
            transformed = self.scaler.transform(x_np).reshape(shape)
            return torch.from_numpy(transformed).float().to(device)
        else:
            x = np.clip(x, 1e-6, None)
            return self.scaler.transform(x.reshape(-1, 1)).reshape(x.shape)

    def inverse_transform(self, x):
        """transformed → original scale"""
        if isinstance(x, torch.Tensor):
            device = x.device
            shape = x.shape
            x_np = x.cpu().numpy().reshape(-1, 1)
            inv = self.scaler.inverse_transform(x_np).reshape(shape)
            return torch.from_numpy(inv).float().to(device)
        else:
            return self.scaler.inverse_transform(x.reshape(-1, 1)).reshape(x.shape)


# ============================================================
# BB Dataset Parser
# ============================================================

def load_benchmark_config():
    """benchmark.toml 로드"""
    with open(BB_DATA_DIR / 'metadata' / 'benchmark.toml', 'rb') as f:
        return tomli.load(f)


def load_oov_list():
    """OOV 건물 목록 로드"""
    oov = []
    oov_path = BB_DATA_DIR / 'metadata' / 'oov.txt'
    if oov_path.exists():
        with open(oov_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    oov.append(parts[1])  # building_id
    return oov


def parse_bb_buildings():
    """BB 테스트 데이터에서 건물별 시계열 추출

    Returns:
        list of dict: [{building_id, dataset, building_type, latlon, series}]
    """
    config = load_benchmark_config()['buildings_bench']
    oov = load_oov_list()
    buildings = []

    # --- BDG-2 (multi-building CSV per site) ---
    # 로딩 방식: per-year 독립 로딩, 빈파일 스킵, 연도별 NaN>50% 제외
    # Bear + Fox + Panther + Rat = 611
    bdg2_config = config.get('bdg-2', {})
    for site in ['bear', 'fox', 'panther', 'rat']:
        site_config = bdg2_config.get(site, {})
        latlon = site_config.get('latlon', [0, 0])
        btype = site_config.get('building_type', 'commercial')

        site_cap = site.capitalize()
        csv_files = sorted(BB_DATA_DIR.glob(f'BDG-2/{site_cap}_clean=*.csv'))
        if not csv_files:
            continue

        per_building = {}  # bid → list of per-year series arrays
        per_building_ts = {}  # bid → list of per-year DatetimeIndex
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
            if df.shape[1] == 0:  # 빈 파일 스킵 (e.g., Panther_clean=2016.csv)
                continue
            for col in df.columns:
                if col in oov:
                    continue
                s = df[col].values.astype(np.float64)
                if np.isnan(s).mean() > BB_FILTER['bdg2']['nan_threshold_per_year']:
                    continue
                per_building.setdefault(col, []).append(np.nan_to_num(s, nan=0.0))
                per_building_ts.setdefault(col, []).append(df.index)

        for bid in sorted(per_building.keys()):
            series = np.concatenate(per_building[bid])
            # 실제 캘린더 timestamps (raw integers, normalized in __getitem__)
            idx_list = per_building_ts[bid]
            full_idx = idx_list[0]
            for ix in idx_list[1:]:
                full_idx = full_idx.append(ix)
            doy = (full_idx.dayofyear.values - 1).astype(np.int32)  # 0-364
            dow = full_idx.dayofweek.values.astype(np.int32)         # 0-6
            hod = full_idx.hour.values.astype(np.int32)              # 0-23
            buildings.append({
                'building_id': bid,
                'dataset': 'BDG-2',
                'building_type': btype,
                'latlon': latlon,
                'series': series,
                'timestamps': (doy, dow, hod),
            })

    # --- Electricity (multi-building CSV) ---
    # 로딩 방식: per-year 독립 로딩, 2011~2014 모든 연도 합집합, OOV 제외 = 344
    elec_config = config.get('electricity', {})
    latlon = elec_config.get('latlon', [0, 0])
    btype = elec_config.get('building_type', 'commercial')

    csv_files = sorted(BB_DATA_DIR.glob('Electricity/LD2011_2014_clean=*.csv'))
    per_building = {}  # bid → list of per-year series arrays
    per_building_ts = {}  # bid → list of per-year DatetimeIndex
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
        for col in df.columns:
            bid = f"Electricity {col}"
            if col in oov or bid in oov:
                continue
            s = df[col].values.astype(np.float64)
            if np.isnan(s).mean() > BB_FILTER['electricity']['nan_threshold_per_year']:
                continue
            per_building.setdefault(bid, []).append(np.nan_to_num(s, nan=0.0))
            per_building_ts.setdefault(bid, []).append(df.index)

    for bid in sorted(per_building.keys()):
        series = np.concatenate(per_building[bid])
        idx_list = per_building_ts[bid]
        full_idx = idx_list[0]
        for ix in idx_list[1:]:
            full_idx = full_idx.append(ix)
        doy = (full_idx.dayofyear.values - 1).astype(np.int32)
        dow = full_idx.dayofweek.values.astype(np.int32)
        hod = full_idx.hour.values.astype(np.int32)
        buildings.append({
            'building_id': bid,
            'dataset': 'Electricity',
            'building_type': btype,
            'latlon': latlon,
            'series': series,
            'timestamps': (doy, dow, hod),
        })

    # --- Single-building datasets (IDEAL, LCL, SMART, Sceaux, Borealis) ---
    single_datasets = {
        'ideal': ('IDEAL', 'home'),
        'lcl': ('LCL', 'MAC'),
        'smart': ('SMART', 'Home'),
        'sceaux': ('Sceaux', 'Sceaux'),
        'borealis': ('Borealis', 'home'),
    }

    for ds_key, (ds_dir, prefix) in single_datasets.items():
        ds_config = config.get(ds_key, {})
        latlon = ds_config.get('latlon', [0, 0])
        btype = ds_config.get('building_type', 'residential')

        ds_path = BB_DATA_DIR / ds_dir
        if not ds_path.exists():
            continue

        # 건물별 CSV 파일 그룹화
        csv_files = sorted(ds_path.glob('*_clean=*.csv'))
        building_groups = {}
        for csv_file in csv_files:
            # filename: home100_clean=2017.csv or MAC000005_clean=2012.csv
            bid = csv_file.stem.split('_clean=')[0]
            if bid not in building_groups:
                building_groups[bid] = []
            building_groups[bid].append(csv_file)

        for bid, files in building_groups.items():
            if bid in oov:
                continue

            dfs = []
            for f in sorted(files):
                # 데이터셋마다 CSV 형식이 다름
                # IDEAL: unnamed index + 'power'
                # LCL: 'DateTime' + 'power'
                # Borealis/SMART: 'timestamp' + 'power'
                # Sceaux: 'timestamp' + 'Global_active_power'
                df = pd.read_csv(f)
                # timestamp 컬럼 찾기
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                elif 'DateTime' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    df = df.set_index('DateTime')
                else:
                    # 첫 번째 컬럼이 unnamed (IDEAL 등)
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df = df.set_index(df.columns[0])
                dfs.append(df)
            combined = pd.concat(dfs).sort_index()

            # power 컬럼 찾기
            if 'power' in combined.columns:
                series = combined['power'].values.astype(np.float64)
            elif 'Global_active_power' in combined.columns:
                series = combined['Global_active_power'].values.astype(np.float64)
            elif len(combined.columns) == 1:
                series = combined.iloc[:, 0].values.astype(np.float64)
            else:
                series = combined.select_dtypes(include=[np.number]).iloc[:, 0].values.astype(np.float64)

            if np.isnan(series).sum() / len(series) > 0.1:
                continue
            series = np.nan_to_num(series, nan=0.0)

            doy = (combined.index.dayofyear.values - 1).astype(np.int32)
            dow = combined.index.dayofweek.values.astype(np.int32)
            hod = combined.index.hour.values.astype(np.int32)

            buildings.append({
                'building_id': bid,
                'dataset': ds_dir,
                'building_type': btype,
                'latlon': latlon,
                'series': series,
                'timestamps': (doy, dow, hod),
            })

    return buildings


# ============================================================
# Sliding Window Dataset for a single building
# ============================================================

class BBBuildingDataset(Dataset):
    """단일 건물의 sliding window dataset"""

    def __init__(self, series, latlon, building_type_int, context_len, pred_len,
                 boxcox_transform, timestamps=None):
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len
        self.boxcox = boxcox_transform
        self.latlon = latlon
        self.btype_int = building_type_int

        n = len(series)
        starts = list(range(0, n - self.total_len + 1, pred_len))

        # 시계열 → sliding windows (Bug fix: 항상 clip, 조건부 아님)
        self.windows = []
        for start in starts:
            window = np.clip(series[start:start + self.total_len], 1e-6, None)
            self.windows.append(window)

        # Timestamps (Bug fix: 실제 캘린더 timestamps 사용)
        self.timestamps = []
        if timestamps is not None:
            doy_1d, dow_1d, hod_1d = timestamps
            for start in starts:
                self.timestamps.append((
                    doy_1d[start:start + self.total_len],
                    dow_1d[start:start + self.total_len],
                    hod_1d[start:start + self.total_len],
                ))
        else:
            # fallback 순차 (timestamps 없을 때만 — 정상 사용 시 발생하지 않음)
            for start in starts:
                hours = np.arange(start, start + self.total_len)
                self.timestamps.append(((hours // 24) % 365, (hours // 24) % 7, hours % 24))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx].copy()
        day_of_year, day_of_week, hour_of_day = self.timestamps[idx]

        # Box-Cox transform
        load = self.boxcox.transform(window)

        # Normalize timestamps → [-1, +1] (BB 방식)
        doy_norm = (day_of_year / 364.0) * 2 - 1
        dow_norm = (day_of_week / 6.0) * 2 - 1
        hod_norm = (hour_of_day / 23.0) * 2 - 1

        # LatLon: 학습(Korean 기준 정규화)과 BB 평가 데이터(US CONUS 가짜 좌표) 모두
        # 좌표가 의미 없으므로 0으로 고정 (train.py fast eval과 동일하게 맞춤)
        # BUG FIX (2026-03-16): 이전 코드는 US rough center 공식 사용 → 학습 시와 불일치
        lat_norm = 0.0  # 학습 시 ignore_spatial=True → 0 효과와 동일하게 맞춤
        lon_norm = 0.0

        seq_len = self.total_len
        return {
            'load': torch.FloatTensor(load).unsqueeze(-1),           # (seq, 1)
            'latitude': torch.full((seq_len, 1), lat_norm, dtype=torch.float32),
            'longitude': torch.full((seq_len, 1), lon_norm, dtype=torch.float32),
            'building_type': torch.full((seq_len, 1), self.btype_int, dtype=torch.long),
            'day_of_year': torch.FloatTensor(doy_norm).unsqueeze(-1),
            'day_of_week': torch.FloatTensor(dow_norm).unsqueeze(-1),
            'hour_of_day': torch.FloatTensor(hod_norm).unsqueeze(-1),
        }


# ============================================================
# Per-building evaluation
# ============================================================

@torch.no_grad()
def evaluate_building(model, dataset, device, boxcox, context_len):
    """단일 건물 → CVRMSE"""
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_se = []
    all_gt = []

    model.eval()
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            predictions, _ = model.predict(batch)
        targets = batch['load'][:, context_len:]

        # Inverse Box-Cox
        pred_inv = boxcox.inverse_transform(predictions)
        tgt_inv = boxcox.inverse_transform(targets)

        pred_np = pred_inv.squeeze(-1).cpu().numpy()
        tgt_np = tgt_inv.squeeze(-1).cpu().numpy()

        all_se.append((pred_np - tgt_np) ** 2)
        all_gt.append(np.abs(tgt_np))

    se = np.concatenate(all_se)
    gt = np.concatenate(all_gt)
    mean_gt = np.mean(gt)
    if mean_gt < 1e-8:
        return None

    cvrmse = float(np.sqrt(np.mean(se)) / mean_gt)
    return {
        'cvrmse': cvrmse,
        'n_windows': len(se),
        'mean_actual': float(mean_gt),
    }


# ============================================================
# Bootstrap CI
# ============================================================

def bootstrap_ci(values, n_reps=50000, ci=0.95):
    rng = np.random.default_rng(42)
    values = values[~np.isnan(values)]  # NaN 제거
    n = len(values)
    medians = np.array([np.nanmedian(values[rng.integers(0, n, size=n)]) for _ in range(n_reps)])
    alpha = (1 - ci) / 2
    return float(np.nanmedian(values)), float(np.percentile(medians, alpha * 100)), float(np.percentile(medians, (1 - alpha) * 100))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate on BB Official Test Set')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--bootstrap_reps', type=int, default=50000)
    parser.add_argument('--cpu', action='store_true', help='Force CPU (when GPU busy with training)')
    parser.add_argument('--commercial_only', action='store_true', help='Evaluate commercial buildings only (955 buildings, skip residential)')
    parser.add_argument('--bdg2_only', action='store_true', help='Evaluate BDG-2 only (611 buildings) — experiment comparison standard')
    args = parser.parse_args()

    # ---- Config ----
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    model_args = config['model']
    context_len = model_args['context_len']
    pred_len = model_args['pred_len']

    # ---- Device ----
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Model ----
    model = model_factory(model_args)
    model.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {args.checkpoint}")

    # ---- Box-Cox transform ----
    # Use training data's boxcox to match checkpoint normalization.
    # BB's S3 v2.0.0 boxcox (lambda=-0.07064) differs from the checkpoint's (lambda=-0.06722).
    train_bc_path = Path(__file__).parent.parent / 'data' / 'korean_bb' / 'metadata' / 'transforms'
    if train_bc_path.exists():
        boxcox = BBBoxCoxTransform(train_bc_path)
    else:
        boxcox = BBBoxCoxTransform(BB_DATA_DIR / 'metadata' / 'transforms')

    # ---- Parse BB buildings ----
    print("\nParsing BB test datasets...")
    buildings = parse_bb_buildings()
    print(f"  Total buildings: {len(buildings)}")

    by_dataset = {}
    for b in buildings:
        ds = b['dataset']
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
    for ds, cnt in sorted(by_dataset.items()):
        print(f"    {ds}: {cnt}")

    # ---- 건물 수 검증 (불변 기준) ----
    n_bdg2 = sum(1 for b in buildings if b['dataset'] == 'BDG-2')
    n_elec = sum(1 for b in buildings if b['dataset'] == 'Electricity')
    n_commercial = sum(1 for b in buildings if b['building_type'] != 'residential')
    exp_bdg2 = BB_FILTER['bdg2']['expected_count']
    exp_elec = BB_FILTER['electricity']['expected_count']
    exp_com = BB_FILTER['commercial_total']
    if n_bdg2 != exp_bdg2 or n_elec != exp_elec or n_commercial != exp_com:
        raise AssertionError(
            f"[건물 수 불일치] BDG-2: {n_bdg2} (기대 {exp_bdg2}), "
            f"Electricity: {n_elec} (기대 {exp_elec}), "
            f"Commercial합계: {n_commercial} (기대 {exp_com}). "
            f"BB_FILTER 기준이 변경되었거나 데이터 파일이 손상되었습니다."
        )
    print(f"  [검증 통과] BDG-2={n_bdg2} ✓  Electricity={n_elec} ✓  상업합계={n_commercial} ✓")

    # ---- 필터 ----
    if args.bdg2_only:
        buildings = [b for b in buildings if b['dataset'] == 'BDG-2']
        print(f"  [bdg2_only] Filtered to {len(buildings)} BDG-2 buildings")
    elif args.commercial_only:
        buildings = [b for b in buildings if b['building_type'] != 'residential']
        print(f"  [commercial_only] Filtered to {len(buildings)} commercial buildings")

    # ---- Per-building evaluation ----
    print(f"\nEvaluating {len(buildings)} buildings...")
    results_rows = []
    t0 = time.time()

    for i, bld in enumerate(buildings):
        btype_int = 0 if bld['building_type'] == 'residential' else 1

        try:
            dataset = BBBuildingDataset(
                series=bld['series'],
                latlon=bld['latlon'],
                building_type_int=btype_int,
                context_len=context_len,
                pred_len=pred_len,
                boxcox_transform=boxcox,
                timestamps=bld.get('timestamps'),
            )

            if len(dataset) == 0:
                continue

            metrics = evaluate_building(model, dataset, device, boxcox, context_len)
            if metrics is None:
                continue

            results_rows.append({
                'building_id': bld['building_id'],
                'dataset': bld['dataset'],
                'building_type': bld['building_type'],
                'metric': 'cvrmse',
                'value': metrics['cvrmse'],
                'mean_actual': metrics['mean_actual'],
                'n_windows': metrics['n_windows'],
            })
        except Exception as e:
            print(f"  [SKIP] {bld['building_id']} ({bld['dataset']}): {e}", flush=True)
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1:>5}/{len(buildings)}] {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  Complete: {len(results_rows)} buildings ({elapsed:.0f}s)", flush=True)

    if not results_rows:
        print("[ERROR] No valid results")
        return

    # ---- Results ----
    results_df = pd.DataFrame(results_rows)

    results_dir = PROJECT_DIR / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.config).stem
    csv_path = results_dir / f'bb_metrics_{model_name}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # ---- Aggregate ----
    print(f"\n{'=' * 70}")
    print(f"BB Official Test Set — {model_name}")
    print(f"{'=' * 70}")

    for btype in ['residential', 'commercial', 'all']:
        if btype == 'all':
            subset = results_df
        else:
            subset = results_df[results_df['building_type'] == btype]

        if len(subset) == 0:
            continue

        values = subset['value'].values
        median, ci_low, ci_high = bootstrap_ci(values, n_reps=args.bootstrap_reps)

        print(f"\n  {btype.upper()} ({len(subset)} buildings)")
        print(f"    CVRMSE median: {median * 100:.2f}% "
              f"(95% CI: {ci_low * 100:.2f}% - {ci_high * 100:.2f}%)")
        print(f"    CVRMSE mean:   {values.mean() * 100:.2f}%")

    # ---- BB SOTA 비교 ----
    print(f"\n  {'=' * 50}")
    print(f"  BB SOTA Comparison (Commercial)")
    print(f"  {'=' * 50}")

    com_df = results_df[results_df['building_type'] == 'commercial']
    if len(com_df) > 0:
        com_median = np.nanmedian(com_df['value'].values) * 100
        print(f"  Our model ({model_name}):     {com_median:.2f}%", flush=True)
        print(f"  BB Transformer-M (SOTA):      13.27%  (reproduced)")
        print(f"  BB Transformer-L:             13.31%  (paper)")
        print(f"  Persistence Ensemble:         16.68%")
        print(f"  Gap vs SOTA-M:                {com_median - 13.27:+.2f}%p")

    res_df = results_df[results_df['building_type'] == 'residential']
    if len(res_df) > 0:
        res_median = np.nanmedian(res_df['value'].values) * 100
        print(f"\n  {'=' * 50}")
        print(f"  BB SOTA Comparison (Residential)")
        print(f"  {'=' * 50}")
        print(f"  Our model ({model_name}):     {res_median:.2f}%")
        print(f"  BB Transformer-L (SOTA):      40.80%")
        print(f"  Persistence Ensemble:         37.58%")

    # Dataset별 분석
    print(f"\n  {'=' * 50}")
    print(f"  Per-Dataset Breakdown")
    print(f"  {'=' * 50}")
    print(f"  {'Dataset':<15} {'Type':<12} {'N':>5} {'Median CVRMSE':>15}")
    print(f"  {'-' * 50}")
    for ds in sorted(results_df['dataset'].unique()):
        ds_df = results_df[results_df['dataset'] == ds]
        btype = ds_df['building_type'].iloc[0]
        med = np.nanmedian(ds_df['value'].values)
        print(f"  {ds:<15} {btype:<12} {len(ds_df):>5} {med * 100:>14.2f}%")

    print(f"\n{'=' * 70}")


if __name__ == '__main__':
    main()
