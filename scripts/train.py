"""Korean_BB 모델 학습 — 단일 GPU, BB 프로토콜

BB pretrain.py 기반이지만 DDP 없이 단일 GPU + 에폭 기반.

Features:
  - TOML 설정 파일 → 모델/학습 파라미터
  - KoreanBBPretrainingDataset (train/val)
  - Box-Cox transform (학습시 transform, 평가시 inverse)
  - Mixed precision (torch.amp)
  - Cosine schedule with warmup
  - Gaussian NLL loss
  - 매 에폭 검증 (NRMSE by building type)
  - 체크포인트 (best + last)
  - TensorBoard 로깅
  - Early stopping
  - Graceful stop (Ctrl+C 또는 stop 파일 → 현재 epoch 완료 후 체크포인트 저장 & 종료)

Usage:
    # M 모델 학습
    python scripts/train.py --config configs/model/TransformerWithGaussian-M.toml

    # office만 빠른 테스트
    python scripts/train.py --config configs/model/TransformerWithGaussian-M.toml \\
        --filter office --epochs 5

    # 체크포인트에서 재개
    python scripts/train.py --config configs/model/TransformerWithGaussian-M.toml \\
        --resume checkpoints/last.pt

    # 학습 중 graceful stop:
    #   방법 1) Ctrl+C → 현재 epoch 완료 후 체크포인트 저장 & 종료
    #   방법 2) touch checkpoints/STOP → 다음 epoch 끝에서 체크포인트 저장 & 종료
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import os
import pickle
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tomli
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.data.korean_dataset import KoreanBBPretrainingDataset
from src.models.transformer import model_factory


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data' / 'korean_bb'
CHECKPOINT_DIR = PROJECT_DIR / 'checkpoints'
STOP_FILE = CHECKPOINT_DIR / 'STOP'
BB_DATA_DIR = PROJECT_DIR / 'external' / 'BuildingsBench_data'


# ============================================================
# BB Fast Evaluation (per-epoch BB benchmark, no bootstrap)
# ============================================================

class _BBBoxCox:
    """BB 공식 Box-Cox transform (lambda=-0.067)"""
    def __init__(self):
        pkl_path = BB_DATA_DIR / 'metadata' / 'transforms' / 'boxcox.pkl'
        with open(pkl_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x.reshape(-1, 1), 1e-6, None)
        return self.scaler.transform(x).reshape(-1)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(x.reshape(-1, 1)).reshape(-1)


def _load_bb_buildings(commercial_only: bool = True, bdg2_only: bool = False):
    """BB 테스트 데이터 로드 → list of (building_id, building_type, series_np, doy_np, dow_np, hod_np)

    공식 BB zero_shot.py 동일 기준: "eval over all real buildings for all available years"
      - 연도별 독립 로딩, 빈 파일 스킵, 연도별 NaN>50% 만 제외
      - commercial_only=True: BDG-2(611) + Electricity(344) ≈ 955건 (OOV 제외)
      - commercial_only=False: +residential 953건 ≈ 1,908건
      - bdg2_only=True: BDG-2(611)만 로드 (inline eval 전용 — Electricity 제외)
    evaluate_bb.py와 동일 기준 (두 함수 완전 일치)
    타임스탬프: 실제 날짜 기반 doy/dow/hod (float32 [-1,1]) 저장 → inline eval 정확도 향상
    """
    if not (BB_DATA_DIR / 'metadata' / 'benchmark.toml').exists():
        return []

    with open(BB_DATA_DIR / 'metadata' / 'benchmark.toml', 'rb') as f:
        _cfg = tomli.load(f)['buildings_bench']

    oov = set()
    oov_path = BB_DATA_DIR / 'metadata' / 'oov.txt'
    if oov_path.exists():
        with open(oov_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    oov.add(parts[1])

    buildings = []

    # BDG-2 (commercial) — 공식 BB: 연도별 독립 로딩
    # 빈 연도 파일 스킵 (Panther_clean=2016.csv), 연도별 NaN>50% 제외
    # → Panther 105건 포함, 전체 BDG-2 611건 (OOV 제외)
    for site in ['bear', 'fox', 'panther', 'rat']:
        files = sorted(BB_DATA_DIR.glob(f'BDG-2/{site.capitalize()}_clean=*.csv'))
        if not files:
            continue
        per_building: dict = {}
        per_building_ts: dict = {}
        for f in files:
            df = pd.read_csv(f, parse_dates=['timestamp'], index_col='timestamp')
            if df.shape[1] == 0:  # 빈 파일 스킵
                continue
            for col in df.columns:
                if col in oov:
                    continue
                s = df[col].values.astype(np.float64)
                if np.isnan(s).mean() > 0.5:  # 연도별 NaN>50% 스킵
                    continue
                per_building.setdefault(col, []).append(np.nan_to_num(s, nan=0.0))
                per_building_ts.setdefault(col, []).append(df.index)
        for bid in sorted(per_building.keys()):
            series = np.concatenate(per_building[bid])
            idx_list = per_building_ts[bid]
            full_idx = idx_list[0]
            for ix in idx_list[1:]:
                full_idx = full_idx.append(ix)
            doy = ((full_idx.dayofyear - 1) / 364.0 * 2 - 1).values.astype(np.float32)
            dow = (full_idx.dayofweek / 6.0 * 2 - 1).values.astype(np.float32)
            hod = (full_idx.hour / 23.0 * 2 - 1).values.astype(np.float32)
            buildings.append((bid, 'commercial', series, doy, dow, hod))

    if bdg2_only:
        return buildings  # inline eval: BDG-2 611건만

    # Electricity (commercial) — 공식 BB: 연도별 독립 로딩
    # 2011년 없는 건물도 2012~2014 포함 → 344건 (OOV 제외)
    files = sorted(BB_DATA_DIR.glob('Electricity/LD2011_2014_clean=*.csv'))
    per_building_e: dict = {}
    per_building_e_ts: dict = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=['timestamp'], index_col='timestamp')
        for col in df.columns:
            bid = f'Electricity {col}'
            if col in oov or bid in oov:
                continue
            s = df[col].values.astype(np.float64)
            if np.isnan(s).mean() > 0.5:  # 연도별 NaN>50% 스킵
                continue
            per_building_e.setdefault(bid, []).append(np.nan_to_num(s, nan=0.0))
            per_building_e_ts.setdefault(bid, []).append(df.index)
    for bid in sorted(per_building_e.keys()):
        series = np.concatenate(per_building_e[bid])
        idx_list = per_building_e_ts[bid]
        full_idx = idx_list[0]
        for ix in idx_list[1:]:
            full_idx = full_idx.append(ix)
        doy = ((full_idx.dayofyear - 1) / 364.0 * 2 - 1).values.astype(np.float32)
        dow = (full_idx.dayofweek / 6.0 * 2 - 1).values.astype(np.float32)
        hod = (full_idx.hour / 23.0 * 2 - 1).values.astype(np.float32)
        buildings.append((bid, 'commercial', series, doy, dow, hod))

    if commercial_only:
        return buildings

    # Residential
    for ds_dir in ['IDEAL', 'LCL', 'SMART', 'Sceaux', 'Borealis']:
        ds_path = BB_DATA_DIR / ds_dir
        if not ds_path.exists():
            continue
        groups: dict = {}
        for f in sorted(ds_path.glob('*_clean=*.csv')):
            bid = f.stem.split('_clean=')[0]
            if bid not in oov:
                groups.setdefault(bid, []).append(f)
        for bid, flist in groups.items():
            dfs = []
            for f in sorted(flist):
                df = pd.read_csv(f)
                for tc in ['timestamp', 'DateTime']:
                    if tc in df.columns:
                        df[tc] = pd.to_datetime(df[tc])
                        df = df.set_index(tc)
                        break
                else:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df = df.set_index(df.columns[0])
                dfs.append(df)
            combined = pd.concat(dfs).sort_index()
            if 'power' in combined.columns:
                s = combined['power'].values
            elif 'Global_active_power' in combined.columns:
                s = combined['Global_active_power'].values
            else:
                s = combined.select_dtypes(include=[np.number]).iloc[:, 0].values
            s = s.astype(np.float64)
            if np.isnan(s).mean() > 0.1:
                continue
            ts_idx = combined.index
            doy = ((ts_idx.dayofyear - 1) / 364.0 * 2 - 1).values.astype(np.float32)
            dow = (ts_idx.dayofweek / 6.0 * 2 - 1).values.astype(np.float32)
            hod = (ts_idx.hour / 23.0 * 2 - 1).values.astype(np.float32)
            buildings.append((bid, 'residential', np.nan_to_num(s), doy, dow, hod))

    return buildings


@torch.no_grad()
def _bb_fast_eval(model, device, context_len, pred_len, bb_buildings, bb_boxcox,
                  commercial_only: bool = True, batch_size: int = 512,
                  window_chunk_size: int = 128):
    """BB 건물들에 대한 fast CVRMSE 계산 (bootstrap 없음, per-epoch 용도)

    전략: 건물별 처리 — numpy fancy indexing으로 윈도우 생성 (Python loop 불필요)
    타임스탬프는 per-building 1D 배열에서 on-the-fly 슬라이싱 → 메모리 1/192로 절감
    (N_total, T) 대형 concat 배열 생성 안 함

    Returns:
        dict: {commercial_cvrmse, n_commercial}
        None: BB 데이터 없음
    """
    model.eval()
    total_len = context_len + pred_len
    target_types = ['commercial'] if commercial_only else ['commercial', 'residential']
    results = {}
    T = total_len

    for target_type in target_types:
        btype_int = 1 if target_type == 'commercial' else 0
        btype_blds = [(bid, s, doy, dow, hod)
                      for bid, bt, s, doy, dow, hod in bb_buildings if bt == target_type]
        if not btype_blds:
            continue

        cvrmses = []

        for bid, series, doy_1d, dow_1d, hod_1d in btype_blds:
            n = len(series)
            starts = np.arange(0, n - total_len + 1, pred_len)
            if len(starts) == 0:
                continue
            time_offsets = np.arange(total_len)
            sum_sq_err = 0.0
            sum_abs_gt = 0.0
            n_targets = 0

            for chunk_start in range(0, len(starts), window_chunk_size):
                chunk_starts = starts[chunk_start:chunk_start + window_chunk_size]
                idx = chunk_starts[:, None] + time_offsets
                wins = np.clip(series[idx], 1e-6, None).astype(np.float32, copy=False)
                wins_bc = bb_boxcox.scaler.transform(
                    wins.reshape(-1, 1)).reshape(wins.shape).astype(np.float32, copy=False)
                doy_w = doy_1d[idx]
                dow_w = dow_1d[idx]
                hod_w = hod_1d[idx]

                for i in range(0, len(chunk_starts), batch_size):
                    bw = wins_bc[i:i + batch_size]
                    B = len(bw)
                    load_t = torch.from_numpy(bw).unsqueeze(-1).float().to(device)
                    zeros = torch.zeros(B, T, 1, device=device)
                    batch_d = {
                        'load': load_t,
                        'latitude': zeros,
                        'longitude': zeros,
                        'building_type': torch.full((B, T, 1), btype_int,
                                                    dtype=torch.long, device=device),
                        'day_of_year': torch.from_numpy(
                            doy_w[i:i + B]).unsqueeze(-1).to(device),
                        'day_of_week': torch.from_numpy(
                            dow_w[i:i + B]).unsqueeze(-1).to(device),
                        'hour_of_day': torch.from_numpy(
                            hod_w[i:i + B]).unsqueeze(-1).to(device),
                    }
                    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                        preds, _ = model.predict(batch_d)
                    pred_bc = preds.squeeze(-1).detach().cpu().numpy()
                    tgt_bc = wins_bc[i:i + B, context_len:]
                    # Clip pred in BC space: z=5 → 3.8e7 kWh (safe for float64 square)
                    pred_bc_safe = np.clip(pred_bc, -5.0, 5.0)
                    pred_orig = bb_boxcox.scaler.inverse_transform(
                        pred_bc_safe.reshape(-1, 1)).reshape(pred_bc_safe.shape)
                    tgt_orig = bb_boxcox.scaler.inverse_transform(
                        tgt_bc.reshape(-1, 1)).reshape(tgt_bc.shape)
                    se = (pred_orig - tgt_orig) ** 2
                    sum_sq_err += float(se.sum())
                    sum_abs_gt += float(np.abs(tgt_orig).sum())
                    n_targets += int(tgt_orig.size)

            if n_targets == 0:
                continue
            mean_gt = sum_abs_gt / n_targets
            if mean_gt < 1e-8:
                continue
            cvrmse_val = float(np.sqrt(sum_sq_err / n_targets) / mean_gt)
            if not np.isfinite(cvrmse_val):
                continue
            cvrmses.append(cvrmse_val)
            # BUG-G: release GPU cache periodically to prevent OOM on large buildings
            if len(cvrmses) % 50 == 0:
                torch.cuda.empty_cache()
            continue

            # numpy fancy indexing — Python loop 불필요 (N_b, T)
            idx = starts[:, None] + np.arange(total_len)
            wins = np.clip(series[idx].astype(np.float64), 1e-6, None)
            wins_bc = bb_boxcox.scaler.transform(
                wins.reshape(-1, 1)).reshape(wins.shape).astype(np.float32)
            # 타임스탬프: per-building 1D 배열 슬라이싱 (N_b, T) — on-the-fly
            doy_w = doy_1d[idx]   # (N_b, T)
            dow_w = dow_1d[idx]   # (N_b, T)
            hod_w = hod_1d[idx]   # (N_b, T)

            N_b = len(starts)
            all_pred = np.empty((N_b, pred_len), dtype=np.float32)

            for i in range(0, N_b, batch_size):
                bw = wins_bc[i:i + batch_size]
                B = len(bw)
                load_t = torch.from_numpy(bw).unsqueeze(-1).float().to(device)
                zeros = torch.zeros(B, T, 1, device=device)
                batch_d = {
                    'load': load_t,
                    'latitude': zeros,
                    'longitude': zeros,
                    'building_type': torch.full((B, T, 1), btype_int,
                                                dtype=torch.long, device=device),
                    'day_of_year': torch.from_numpy(
                        doy_w[i:i + B]).unsqueeze(-1).to(device),
                    'day_of_week': torch.from_numpy(
                        dow_w[i:i + B]).unsqueeze(-1).to(device),
                    'hour_of_day': torch.from_numpy(
                        hod_w[i:i + B]).unsqueeze(-1).to(device),
                }
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    preds, _ = model.predict(batch_d)   # (B, 24, 1)
                all_pred[i:i + B] = preds.squeeze(-1).cpu().numpy()

            # Per-building CVRMSE
            tgt_bc = wins_bc[:, context_len:]       # (N_b, 24)
            all_pred_safe = np.clip(all_pred, -5.0, 5.0)
            pred_orig = bb_boxcox.scaler.inverse_transform(
                all_pred_safe.reshape(-1, 1)).reshape(all_pred_safe.shape)
            tgt_orig = bb_boxcox.scaler.inverse_transform(
                tgt_bc.reshape(-1, 1)).reshape(tgt_bc.shape)
            mean_gt = float(np.mean(np.abs(tgt_orig)))
            if mean_gt < 1e-8:
                continue
            se = (pred_orig - tgt_orig) ** 2
            cvrmse_val = float(np.sqrt(np.mean(se)) / mean_gt)
            if not np.isfinite(cvrmse_val):
                continue
            cvrmses.append(cvrmse_val)

        if cvrmses:
            results[f'{target_type}_cvrmse'] = float(np.median(cvrmses))
            results[f'n_{target_type}'] = len(cvrmses)

    model.train()
    return results if results else None


# ============================================================
# Graceful Stop
# ============================================================

_stop_requested = False


def _signal_handler(signum, frame):
    """Ctrl+C → 현재 epoch 완료 후 체크포인트 저장 & 종료"""
    global _stop_requested
    if _stop_requested:
        print("\n  [STOP] 강제 종료 (Ctrl+C x2)")
        sys.exit(1)
    _stop_requested = True
    print("\n  [STOP] 요청 접수 — 현재 epoch 완료 후 체크포인트 저장 & 종료합니다...")


def _check_stop() -> bool:
    """stop 신호 확인 (Ctrl+C 또는 STOP 파일)"""
    if _stop_requested:
        return True
    if STOP_FILE.exists():
        STOP_FILE.unlink()  # 파일 삭제 (다음 실행에 영향 안 주도록)
        print("  [STOP] STOP 파일 감지 — 체크포인트 저장 & 종료합니다...")
        return True
    return False


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, val_loader, device, load_transform, context_len):
    """매 에폭 검증: loss + NRMSE (building type별)

    Returns:
        dict with val_loss, nrmse_residential, nrmse_commercial, nrmse_all
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Per-building-type 에러 수집
    all_se = {'residential': [], 'commercial': []}
    all_gt = {'residential': [], 'commercial': []}

    for batch in val_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)

        # Targets (raw, for NRMSE)
        raw_targets = batch['load'][:, context_len:].clone()

        # Forward
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            preds = model(batch)
            targets = batch['load'][:, context_len:]
            loss = model.loss(preds, targets)

        total_loss += loss.item()
        n_batches += 1

        # Predictions for NRMSE (inverse transform)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            predictions, _ = model.predict(batch)
        if load_transform and predictions.numel() > 0:
            predictions = load_transform.undo_transform(predictions)
            raw_targets = load_transform.undo_transform(raw_targets)

        # Building type mask (0=residential, 1=commercial)
        btype_mask = batch['building_type'][:, 0, 0]  # (batch,)

        pred_np = predictions.squeeze(-1).cpu().numpy()
        tgt_np = raw_targets.squeeze(-1).cpu().numpy()

        res_mask = (btype_mask == 0).cpu().numpy()
        com_mask = (btype_mask == 1).cpu().numpy()

        if res_mask.any():
            se = (pred_np[res_mask] - tgt_np[res_mask]) ** 2
            all_se['residential'].append(se)
            all_gt['residential'].append(np.abs(tgt_np[res_mask]))
        if com_mask.any():
            se = (pred_np[com_mask] - tgt_np[com_mask]) ** 2
            all_se['commercial'].append(se)
            all_gt['commercial'].append(np.abs(tgt_np[com_mask]))

        if n_batches >= 200:  # 검증 시간 제한
            break

    model.train()

    # NRMSE = sqrt(mean(SE)) / mean(|actual|)
    results = {
        'val_loss': total_loss / max(n_batches, 1),
    }

    for btype in ['residential', 'commercial']:
        if all_se[btype]:
            se = np.concatenate(all_se[btype])
            gt = np.concatenate(all_gt[btype])
            nrmse = np.sqrt(np.mean(se)) / np.mean(gt)
            results[f'nrmse_{btype}'] = float(nrmse)
        else:
            results[f'nrmse_{btype}'] = float('nan')

    # Overall
    all_se_flat = []
    all_gt_flat = []
    for btype in ['residential', 'commercial']:
        all_se_flat.extend(all_se[btype])
        all_gt_flat.extend(all_gt[btype])
    if all_se_flat:
        se = np.concatenate(all_se_flat)
        gt = np.concatenate(all_gt_flat)
        results['nrmse_all'] = float(np.sqrt(np.mean(se)) / np.mean(gt))
    else:
        results['nrmse_all'] = float('nan')

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Korean_BB Transformer Training')
    parser.add_argument('--config', type=str, required=True,
                        help='TOML config path')
    parser.add_argument('--filter', type=str, default='',
                        help='building_id filter (e.g., office)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint path')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--note', type=str, default='',
                        help='Experiment note (appended to checkpoint name)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation (jitter, noise, scaling)')
    parser.add_argument('--bb_eval_interval', type=int, default=5,
                        help='BB benchmark eval every N epochs (0=disable). default=5')
    parser.add_argument('--train_index', type=str, default='train_weekly.csv',
                        help='Training index file name in data/korean_bb/metadata/ (default: train_weekly.csv)')
    parser.add_argument('--val_index', type=str, default='val_weekly.csv',
                        help='Validation index file name in data/korean_bb/metadata/ (default: val_weekly.csv)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Stop after N optimizer steps (sub-epoch training). '
                             'Overrides epochs. Runs val+BB eval at end.')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Override warmup_steps from config (useful for small-scale experiments)')
    parser.add_argument('--lru_cache', type=int, default=5000,
                        help='LRU cache size for parquet loading (default: 5000). '
                             'Increase for large caps (e.g. 50000 for 10K cap)')
    parser.add_argument('--preload', action='store_true',
                        help='Preload all parquet files into RAM at init (fast training, ~5GB RAM for 140K buildings). '
                             'Disables LRU cache. Recommended for scaling experiments.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override dataset dir (default: data/korean_bb). Used for BB sub-sampling experiments.')
    args = parser.parse_args()
    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        print(f'Using custom data_dir: {DATA_DIR}')

    # ---- Config ----
    with open(args.config, 'rb') as f:
        config = tomli.load(f)

    model_args = config['model']
    train_args = config['train']

    epochs = args.epochs or train_args.get('epochs', 50)
    batch_size = args.batch_size or train_args.get('batch_size', 64)
    lr = args.lr or train_args.get('lr', 6e-5)
    warmup_epochs = train_args.get('warmup_epochs', None)
    warmup_steps_cfg = train_args.get('warmup_steps', None)
    apply_scaler = train_args.get('apply_scaler_transform', 'boxcox')
    init_scale = train_args.get('init_scale', 0.02)
    grad_clip = train_args.get('grad_clip', 1.0)
    val_every = train_args.get('val_every', 1)
    patience = train_args.get('patience', 10)

    # ---- Seed + Deterministic (BUG-M fix) ----
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Dataset ----
    context_len = model_args['context_len']
    pred_len = model_args['pred_len']
    aug_str = " +augmentation" if args.augment else ""
    print(f"\nLoading datasets (filter='{args.filter}'){aug_str}...")
    train_dataset = KoreanBBPretrainingDataset(
        DATA_DIR,
        index_file=args.train_index,
        context_len=context_len,
        pred_len=pred_len,
        apply_scaler_transform=apply_scaler,
        filter_str=args.filter,
        augment=args.augment,
        lru_maxsize=args.lru_cache,
        preload=args.preload,
    )
    val_dataset = KoreanBBPretrainingDataset(
        DATA_DIR,
        index_file=args.val_index,
        context_len=context_len,
        pred_len=pred_len,
        apply_scaler_transform=apply_scaler,
        filter_str=args.filter,
        augment=False,  # val은 augmentation 없음
        lru_maxsize=args.lru_cache,
        preload=args.preload,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn(),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),  # 에포크 간 worker 유지 (재시작 오버헤드 제거)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn(),
        persistent_workers=(args.num_workers > 0),
    )

    load_transform = train_dataset.load_transform

    # ---- Model ----
    model = model_factory(model_args)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {Path(args.config).stem}")
    print(f"  Parameters: {n_params:,}")
    print(f"  d_model={model_args['d_model']}, layers={model_args['num_encoder_layers']}+{model_args['num_decoder_layers']}")

    # Weight init
    if not args.resume:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=init_scale)

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01,
    )

    steps_per_epoch = len(train_loader)

    # --max_steps: sub-epoch 학습 (BB 동등 학습량)
    max_steps = args.max_steps
    if max_steps is not None:
        total_steps = max_steps
        # epochs를 충분히 크게 설정 (max_steps에서 break)
        epochs = (max_steps // steps_per_epoch) + 2
    else:
        total_steps = epochs * steps_per_epoch

    # Warmup: CLI --warmup_steps > config warmup_steps > warmup_epochs > 기본 2 에폭
    if args.warmup_steps is not None:
        warmup_steps_cfg = args.warmup_steps
    if warmup_steps_cfg is not None:
        warmup_steps = warmup_steps_cfg
    elif warmup_epochs is not None:
        warmup_steps = warmup_epochs * steps_per_epoch
    else:
        warmup_steps = 2 * steps_per_epoch

    # Cosine schedule with warmup (BB pretrain.py 동일)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    try:
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

        # 데이터 크기 변경 시 LR 스케줄러만 새로 생성 (모델 가중치 유지)
        saved_n_windows = ckpt.get('n_windows', None)
        current_n_windows = len(train_dataset)
        if saved_n_windows and saved_n_windows != current_n_windows:
            print(f"  Data size changed: {saved_n_windows:,} → {current_n_windows:,} windows")
            print(f"  Resetting LR scheduler (keeping model weights + optimizer)")
            # scheduler은 위에서 새로 생성된 상태 유지 (load_state_dict 생략)
        else:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.5f}")

    # ---- Checkpoint & Logging ----
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.config).stem
    if args.filter:
        ckpt_name += f'_{args.filter}'
    if args.note:
        ckpt_name += f'_{args.note}'

    log_dir = PROJECT_DIR / 'runs' / ckpt_name
    writer = SummaryWriter(log_dir=str(log_dir))

    # ---- BB Fast Eval 초기화 ----
    bb_buildings = []
    bb_boxcox = None
    best_bb_cvrmse = float('inf')
    if args.bb_eval_interval > 0 and BB_DATA_DIR.exists():
        try:
            print("\nPre-loading BB test data (Commercial 955, inline eval)...")
            t_bb = time.time()
            bb_buildings = _load_bb_buildings(commercial_only=True, bdg2_only=False)
            bb_boxcox = _BBBoxCox()
            print(f"  Loaded {len(bb_buildings)} commercial buildings "
                  f"(lambda={bb_boxcox.scaler.lambdas_[0]:.4f}) in {time.time()-t_bb:.1f}s")
        except Exception as e:
            print(f"  [WARN] BB data load failed: {e} — BB eval disabled")
            bb_buildings = []

    # ---- Graceful Stop handler ----
    signal.signal(signal.SIGINT, _signal_handler)

    # ---- Training ----
    context_len = model_args['context_len']
    pred_len = model_args['pred_len']
    no_improve = 0
    stopped_early = False

    print(f"\n{'=' * 70}")
    print(f"Training: {ckpt_name}")
    if max_steps is not None:
        frac = max_steps / steps_per_epoch
        print(f"  Mode: SUB-EPOCH (max_steps={max_steps:,}, "
              f"= {frac:.3f} epochs, BB-equivalent)")
    else:
        print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Train: {len(train_dataset):,} windows, Val: {len(val_dataset):,} windows")
    print(f"  Steps/epoch: {steps_per_epoch:,}, Total: {total_steps:,}")
    if warmup_steps_cfg is not None:
        print(f"  Warmup: {warmup_steps:,} steps (fixed)")
    else:
        we = warmup_epochs if warmup_epochs is not None else 2
        print(f"  Warmup: {warmup_steps:,} steps ({we} epochs)")
    if grad_clip > 0:
        print(f"  Grad clip: {grad_clip}")
    else:
        print(f"  Grad clip: disabled (BB default)")
    if bb_buildings:
        print(f"  BB eval: every {args.bb_eval_interval} epochs "
              f"({len(bb_buildings)} commercial buildings)")
    print(f"{'=' * 70}")

    global_step = start_epoch * steps_per_epoch
    model.train()

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            for k, v in batch.items():
                batch[k] = v.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                if batch['load'].shape[1] != context_len + pred_len:
                    print(f"  [WARN] batch load shape={batch['load'].shape}, expected ({batch['load'].shape[0]}, {context_len+pred_len}, 1)", flush=True)
                    continue
                preds = model(batch)
                targets = batch['load'][:, context_len:]
                loss = model.loss(preds, targets)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only step scheduler if optimizer.step() was NOT skipped by GradScaler
            # (skip happens when inf/nan detected → scaler downsizes, prev_scale > current)
            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            if global_step % 1000 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                avg_so_far = epoch_loss / max(n_batches, 1)
                elapsed_so_far = time.time() - t0
                steps_done = global_step
                steps_left = (max_steps - global_step) if max_steps else (steps_per_epoch * epochs - global_step)
                eta_s = int(steps_left / max(global_step / max(elapsed_so_far, 1), 1e-6))
                eta_str = f"{eta_s//3600}h{(eta_s%3600)//60:02d}m" if eta_s >= 3600 else f"{eta_s//60}m{eta_s%60:02d}s"
                print(f"  step {steps_done:>6,}/{max_steps or steps_per_epoch:,} | loss={loss.item():.5f} | avg={avg_so_far:.5f} | lr={lr_now:.2e} | ETA {eta_str}", flush=True)

            # --max_steps: sub-epoch break
            if max_steps is not None and global_step >= max_steps:
                break

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # --max_steps 도달 시 강제 validation + BB eval 후 종료
        if max_steps is not None and global_step >= max_steps:
            frac_epoch = global_step / steps_per_epoch
            print(f"\n  [max_steps={max_steps:,}] reached at {frac_epoch:.2f} epochs "
                  f"(step {global_step:,})")
            print(f"  train_loss={avg_loss:.5f} | {elapsed:.0f}s")

            # Validation
            val_results = validate(model, val_loader, device, load_transform, context_len)
            val_loss = val_results['val_loss']
            nrmse = val_results['nrmse_all']
            writer.add_scalar('val/loss', val_loss, global_step)
            writer.add_scalar('val/nrmse_all', nrmse, global_step)
            print(f"  val_loss={val_loss:.5f} | NRMSE={nrmse:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                             CHECKPOINT_DIR / f'{ckpt_name}_best.pt',
                             n_windows=len(train_dataset))
            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                             CHECKPOINT_DIR / f'{ckpt_name}_last.pt',
                             n_windows=len(train_dataset))

            # BB eval (optional — wrapped to prevent crash from killing training)
            if bb_buildings:
                try:
                    t_bb = time.time()
                    bb_res = _bb_fast_eval(model, device, context_len, pred_len,
                                           bb_buildings, bb_boxcox, commercial_only=True)
                    if bb_res:
                        bb_cvrmse = bb_res['commercial_cvrmse']
                        bb_best_str = ""
                        if bb_cvrmse < best_bb_cvrmse:
                            best_bb_cvrmse = bb_cvrmse
                            bb_best_str = " *BB_BEST*"
                            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                             CHECKPOINT_DIR / f'{ckpt_name}_bb_best.pt',
                                             n_windows=len(train_dataset))
                        print(f"  BB Commercial CVRMSE: {bb_cvrmse * 100:.2f}% "
                              f"(gap vs SOTA: {bb_cvrmse * 100 - 13.31:+.2f}%p, "
                              f"{time.time()-t_bb:.0f}s){bb_best_str}")
                except Exception as e:
                    print(f"  [WARN] inline BB eval skipped: {e}")

            stopped_early = True
            break

        # ---- Validation ----
        val_results = None
        if (epoch + 1) % val_every == 0:
            val_results = validate(model, val_loader, device, load_transform, context_len)
            val_loss = val_results['val_loss']
            nrmse = val_results['nrmse_all']

            writer.add_scalar('val/loss', val_loss, global_step)
            writer.add_scalar('val/nrmse_all', nrmse, global_step)
            for btype in ['residential', 'commercial']:
                v = val_results.get(f'nrmse_{btype}', float('nan'))
                if not np.isnan(v):
                    writer.add_scalar(f'val/nrmse_{btype}', v, global_step)

            # Checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                no_improve = 0
                _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                 CHECKPOINT_DIR / f'{ckpt_name}_best.pt',
                                 n_windows=len(train_dataset))
            else:
                no_improve += 1

            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                             CHECKPOINT_DIR / f'{ckpt_name}_last.pt',
                             n_windows=len(train_dataset))

            nrmse_str = f"NRMSE={nrmse:.4f}" if not np.isnan(nrmse) else "NRMSE=N/A"
            best_str = " *BEST*" if is_best else ""
            print(f"  Epoch {epoch + 1:>3}/{epochs} | "
                  f"train_loss={avg_loss:.5f} | val_loss={val_loss:.5f} | "
                  f"{nrmse_str} | {elapsed:.0f}s{best_str}")

            # ---- BB Fast Eval (per N epochs) ----
            if (bb_buildings and args.bb_eval_interval > 0
                    and (epoch + 1) % args.bb_eval_interval == 0):
                try:
                    t_bb = time.time()
                    bb_res = _bb_fast_eval(model, device, context_len, pred_len,
                                           bb_buildings, bb_boxcox, commercial_only=True)
                    bb_elapsed = time.time() - t_bb
                    if bb_res:
                        bb_cvrmse = bb_res['commercial_cvrmse']
                        n_com = bb_res['n_commercial']
                        writer.add_scalar('bb/commercial_cvrmse', bb_cvrmse, global_step)
                        bb_best_str = ""
                        if bb_cvrmse < best_bb_cvrmse:
                            best_bb_cvrmse = bb_cvrmse
                            bb_best_str = " *BB_BEST*"
                            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                             CHECKPOINT_DIR / f'{ckpt_name}_bb_best.pt',
                                             n_windows=len(train_dataset))
                        print(f"  BB Commercial CVRMSE: {bb_cvrmse * 100:.2f}% "
                              f"(n={n_com}, best={best_bb_cvrmse * 100:.2f}%, "
                              f"{bb_elapsed:.0f}s){bb_best_str}")
                except Exception as e:
                    print(f"  [WARN] per-epoch BB eval skipped: {e}")

            # Early stopping
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch + 1} (patience={patience})")
                break
        else:
            # val_every > 1일 때도 체크포인트 저장 (stop 요청 시)
            if _check_stop():
                _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                 CHECKPOINT_DIR / f'{ckpt_name}_last.pt',
                                 n_windows=len(train_dataset))
                print(f"  Epoch {epoch + 1:>3}/{epochs} | "
                      f"train_loss={avg_loss:.5f} | {elapsed:.0f}s | STOPPED")
                stopped_early = True
                break
            print(f"  Epoch {epoch + 1:>3}/{epochs} | "
                  f"train_loss={avg_loss:.5f} | {elapsed:.0f}s")
            continue

        # val_every == 1 경로에서의 graceful stop 체크
        if _check_stop():
            print(f"  [STOP] 체크포인트 저장 완료 — 종료합니다.")
            stopped_early = True
            break

    writer.close()

    status = "Stopped (graceful)" if stopped_early else "Complete"
    print(f"\n{'=' * 70}")
    print(f"Training {status}")
    print(f"  Best val_loss: {best_val_loss:.5f}")
    if best_bb_cvrmse < float('inf'):
        print(f"  Best BB Commercial CVRMSE: {best_bb_cvrmse * 100:.2f}% "
              f"(gap vs SOTA: {best_bb_cvrmse * 100 - 13.31:+.2f}%p)")
    print(f"  Checkpoints: {CHECKPOINT_DIR / ckpt_name}_{{best,last}}.pt")
    if best_bb_cvrmse < float('inf'):
        print(f"  BB Best ckpt: {CHECKPOINT_DIR / ckpt_name}_bb_best.pt")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    if stopped_early:
        print(f"  Resume: python scripts/train.py --config {args.config} "
              f"--resume {CHECKPOINT_DIR / ckpt_name}_last.pt")
    print(f"{'=' * 70}")


def _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path,
                     n_windows: int = 0):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'n_windows': n_windows,  # 데이터 크기 변경 감지용
    }, path)


if __name__ == '__main__':
    main()
