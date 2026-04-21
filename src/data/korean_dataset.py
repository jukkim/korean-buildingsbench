"""Korean_BB Dataset — BB 호환 PyTorch 데이터셋

postprocess.py가 생성한 데이터를 로드:
  - data/korean_bb/individual/{building_id}.parquet  (timestamp index + power column)
  - data/korean_bb/metadata/catalog.csv              (building metadata)
  - data/korean_bb/metadata/transforms/boxcox.pkl    (BB Box-Cox pickle)
  - data/korean_bb/metadata/train_weekly.csv         (train index)
  - data/korean_bb/metadata/val_weekly.csv           (val index)

BB TorchBuildingDataset 호환 sample dict:
  {load, latitude, longitude, day_of_year, day_of_week, hour_of_day, building_type}

v3 추가: Data Augmentation (학습 시 적용)
  - Random window jitter: context window ±1~6h shift
  - Gaussian noise injection: Box-Cox 변환 후 N(0, 0.02) 추가
  - Random amplitude scaling: 건물별 U(0.85, 1.15) 곱

v3 메모리 최적화: Lazy loading + LRU cache
  - 초기화 시 catalog만 로드, Parquet 데이터는 첫 접근 시 로드
  - LRU cache (maxsize=5000): 50K buildings × 8760 × 4B → 175MB (vs 전체 1.75GB+)
"""
import pickle as pkl
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# LRU Cache (Parquet lazy loading용)
# ============================================================

class _LRUCache:
    """OrderedDict 기반 LRU 캐시 (maxsize 건물 데이터만 보유)"""

    def __init__(self, maxsize: int = 5000):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # 가장 오래된 항목 제거
        self._cache[key] = value

    def __len__(self):
        return len(self._cache)


# ============================================================
# BB 호환 상수
# ============================================================
CONTEXT_LEN = 168
PRED_LEN = 24
STRIDE = 24

# Augmentation 기본 설정
AUG_JITTER_MAX = 6       # ±6h window jitter
AUG_NOISE_STD = 0.02     # Box-Cox 공간에서 Gaussian noise σ
AUG_SCALE_RANGE = (0.85, 1.15)  # amplitude scaling 범위


# ============================================================
# Transforms (BB 방식 재현)
# ============================================================

class BoxCoxTransform:
    """BB 호환 Box-Cox transform — pickle 로드/적용"""

    def __init__(self):
        self.boxcox = None

    def load(self, transform_dir: Path) -> None:
        pkl_path = transform_dir / 'boxcox.pkl'
        with open(pkl_path, 'rb') as f:
            self.boxcox = pkl.load(f)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform: add 1e-6 offset then Box-Cox (BB 방식)"""
        shape = x.shape
        return self.boxcox.transform(
            1e-6 + x.flatten().reshape(-1, 1)
        ).reshape(shape).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        return self.boxcox.inverse_transform(
            x.flatten().reshape(-1, 1)
        ).reshape(shape).astype(np.float32)

    def undo_transform(self, x):
        """BB API 호환 (torch.Tensor or np.ndarray)"""
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            device = x.device
            x_np = x.cpu().numpy()
        else:
            x_np = x
        result = self.inverse_transform(x_np)
        if is_tensor:
            return torch.from_numpy(result).to(device)
        return result


class TimestampTransform:
    """BB 호환 timestamp feature 추출

    day_of_year, day_of_week, hour_of_day → [-1, +1] 정규화
    """

    def __init__(self, is_leap_year: bool = False):
        self.day_year_norm = 365 if is_leap_year else 364
        self.day_week_norm = 6
        self.hour_norm = 23

    def transform(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """(N,) timestamps → (N, 3) [day_of_year, day_of_week, hour_of_day]"""
        if isinstance(timestamps, pd.Series):
            timestamps = pd.to_datetime(timestamps)
        day_of_year = timestamps.dayofyear / self.day_year_norm
        day_of_week = timestamps.dayofweek / self.day_week_norm
        hour_of_day = timestamps.hour / self.hour_norm
        features = np.stack([day_of_year, day_of_week, hour_of_day], axis=1)
        return (features * 2 - 1).astype(np.float32)


class LatLonTransform:
    """한국 도시 기준 lat/lon 정규화"""

    def __init__(self, catalog: pd.DataFrame):
        self.lat_mean = catalog['latitude'].mean()
        self.lat_std = catalog['latitude'].std()
        self.lon_mean = catalog['longitude'].mean()
        self.lon_std = catalog['longitude'].std()
        # BB subset experiments use lat/lon=0 for all buildings → std=0 → NaN
        # Guard: if std is near zero (or NaN), force to 1.0 → transform returns 0
        if not np.isfinite(self.lat_std) or self.lat_std < 1e-8:
            self.lat_std = 1.0
        if not np.isfinite(self.lon_std) or self.lon_std < 1e-8:
            self.lon_std = 1.0

    def transform(self, lat: float, lon: float) -> np.ndarray:
        """(lat, lon) → normalized (2,)"""
        return np.array([
            (lat - self.lat_mean) / self.lat_std,
            (lon - self.lon_mean) / self.lon_std,
        ], dtype=np.float32)


# ============================================================
# KoreanBBDataset — 건물별 Parquet 기반 (평가용)
# ============================================================

class KoreanBBDataset(Dataset):
    """단일 건물의 시계열 → BB 호환 sliding window 샘플

    BB TorchBuildingDataset과 동일한 인터페이스.
    평가/전이학습에서 건물 단위로 사용.
    """

    def __init__(
        self,
        parquet_path: Path,
        building_latlon: np.ndarray,
        building_type_int: int,
        context_len: int = CONTEXT_LEN,
        pred_len: int = PRED_LEN,
        sliding_window: int = STRIDE,
        apply_scaler_transform: str = '',
        scaler_transform_path: Optional[Path] = None,
    ):
        self.context_len = context_len
        self.pred_len = pred_len
        self.sliding_window = sliding_window
        self.building_type_int = building_type_int
        self.latlon = building_latlon  # already normalized (2,)

        # Parquet 로드
        df = pd.read_parquet(parquet_path)
        self.power = df['power'].values.astype(np.float32)
        self.timestamps = df.index

        # Transforms
        self.time_transform = TimestampTransform()

        self.apply_scaler_transform = apply_scaler_transform
        if apply_scaler_transform == 'boxcox' and scaler_transform_path:
            self.load_transform = BoxCoxTransform()
            self.load_transform.load(scaler_transform_path)
        else:
            self.load_transform = None

    def __len__(self):
        return (len(self.power) - self.context_len - self.pred_len) // self.sliding_window

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        ptr = self.context_len + self.sliding_window * idx
        start = ptr - self.context_len
        end = ptr + self.pred_len

        load = self.power[start:end].copy()
        if self.load_transform:
            load = self.load_transform.transform(load)

        time_features = self.time_transform.transform(self.timestamps[start:end])
        seq_len = self.context_len + self.pred_len
        latlon = self.latlon.reshape(1, 2).repeat(seq_len, axis=0)
        btype = np.full((seq_len, 1), self.building_type_int, dtype=np.int32)

        return {
            'load': load[..., None],                    # (192, 1)
            'latitude': latlon[:, 0:1],                  # (192, 1)
            'longitude': latlon[:, 1:2],                 # (192, 1)
            'day_of_year': time_features[:, 0:1],        # (192, 1)
            'day_of_week': time_features[:, 1:2],        # (192, 1)
            'hour_of_day': time_features[:, 2:3],        # (192, 1)
            'building_type': btype,                      # (192, 1)
        }


# ============================================================
# KoreanBBPretrainingDataset — 인덱스 기반 (학습용)
# ============================================================

class KoreanBBPretrainingDataset(Dataset):
    """인덱스 파일 기반 대규모 학습 데이터셋

    로딩 모드:
    - preload=True  (기본): 초기화 시 전체 Parquet을 RAM에 로드 → 학습 중 디스크 I/O 없음
                            140K 건물 기준 ~4.9GB RAM. 학습 속도 최대화.
    - preload=False : Lazy loading + LRU cache (구버전, 소규모 실험용)
    """

    def __init__(
        self,
        data_dir: Path,
        index_file: str = 'train_weekly.csv',
        context_len: int = CONTEXT_LEN,
        pred_len: int = PRED_LEN,
        apply_scaler_transform: str = '',
        filter_str: str = '',
        augment: bool = False,
        lru_maxsize: int = 5000,
        preload: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.individual_dir = self.data_dir / 'individual'
        self.metadata_dir = self.data_dir / 'metadata'
        self.context_len = context_len
        self.pred_len = pred_len
        self.augment = augment
        self.preload = preload

        # 카탈로그 로드
        self.catalog = pd.read_csv(self.metadata_dir / 'catalog.csv')
        if filter_str:
            self.catalog = self.catalog[
                self.catalog['building_id'].str.contains(filter_str)
            ].reset_index(drop=True)

        # LatLon 정규화
        self.latlon_transform = LatLonTransform(self.catalog)
        self.time_transform = TimestampTransform()

        # Box-Cox
        self.apply_scaler_transform = apply_scaler_transform
        if apply_scaler_transform == 'boxcox':
            self.load_transform = BoxCoxTransform()
            self.load_transform.load(self.metadata_dir / 'transforms')
        else:
            self.load_transform = None

        self._latlon_cache: Dict[str, np.ndarray] = {}
        self._btype_cache: Dict[str, int] = {}
        # preload=True: 전체 RAM 로드 / preload=False: LRU lazy loading
        self._preloaded: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]] = {}
        self._parquet_paths: Dict[str, Path] = {}
        self._lru: _LRUCache = _LRUCache(maxsize=lru_maxsize)

        timestamps = pd.date_range('2023-01-01', periods=8760, freq='h')

        # 인덱스 로드 (preload 시 필요한 건물만 파악하기 위해 먼저 로드)
        idx_df = pd.read_csv(self.metadata_dir / index_file)
        if filter_str:
            idx_df = idx_df[
                idx_df['building_id'].str.contains(filter_str)
            ].reset_index(drop=True)
        needed_bids = set(idx_df['building_id'].unique()) if preload else None

        # catalog에서 building 메타데이터 로드
        catalog_map = {row['building_id']: row for _, row in self.catalog.iterrows()}

        available_bids = set()
        # preload 모드: index에 있는 건물만 순회. lazy 모드: catalog 전체 순회
        if preload:
            iter_bids = [b for b in needed_bids if b in catalog_map]
        else:
            iter_bids = list(catalog_map.keys())
        n = len(iter_bids)

        for i, bid in enumerate(iter_bids):
            row = catalog_map[bid]
            pq_path = self.individual_dir / f'{bid}.parquet'
            if not pq_path.exists():
                continue
            self._latlon_cache[bid] = self.latlon_transform.transform(
                row['latitude'], row['longitude']
            )
            self._btype_cache[bid] = int(row['building_type_int'])
            available_bids.add(bid)
            if preload:
                df = pd.read_parquet(pq_path)
                self._preloaded[bid] = (df['power'].values.astype(np.float32), timestamps)
                if (i + 1) % 10000 == 0:
                    print(f"    preloading {i+1}/{n}...")
            else:
                self._parquet_paths[bid] = pq_path

        # Parquet 파일이 있는 건물만
        idx_df = idx_df[idx_df['building_id'].isin(available_bids)]
        self.index = idx_df.reset_index(drop=True)

        aug_str = " +augmentation" if self.augment else ""
        mode_str = "preloaded" if preload else f"LRU cache={lru_maxsize}"
        n_train_bids = self.index['building_id'].nunique()
        print(f"  KoreanBBPretrainingDataset: {n_train_bids} buildings, "
              f"{len(self.index):,} windows ({index_file}){aug_str} [{mode_str}]")

    def _get_building_data(self, bid: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """preload 모드: dict에서 O(1) 조회 / lazy 모드: LRU cache"""
        if self.preload:
            return self._preloaded[bid]
        cached = self._lru.get(bid)
        if cached is not None:
            return cached
        df = pd.read_parquet(self._parquet_paths[bid])
        data = (df['power'].values.astype(np.float32), df.index)
        self._lru.put(bid, data)
        return data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        row = self.index.iloc[idx]
        bid = row['building_id']
        ptr = int(row['seq_ptr'])

        power, timestamps = self._get_building_data(bid)

        # ---- Augmentation: Random window jitter ----
        if self.augment:
            jitter = np.random.randint(-AUG_JITTER_MAX, AUG_JITTER_MAX + 1)
            ptr_aug = ptr + jitter
            # 경계 클리핑
            ptr_aug = max(self.context_len, min(ptr_aug, len(power) - self.pred_len))
        else:
            ptr_aug = ptr

        start = ptr_aug - self.context_len
        end = ptr_aug + self.pred_len

        load = power[start:end].copy()

        # ---- Augmentation: Random amplitude scaling (before transform) ----
        if self.augment:
            amp_scale = np.random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
            load = load * amp_scale

        if self.load_transform:
            load = self.load_transform.transform(load)

        # ---- Augmentation: Gaussian noise (after Box-Cox transform) ----
        if self.augment:
            noise = np.random.normal(0, AUG_NOISE_STD, size=load.shape).astype(np.float32)
            load = load + noise

        time_features = self.time_transform.transform(timestamps[start:end])
        seq_len = self.context_len + self.pred_len
        latlon = self._latlon_cache[bid].reshape(1, 2).repeat(seq_len, axis=0)
        btype = np.full((seq_len, 1), self._btype_cache[bid], dtype=np.int32)

        return {
            'load': load[..., None],
            'latitude': latlon[:, 0:1],
            'longitude': latlon[:, 1:2],
            'day_of_year': time_features[:, 0:1],
            'day_of_week': time_features[:, 1:2],
            'hour_of_day': time_features[:, 2:3],
            'building_type': btype,
        }

    def collate_fn(self):
        """BB 호환 collate function"""
        return _korean_bb_collate


def _korean_bb_collate(samples):
    """Module-level collate for Windows multiprocessing pickle compatibility."""
    return {
        k: torch.stack([torch.from_numpy(s[k]) for s in samples]).float()
        if k != 'building_type'
        else torch.stack([torch.from_numpy(s[k]) for s in samples]).long()
        for k in samples[0].keys()
    }
