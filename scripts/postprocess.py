"""
시뮬레이션 결과 후처리 — BuildingsBench 호환 형식

EnergyPlus CSV → BB 호환 학습 데이터:
  1. 시간별 총 전력 추출 (J → kWh)
  2. 건물별 Parquet 저장 (timestamp index + 'power' column)
  3. 도시별 다건물 Parquet 저장 (pretraining용)
  4. Box-Cox 정규화 (BB 방식: 1e-6 offset, pickle 저장)
  5. 메타데이터 카탈로그 (building_id, lat, lon, building_type)
  6. Train/Val 인덱스 생성

BB TorchBuildingDataset 호환:
  - DataFrame: timestamp index + 'power' column (kWh)
  - building_latlon: [lat, lon]
  - building_type: RESIDENTIAL (0) or COMMERCIAL (1)

Usage:
    # 전체 후처리
    python scripts/postprocess.py

    # 특정 아키타입만
    python scripts/postprocess.py --filter office

    # Box-Cox만 재학습
    python scripts/postprocess.py --fit-only

    # 인덱스만 재생성
    python scripts/postprocess.py --index-only

    # 통계만 출력
    python scripts/postprocess.py --stats-only
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import pickle as pkl
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional

from sklearn.preprocessing import PowerTransformer
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 경로 설정
# ============================================================
PROJECT_DIR = Path(__file__).parent.parent
RESULT_DIR = PROJECT_DIR / 'simulations' / 'results'
IDF_DIR = PROJECT_DIR / 'simulations' / 'idfs'

# v3 기본 경로 (--result-dir / --idf-dir 로 오버라이드 가능)
RESULT_DIR_V3 = PROJECT_DIR / 'simulations' / 'results_v3'
IDF_DIR_V3 = PROJECT_DIR / 'simulations' / 'idfs_v3'

# 출력 디렉토리 (BB 스타일)
DATA_DIR = PROJECT_DIR / 'data' / 'korean_bb'
INDIVIDUAL_DIR = DATA_DIR / 'individual'      # 건물별 Parquet (evaluation용)
BY_CITY_DIR = DATA_DIR / 'by_city'            # 도시별 다건물 Parquet (pretraining용)
METADATA_DIR = DATA_DIR / 'metadata'
TRANSFORM_DIR = METADATA_DIR / 'transforms'

# BB 프로토콜 상수
CONTEXT_LEN = 168   # 1주일
PRED_LEN = 24       # 1일
STRIDE = 24         # 1일
TOTAL_HOURS = 8760  # 1년

# 학습/검증 분할 (BB 방식: 마지막 2주 = val)
TRAIN_END_HOUR = 8760 - 14 * 24  # 12/18 00:00 (마지막 336h = val)


# ============================================================
# 도시 좌표 (위도, 경도)
# ============================================================
CITY_LATLON = {
    'chuncheon': [37.8813, 127.7300],
    'wonju':     [37.3422, 127.9202],
    'seoul':     [37.5665, 126.9780],
    'incheon':   [37.4563, 126.7052],
    'daejeon':   [36.3504, 127.3845],
    'sejong':    [36.4800, 127.2600],
    'busan':     [35.1796, 129.0756],
    'daegu':     [35.8714, 128.6014],
    'gwangju':   [35.1595, 126.8526],
    'gangneung': [37.7519, 128.8761],
    'jeju':      [33.4996, 126.5312],
    'ulsan':     [35.5384, 129.3114],
}

# 아키타입 → 건물 유형 (BB: RESIDENTIAL=0, COMMERCIAL=1)
ARCHETYPE_BUILDING_TYPE = {
    'apartment_highrise': 'RESIDENTIAL',
    'apartment_midrise':  'RESIDENTIAL',
    'office':             'COMMERCIAL',
    'school':             'COMMERCIAL',
    'retail':             'COMMERCIAL',
    'hospital':           'COMMERCIAL',
    'hotel':              'COMMERCIAL',
    'small_office':       'COMMERCIAL',
    'large_office':       'COMMERCIAL',
    'warehouse':          'COMMERCIAL',
    'restaurant_full':    'COMMERCIAL',
    'restaurant_quick':   'COMMERCIAL',
    'strip_mall':         'COMMERCIAL',
    'university':         'COMMERCIAL',
}


# ============================================================
# 0. 8.simulation 표준 NPY 내보내기 경로
# ============================================================

# NPY 결과 루트: results/{sim_id}/
NPY_RESULTS_DIR = PROJECT_DIR / 'results'

# 열 → NPY 파일명 매핑 (EnergyPlus CSV 컬럼 키워드 → npy 이름)
METER_TO_NPY = {
    'Electricity:Facility':         'hourly_electricity',
    'Cooling:Electricity':          'hourly_cooling',
    'Heating:Electricity':          'hourly_heating',
    'Heating:NaturalGas':           'hourly_gas',
    'Fans:Electricity':             'hourly_fans',
    'InteriorEquipment:Electricity': 'hourly_equipment',
    'InteriorLights:Electricity':   'hourly_lights',
    'Pumps:Electricity':            'hourly_pumps',
    'WaterSystems:NaturalGas':      'hourly_water_gas',
}

VARIABLE_TO_NPY = {
    'Site Outdoor Air Drybulb Temperature': 'hourly_outdoor_temp',
    'Site Outdoor Air Relative Humidity':   'hourly_outdoor_humidity',
    'Site Direct Solar Radiation':          'hourly_solar',
    'Zone Mean Air Temperature':            'hourly_zone_temp',
    'Zone Air Relative Humidity':           'hourly_indoor_humidity',
    'Zone Thermostat Cooling Setpoint':     'hourly_setpoint_cool',
    'Zone Thermostat Heating Setpoint':     'hourly_setpoint_heat',
}


def extract_all_channels(result_dir: Path) -> dict:
    """EnergyPlus CSV → 전체 채널 추출 (Tier A용)

    Returns: dict {npy_name: np.ndarray (8760,)} — 없는 채널은 NaN 배열
    """
    csv_path = result_dir / 'eplusout.csv'
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception:
        return {}

    # sub-hourly → hourly 압축 함수
    def to_hourly(arr: np.ndarray) -> np.ndarray:
        if len(arr) > 8760:
            n = len(arr) // 8760
            arr = arr[:8760 * n].reshape(8760, n).sum(axis=1)
        arr = arr[:8760]
        if len(arr) < 8760:
            pad = np.zeros(8760)
            pad[:len(arr)] = arr
            arr = pad
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    channels = {}

    # Energy Meter 채널 (J → kWh)
    for keyword, npy_name in METER_TO_NPY.items():
        cols = [c for c in df.columns if keyword in c]
        if cols:
            vals = df[cols[0]].values.astype(np.float64)
            channels[npy_name] = to_hourly(vals) / 3_600_000.0
        else:
            channels[npy_name] = np.full(8760, np.nan, dtype=np.float32)

    # Output:Variable 채널 — zone 레벨은 평균 (면적가중 없음)
    for keyword, npy_name in VARIABLE_TO_NPY.items():
        cols = [c for c in df.columns if keyword in c]
        if cols:
            arr = df[cols].values.astype(np.float64)
            if arr.ndim == 2 and arr.shape[1] > 1:
                arr = arr.mean(axis=1)  # Zone 평균
            else:
                arr = arr.flatten()
            # 변수는 단위 변환 없음 (°C, %, W/m²)
            channels[npy_name] = to_hourly(arr)
        else:
            channels[npy_name] = np.full(8760, np.nan, dtype=np.float32)

    # peak_demand: Electricity:Facility [kWh/h] = [kW] (같은 배열 참조)
    if 'hourly_electricity' in channels:
        channels['hourly_peak_demand'] = channels['hourly_electricity'].copy()

    return channels


def save_npy_tier_a(sim_id: str, channels: dict, meta: dict,
                    npy_root: Optional[Path] = None) -> Optional[Path]:
    """Tier A 표준 NPY 저장

    results/{sim_id}/
        metadata.json
        hourly_*.npy

    partial_tier_a: True면 일부 채널 NaN (기존 시뮬 호환)
    """
    if not sim_id:
        return None

    npy_dir = (npy_root or NPY_RESULTS_DIR) / sim_id
    npy_dir.mkdir(parents=True, exist_ok=True)

    # metadata.json 저장
    # 기존 메타에서 sim_id 표준 필드만 추출
    std_meta = {
        'sim_id': sim_id,
        'source_project': meta.get('source_project', 'Korean_BB'),
        'building': meta.get('archetype', ''),
        'building_id': meta.get('building_id', ''),
        'hvac': meta.get('hvac', ''),
        'city': meta.get('city', ''),
        'vintage': meta.get('vintage', ''),
        'ems': 'M00',
        'output_tier': 'A',
        'partial_tier_a': any(np.all(np.isnan(v)) for v in channels.values()),
        'version': meta.get('version', 'v3'),
    }
    with open(npy_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(std_meta, f, indent=2, ensure_ascii=False)

    # NPY 저장
    for npy_name, arr in channels.items():
        np.save(npy_dir / f'{npy_name}.npy', arr)

    return npy_dir


# ============================================================
# 1. EnergyPlus CSV → hourly kWh 추출
# ============================================================

def extract_hourly_electricity(result_dir: Path) -> Optional[np.ndarray]:
    """EnergyPlus CSV → hourly total electricity (kWh)

    Returns:
        np.ndarray of shape (8760,) or None if failed
    """
    csv_path = result_dir / 'eplusout.csv'
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, index_col=0)

        # Electricity:Facility 컬럼 찾기
        elec_cols = [c for c in df.columns if 'Electricity:Facility' in c]
        if not elec_cols:
            return None

        values = df[elec_cols[0]].values.astype(np.float64)

        # sub-hourly → hourly 합산 (timestep=4 → 15분 간격)
        if len(values) > 8760:
            n_per_hour = len(values) // 8760
            if n_per_hour > 1:
                values = values[:8760 * n_per_hour].reshape(8760, n_per_hour).sum(axis=1)

        # J → kWh
        values_kwh = values / 3_600_000.0

        if len(values_kwh) < 8760:
            padded = np.zeros(8760)
            padded[:len(values_kwh)] = values_kwh
            values_kwh = padded
        elif len(values_kwh) > 8760:
            values_kwh = values_kwh[:8760]

        # NaN/Inf → 0
        values_kwh = np.nan_to_num(values_kwh, nan=0.0, posinf=0.0, neginf=0.0)

        return values_kwh.astype(np.float32)

    except Exception as e:
        print(f"  [ERROR] CSV parse: {result_dir.name}: {e}")
        return None


# ============================================================
# 2. 메타데이터 로드
# ============================================================

def load_building_metadata(building_id: str) -> Optional[dict]:
    """IDF 메타데이터 로드"""
    meta_path = IDF_DIR / building_id / 'metadata.json'
    if not meta_path.exists():
        return None
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 3. 건물별 Parquet 저장 (BB TorchBuildingDataset 호환)
# ============================================================

def save_individual_parquet(building_id: str, hourly_kwh: np.ndarray,
                            output_dir: Path) -> Path:
    """건물별 Parquet: timestamp index + 'power' column (kWh)

    BB TorchBuildingDataset이 기대하는 형식:
        df['power'].iloc[start:end].values
    """
    timestamps = pd.date_range('2023-01-01', periods=TOTAL_HOURS, freq='h')
    df = pd.DataFrame({
        'power': hourly_kwh,
    }, index=pd.Index(timestamps, name='timestamp'))

    out_path = output_dir / f'{building_id}.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine='pyarrow', compression='snappy')
    return out_path


# ============================================================
# 4. 도시별 다건물 Parquet (pretraining용 — BB Buildings900K 스타일)
# ============================================================

def save_city_parquets(catalog: pd.DataFrame, individual_dir: Path,
                       output_dir: Path) -> None:
    """도시별로 다건물 Parquet 생성

    BB Buildings900K 구조:
        timestamp | bldg_001 | bldg_002 | ...
    각 컬럼 = 해당 건물의 power (kWh)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range('2023-01-01', periods=TOTAL_HOURS, freq='h')

    for city in sorted(catalog['city'].unique()):
        city_dir = output_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)

        city_bldgs = catalog[catalog['city'] == city]

        # 아키타입별 Parquet
        for arch in sorted(city_bldgs['archetype'].unique()):
            arch_bldgs = city_bldgs[city_bldgs['archetype'] == arch]
            bldg_ids = arch_bldgs['building_id'].tolist()

            data = {'timestamp': timestamps}
            for bid in bldg_ids:
                pq_path = individual_dir / f'{bid}.parquet'
                if pq_path.exists():
                    bdf = pd.read_parquet(pq_path)
                    data[bid] = bdf['power'].values

            if len(data) <= 1:  # timestamp만
                continue

            df = pd.DataFrame(data)
            out_path = city_dir / f'{arch}.parquet'
            df.to_parquet(out_path, engine='pyarrow', compression='snappy')

    print(f"  도시별 Parquet → {output_dir}")


# ============================================================
# 5. Box-Cox (BB 방식 — pickle 저장)
# ============================================================

def fit_and_save_boxcox(all_values: np.ndarray, output_dir: Path) -> dict:
    """BB 방식 Box-Cox 학습 + pickle 저장

    BB transforms.py 원본:
        self.boxcox = PowerTransformer(method='box-cox', standardize=True)
        data = data.flatten().reshape(-1,1)
        if data.shape[0] > 1_000_000:
            data = data[np.random.choice(..., 1_000_000, replace=False)]
        self.boxcox.fit_transform(1e-6 + data)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 양수만 (Box-Cox 요구사항)
    positive = all_values[all_values > 0].flatten().reshape(-1, 1)
    if len(positive) < 100:
        print("  [WARN] 양수 데이터 부족 — Box-Cox 건너뜀")
        return {'method': 'none', 'reason': 'too_few_positive'}

    # 1M 서브샘플 (BB 방식)
    max_samples = 1_000_000
    if len(positive) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(positive), max_samples, replace=False)
        positive = positive[idx]

    # BB 원본과 동일: 1e-6 offset
    boxcox = PowerTransformer(method='box-cox', standardize=True)
    boxcox.fit_transform(1e-6 + positive)

    # Pickle 저장 (BB BoxCoxTransform.save 방식)
    pkl_path = output_dir / 'boxcox.pkl'
    with open(pkl_path, 'wb') as f:
        pkl.dump(boxcox, f)

    # JSON도 저장 (가독성)
    params = {
        'method': 'box-cox',
        'lambda': float(boxcox.lambdas_[0]),
        'mean': float(boxcox._scaler.mean_[0]),
        'std': float(boxcox._scaler.scale_[0]),
        'n_samples': len(positive),
        'offset': 1e-6,
        'pkl_path': str(pkl_path),
    }
    json_path = output_dir / 'boxcox_params.json'
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"  Box-Cox: lambda={params['lambda']:.4f}, "
          f"mean={params['mean']:.4f}, std={params['std']:.4f}")
    print(f"  → {pkl_path}")
    return params


# ============================================================
# 6. 카탈로그 CSV (건물 메타데이터)
# ============================================================

def build_catalog(building_records: list) -> pd.DataFrame:
    """건물 메타데이터 카탈로그 생성

    Columns:
        building_id, archetype, vintage, city, climate_zone,
        weather_year, schedule_class, variant_id,
        latitude, longitude, building_type,
        mean_kwh, peak_kwh, baseload_ratio
    """
    rows = []
    for rec in building_records:
        meta = rec['meta']
        hourly = rec['hourly_kwh']
        city = meta.get('city', 'seoul')
        latlon = CITY_LATLON.get(city, [37.5665, 126.9780])
        archetype = meta.get('archetype', 'unknown')
        btype = ARCHETYPE_BUILDING_TYPE.get(archetype, 'COMMERCIAL')

        mean_kwh = float(np.mean(hourly))
        peak_kwh = float(np.max(hourly))
        min_kwh = float(np.min(hourly[hourly > 0])) if np.any(hourly > 0) else 0.0
        baseload_ratio = min_kwh / mean_kwh if mean_kwh > 0 else 0.0

        rows.append({
            'building_id': meta['building_id'],
            'archetype': archetype,
            'vintage': meta.get('vintage', ''),
            'city': city,
            'climate_zone': meta.get('climate_zone', ''),
            'weather_year': meta.get('weather_year', 'tmy'),
            'schedule_class': meta.get('schedule_class', ''),
            'variant_id': meta.get('variant_id', 0),
            'latitude': latlon[0],
            'longitude': latlon[1],
            'building_type': btype,
            'building_type_int': 0 if btype == 'RESIDENTIAL' else 1,
            'mean_kwh': round(mean_kwh, 2),
            'peak_kwh': round(peak_kwh, 2),
            'baseload_ratio': round(baseload_ratio, 3),
        })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 7. Train/Val 인덱스 (BB 스타일)
# ============================================================

def generate_index_files(catalog: pd.DataFrame, output_dir: Path) -> dict:
    """BB 스타일 인덱스 파일 생성

    BB Buildings900K 인덱스 (tab-separated, fixed-width):
        building-type  region  puma-id  building-id  hour-pointer

    Korean_BB 인덱스 (tab-separated):
        building_type_int  city_idx  archetype_idx  building_id  hour_pointer

    또한 CSV 형태 인덱스도 생성 (사용 편의):
        building_id, start_hour, context_end, target_end, split
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    window = CONTEXT_LEN + PRED_LEN  # 192h
    building_ids = catalog['building_id'].tolist()

    train_rows = []
    val_rows = []

    for bid in building_ids:
        # Train: hour CONTEXT_LEN ~ TRAIN_END_HOUR (stride=24)
        for ptr in range(CONTEXT_LEN, TRAIN_END_HOUR, STRIDE):
            if ptr + PRED_LEN <= TRAIN_END_HOUR:
                train_rows.append({
                    'building_id': bid,
                    'seq_ptr': ptr,
                })

        # Val: hour TRAIN_END_HOUR ~ end (stride=24)
        for ptr in range(max(TRAIN_END_HOUR, CONTEXT_LEN),
                         TOTAL_HOURS - PRED_LEN + 1, STRIDE):
            val_rows.append({
                'building_id': bid,
                'seq_ptr': ptr,
            })

    # --- BB 스타일 .idx 파일 (tab-separated, fixed-width) ---
    # 도시/아키타입 인코딩
    cities = sorted(catalog['city'].unique())
    archetypes = sorted(catalog['archetype'].unique())
    city_map = {c: i for i, c in enumerate(cities)}
    arch_map = {a: i for i, a in enumerate(archetypes)}

    # 맵 저장 (디코딩용)
    with open(output_dir / 'encoding_maps.json', 'w', encoding='utf-8') as f:
        json.dump({
            'cities': cities,
            'archetypes': archetypes,
            'city_map': city_map,
            'arch_map': arch_map,
        }, f, indent=2, ensure_ascii=False)

    # O(n) lookup dict (O(n^2) 방지: 50K buildings × 17M windows)
    catalog_dict = catalog.set_index('building_id').to_dict('index')

    def _write_idx(rows, filename):
        """BB-style fixed-width tab-separated index — O(n) lookup"""
        lines = []
        for r in rows:
            bid = r['building_id']
            row_meta = catalog_dict[bid]  # O(1) lookup
            btype = int(row_meta['building_type_int'])
            city_idx = city_map[row_meta['city']]
            arch_idx = arch_map[row_meta['archetype']]
            ptr = r['seq_ptr']

            # 고정 너비: btype(1) \t city(2) \t arch(1) \t bid(80) \t ptr(4)
            line = f"{btype}\t{city_idx:02d}\t{arch_idx}\t{bid:80s}\t{ptr:04d}"
            lines.append(line)

        # 모든 줄 동일 길이로 패딩 (BB 방식)
        if lines:
            max_len = max(len(l) for l in lines)
            lines = [l.ljust(max_len) for l in lines]

        with open(output_dir / filename, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    _write_idx(train_rows, 'train_weekly.idx')
    _write_idx(val_rows, 'val_weekly.idx')

    # --- 편의용 CSV 인덱스도 저장 ---
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    train_df.to_csv(output_dir / 'train_weekly.csv', index=False)
    val_df.to_csv(output_dir / 'val_weekly.csv', index=False)

    stats = {
        'n_buildings': len(building_ids),
        'train_windows': len(train_rows),
        'val_windows': len(val_rows),
        'context_len': CONTEXT_LEN,
        'pred_len': PRED_LEN,
        'stride': STRIDE,
        'train_end_hour': TRAIN_END_HOUR,
    }

    print(f"  Train: {len(train_rows):,} windows")
    print(f"  Val:   {len(val_rows):,} windows")
    print(f"  → {output_dir}")

    return stats


# ============================================================
# 8. 통계 요약
# ============================================================

def compute_summary(catalog: pd.DataFrame) -> dict:
    """아키타입별, 도시별, 전체 통계"""
    summary = {
        'total_buildings': len(catalog),
        'by_archetype': {},
        'by_city': {},
        'by_building_type': {},
        'overall': {},
    }

    # 전체
    summary['overall'] = {
        'mean_kwh': round(float(catalog['mean_kwh'].mean()), 2),
        'peak_kwh_mean': round(float(catalog['peak_kwh'].mean()), 2),
        'baseload_ratio_mean': round(float(catalog['baseload_ratio'].mean()), 3),
        'baseload_ratio_median': round(float(catalog['baseload_ratio'].median()), 3),
    }

    # 아키타입별
    for arch, group in catalog.groupby('archetype'):
        summary['by_archetype'][arch] = {
            'count': len(group),
            'mean_kwh': round(float(group['mean_kwh'].mean()), 2),
            'peak_kwh_mean': round(float(group['peak_kwh'].mean()), 2),
            'baseload_ratio_mean': round(float(group['baseload_ratio'].mean()), 3),
        }

    # 도시별
    for city, group in catalog.groupby('city'):
        summary['by_city'][city] = {
            'count': len(group),
            'mean_kwh': round(float(group['mean_kwh'].mean()), 2),
        }

    # 건물 유형별
    for btype, group in catalog.groupby('building_type'):
        summary['by_building_type'][btype] = {
            'count': len(group),
            'mean_kwh': round(float(group['mean_kwh'].mean()), 2),
            'baseload_ratio_mean': round(float(group['baseload_ratio'].mean()), 3),
        }

    return summary


def print_summary(summary: dict, catalog: pd.DataFrame) -> None:
    """통계 출력"""
    print(f"\n{'=' * 80}")
    print(f"Korean_BB Postprocess Summary")
    print(f"{'=' * 80}")

    print(f"\nTotal buildings: {summary['total_buildings']:,}")
    print(f"Overall mean: {summary['overall']['mean_kwh']:.2f} kWh/h")
    print(f"Baseload ratio: {summary['overall']['baseload_ratio_mean']:.3f} "
          f"(median: {summary['overall']['baseload_ratio_median']:.3f})")

    # 아키타입별
    print(f"\n{'Archetype':<25} {'N':>6} {'Mean kWh/h':>12} {'Peak kWh':>12} {'Baseload':>10}")
    print("-" * 70)
    for arch in sorted(summary['by_archetype']):
        s = summary['by_archetype'][arch]
        print(f"  {arch:<23} {s['count']:>6,} {s['mean_kwh']:>12.2f} "
              f"{s['peak_kwh_mean']:>12.2f} {s['baseload_ratio_mean']:>10.3f}")

    # 건물 유형별
    print(f"\n{'Building Type':<15} {'N':>6} {'Mean kWh/h':>12} {'Baseload':>10}")
    print("-" * 50)
    for bt in sorted(summary['by_building_type']):
        s = summary['by_building_type'][bt]
        print(f"  {bt:<13} {s['count']:>6,} {s['mean_kwh']:>12.2f} "
              f"{s['baseload_ratio_mean']:>10.3f}")

    # 도시별
    print(f"\n{'City':<15} {'N':>6} {'Mean kWh/h':>12}")
    print("-" * 40)
    for city in sorted(summary['by_city']):
        s = summary['by_city'][city]
        print(f"  {city:<13} {s['count']:>6,} {s['mean_kwh']:>12.2f}")

    print(f"\n{'=' * 80}")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='EnergyPlus 결과 → BB 호환 학습 데이터')
    parser.add_argument('--filter', type=str, default='',
                        help='building_id 필터 (예: office)')
    parser.add_argument('--fit-only', action='store_true',
                        help='Box-Cox 파라미터 학습만')
    parser.add_argument('--index-only', action='store_true',
                        help='인덱스 파일 생성만')
    parser.add_argument('--stats-only', action='store_true',
                        help='통계만 출력 (기존 catalog 사용)')
    parser.add_argument('--skip-city-parquet', action='store_true',
                        help='도시별 다건물 Parquet 생략')
    parser.add_argument('--idf-dir', type=str, default='',
                        help='IDF 디렉토리 (기본: simulations/idfs)')
    parser.add_argument('--result-dir', type=str, default='',
                        help='결과 디렉토리 (기본: simulations/results)')
    parser.add_argument('--append', action='store_true',
                        help='기존 데이터에 추가 (새 건물만 처리)')
    parser.add_argument('--version', type=str, default='',
                        choices=['', 'v2', 'v3'],
                        help='v3 지정시 기본 경로를 results_v3/idfs_v3로 설정')
    parser.add_argument('--no-refit', action='store_true',
                        help='Box-Cox lambda 고정 (10K 이후 사용, lambda drift 방지)')
    parser.add_argument('--export-npy', action='store_true',
                        help='8.simulation 표준 NPY 내보내기 (Tier A, sim_id 기반). '
                             's6+부터 권장. 기존 건물은 partial_tier_a=true로 저장.')
    args = parser.parse_args()

    # --version shortcut: v3 경로 자동 적용 (--result-dir/--idf-dir 미지정시)
    if args.version == 'v3':
        if not args.result_dir:
            args.result_dir = str(RESULT_DIR_V3)
        if not args.idf_dir:
            args.idf_dir = str(IDF_DIR_V3)

    # --stats-only: 기존 카탈로그로 통계만
    if args.stats_only:
        catalog_path = METADATA_DIR / 'catalog.csv'
        if not catalog_path.exists():
            print("[ERROR] catalog.csv 없음 — 먼저 전체 후처리 실행")
            return
        catalog = pd.read_csv(catalog_path)
        summary = compute_summary(catalog)
        print_summary(summary, catalog)
        return

    # --fit-only: 기존 Parquet에서 Box-Cox만 재학습
    if args.fit_only:
        catalog_path = METADATA_DIR / 'catalog.csv'
        if not catalog_path.exists():
            print("[ERROR] catalog.csv 없음 — 먼저 전체 후처리 실행")
            return
        catalog = pd.read_csv(catalog_path)
        print(f"\n[FIT-ONLY] {len(catalog)} buildings에서 Box-Cox 재학습...")
        all_vals = []
        for bid in catalog['building_id']:
            pq_path = INDIVIDUAL_DIR / f'{bid}.parquet'
            if pq_path.exists():
                bdf = pd.read_parquet(pq_path)
                all_vals.append(bdf['power'].values)
        if all_vals:
            fit_and_save_boxcox(np.concatenate(all_vals), TRANSFORM_DIR)
        return

    # --index-only: 기존 카탈로그로 인덱스만
    if args.index_only:
        catalog_path = METADATA_DIR / 'catalog.csv'
        if not catalog_path.exists():
            print("[ERROR] catalog.csv 없음")
            return
        catalog = pd.read_csv(catalog_path)
        print(f"\n[INDEX] {len(catalog)} buildings")
        generate_index_files(catalog, METADATA_DIR)
        return

    # ============================================================
    # 전체 파이프라인
    # ============================================================

    # 사용자 지정 디렉토리
    global IDF_DIR, RESULT_DIR
    if args.idf_dir:
        IDF_DIR = Path(args.idf_dir)
    if args.result_dir:
        RESULT_DIR = Path(args.result_dir)

    for d in [DATA_DIR, INDIVIDUAL_DIR, METADATA_DIR, TRANSFORM_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- append 모드: 기존 카탈로그에서 처리 완료된 건물 목록 ----
    existing_ids = set()
    if args.append:
        catalog_path = METADATA_DIR / 'catalog.csv'
        if catalog_path.exists():
            existing_catalog = pd.read_csv(catalog_path)
            existing_ids = set(existing_catalog['building_id'].tolist())
            print(f"[APPEND] 기존 {len(existing_ids):,}개 건물 유지")

    # ---- 결과 디렉토리 스캔 ----
    result_dirs = sorted(d for d in RESULT_DIR.iterdir()
                         if d.is_dir() and not d.name.startswith('smoke'))

    if args.filter:
        result_dirs = [d for d in result_dirs if args.filter in d.name]

    # append 모드: 이미 처리된 건물 스킵
    if existing_ids:
        result_dirs = [d for d in result_dirs if d.name not in existing_ids]
        print(f"[APPEND] 새로 처리할 건물: {len(result_dirs):,}개")

    print(f"시뮬레이션 결과 디렉토리: {len(result_dirs):,}개")

    # ---- 1단계: CSV → hourly kWh + 건물별 Parquet ----
    print(f"\n[1/5] EnergyPlus CSV → hourly kWh 추출 + Parquet 저장...")
    t0 = time.time()

    records = []  # 성공한 건물 기록
    all_values_for_fit = []
    skipped = 0
    failed = 0

    for i, rdir in enumerate(result_dirs):
        building_id = rdir.name

        # status.json 확인
        status_file = rdir / 'status.json'
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            # 로컬: 'success', 클라우드(run_simulation.py): 'ok'
            if not (status.get('success', False) or status.get('ok', False)):
                skipped += 1
                continue
        elif not (rdir / 'eplusout.csv').exists():
            skipped += 1
            continue

        # 메타데이터 로드
        meta = load_building_metadata(building_id)
        if meta is None:
            # 메타데이터 없으면 building_id에서 파싱
            meta = {'building_id': building_id}
            parts = building_id.split('_')
            # archetype 추출 (첫 토큰들에서)
            for arch in ARCHETYPE_BUILDING_TYPE:
                if building_id.startswith(arch):
                    meta['archetype'] = arch
                    break
            # city 추출
            for city in CITY_LATLON:
                if f'_{city}_' in building_id:
                    meta['city'] = city
                    break

        # CSV → hourly kWh (Electricity:Facility)
        hourly = extract_hourly_electricity(rdir)
        if hourly is None:
            failed += 1
            continue

        # 품질 검증
        if np.sum(hourly) < 1.0:  # 거의 0인 데이터 제외
            failed += 1
            continue

        # 건물별 Parquet 저장 (BB 평가용)
        save_individual_parquet(building_id, hourly, INDIVIDUAL_DIR)

        # NPY 내보내기 (8.simulation 표준, --export-npy 시)
        if args.export_npy and meta.get('sim_id'):
            channels = extract_all_channels(rdir)
            if channels:
                save_npy_tier_a(meta['sim_id'], channels, meta)

        records.append({
            'building_id': building_id,
            'meta': meta,
            'hourly_kwh': hourly,
        })
        all_values_for_fit.append(hourly)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1:>6,}/{len(result_dirs):,}] "
                  f"OK:{len(records):,} skip:{skipped} fail:{failed} "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  완료: {len(records):,} buildings ({elapsed:.1f}s)")
    print(f"  (skipped: {skipped}, failed: {failed})")

    if not records:
        print("[ERROR] 처리할 데이터 없음")
        return

    # ---- 2단계: 카탈로그 생성 ----
    print(f"\n[2/5] 메타데이터 카탈로그 생성...")
    new_catalog = build_catalog(records)

    # append 모드: 기존 카탈로그와 병합
    if args.append and existing_ids:
        catalog = pd.concat([existing_catalog, new_catalog], ignore_index=True)
        catalog = catalog.drop_duplicates(subset='building_id', keep='last')
        print(f"  [APPEND] 기존 {len(existing_ids):,} + 신규 {len(new_catalog):,} = {len(catalog):,}")
    else:
        catalog = new_catalog

    catalog_path = METADATA_DIR / 'catalog.csv'
    catalog.to_csv(catalog_path, index=False)
    print(f"  {len(catalog):,} buildings → {catalog_path}")

    # ---- 3단계: Box-Cox ----
    print(f"\n[3/5] Box-Cox 학습 (BB 방식)...")
    if args.no_refit:
        # --no-refit: lambda 고정 (10K 이후 사용)
        pkl_path = TRANSFORM_DIR / 'boxcox.pkl'
        if pkl_path.exists():
            print(f"  [NO-REFIT] 기존 Box-Cox lambda 유지 ({pkl_path})")
            json_path = TRANSFORM_DIR / 'boxcox_params.json'
            boxcox_params = json.load(open(json_path)) if json_path.exists() else {'method': 'box-cox (frozen)'}
        else:
            print(f"  [WARN] --no-refit 지정했으나 기존 boxcox.pkl 없음 → 신규 학습")
            all_vals = np.concatenate(all_values_for_fit)
            boxcox_params = fit_and_save_boxcox(all_vals, TRANSFORM_DIR)
    else:
        # append 모드: 기존 건물 데이터도 포함하여 Box-Cox 재학습
        if args.append and existing_ids:
            print(f"  [APPEND] 기존 건물 데이터 로드 중...")
            for bid in existing_ids:
                pq_path = INDIVIDUAL_DIR / f'{bid}.parquet'
                if pq_path.exists():
                    bdf = pd.read_parquet(pq_path)
                    all_values_for_fit.append(bdf['power'].values)
        all_vals = np.concatenate(all_values_for_fit)
        boxcox_params = fit_and_save_boxcox(all_vals, TRANSFORM_DIR)

    # ---- 4단계: 도시별 다건물 Parquet ----
    if not args.skip_city_parquet:
        print(f"\n[4/5] 도시별 다건물 Parquet 생성...")
        save_city_parquets(catalog, INDIVIDUAL_DIR, BY_CITY_DIR)
    else:
        print(f"\n[4/5] 도시별 Parquet 건너뜀 (--skip-city-parquet)")

    # ---- 5단계: 인덱스 + 통계 ----
    print(f"\n[5/5] Train/Val 인덱스 생성...")
    idx_stats = generate_index_files(catalog, METADATA_DIR)

    # 통계
    summary = compute_summary(catalog)
    summary['index'] = idx_stats
    summary['boxcox'] = boxcox_params

    summary_path = METADATA_DIR / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary, catalog)

    # BB 호환 빈 oov.txt (Korean_BB는 OOV 없음)
    oov_path = METADATA_DIR / 'oov.txt'
    if not oov_path.exists():
        oov_path.write_text('')

    print(f"\n출력 디렉토리: {DATA_DIR}")
    print(f"  individual/  — 건물별 Parquet ({len(records):,}개)")
    print(f"  by_city/     — 도시별 다건물 Parquet")
    print(f"  metadata/    — 카탈로그, Box-Cox, 인덱스")


if __name__ == '__main__':
    main()
