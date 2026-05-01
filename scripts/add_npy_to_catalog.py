"""
npy_tier_a에 있지만 catalog에 없는 건물을 parquet + catalog + train/val 인덱스에 추가.

Usage:
    python scripts/add_npy_to_catalog.py [--dry-run]
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 상수 ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
NPY_DIR     = ROOT / 'simulations' / 'npy_tier_a'
DATA_DIR    = ROOT / 'data' / 'korean_bb'
META_DIR    = DATA_DIR / 'metadata'
INDIV_DIR   = DATA_DIR / 'individual'

CATALOG_CSV     = META_DIR / 'catalog.csv'
TRAIN_CSV       = META_DIR / 'train_weekly.csv'
VAL_CSV         = META_DIR / 'val_weekly.csv'

# ── 시계열 상수 ──────────────────────────────────────────────────────────────
CONTEXT_LEN     = 168
PRED_LEN        = 24
STRIDE          = 24
TOTAL_HOURS     = 8760
TRAIN_END_HOUR  = 8760 - 14 * 24   # = 8424

# ── 룩업 테이블 (catalog에서 확인된 값) ────────────────────────────────────
ARCH_TO_TYPE = {
    'apartment_highrise': ('RESIDENTIAL', 0),
    'apartment_midrise':  ('RESIDENTIAL', 0),
    'hospital':           ('COMMERCIAL',  1),
    'hotel':              ('COMMERCIAL',  1),
    'large_office':       ('COMMERCIAL',  1),
    'office':             ('COMMERCIAL',  1),
    'restaurant_full':    ('COMMERCIAL',  1),
    'restaurant_quick':   ('COMMERCIAL',  1),
    'retail':             ('COMMERCIAL',  1),
    'school':             ('COMMERCIAL',  1),
    'small_office':       ('COMMERCIAL',  1),
    'strip_mall':         ('COMMERCIAL',  1),
    'university':         ('COMMERCIAL',  1),
    'warehouse':          ('COMMERCIAL',  1),
}

CITY_META = {
    'busan':     {'latitude': 35.1796, 'longitude': 129.0756, 'climate_zone': 'southern'},
    'daegu':     {'latitude': 35.8714, 'longitude': 128.6014, 'climate_zone': 'southern'},
    'gangneung': {'latitude': 37.7519, 'longitude': 128.8761, 'climate_zone': 'central_1'},
    'jeju':      {'latitude': 33.4996, 'longitude': 126.5312, 'climate_zone': 'jeju'},
    'seoul':     {'latitude': 37.5665, 'longitude': 126.9780, 'climate_zone': 'central_2'},
}

KNOWN_ARCHETYPES = set(ARCH_TO_TYPE.keys())


def parse_building_id(building_id: str) -> dict | None:
    """building_id → {archetype, vintage, city, variant_id} 파싱"""
    # metadata.json 우선 사용
    meta_path = NPY_DIR / building_id / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        # p-index 파싱
        import re
        p = re.search(r'_p(\d+)$', building_id)
        return {
            'archetype':  m.get('archetype', ''),
            'vintage':    m.get('vintage', ''),
            'city':       m.get('city', ''),
            'variant_id': int(p.group(1)) if p else 0,
        }
    # fallback: 이름 파싱
    import re
    vintages = ['v5_2018_plus', 'v4_2011_2017', 'v3_2001_2010', 'v2_1991_2000', 'v1_pre1990']
    cities   = list(CITY_META.keys())
    for arch in sorted(KNOWN_ARCHETYPES, key=len, reverse=True):
        if building_id.startswith(arch + '_'):
            rest = building_id[len(arch)+1:]
            for vint in vintages:
                if rest.startswith(vint + '_'):
                    rest2 = rest[len(vint)+1:]
                    for city in cities:
                        if rest2.startswith(city + '_tmy_p'):
                            p_str = rest2[len(city)+6:]
                            return {
                                'archetype':  arch,
                                'vintage':    vint,
                                'city':       city,
                                'variant_id': int(p_str),
                            }
    return None


def compute_stats(elec: np.ndarray) -> dict:
    mean_kwh      = float(np.mean(elec))
    peak_kwh      = float(np.max(elec))
    p5            = float(np.percentile(elec, 5))
    baseload_ratio = p5 / mean_kwh if mean_kwh > 0 else 0.0
    return {'mean_kwh': round(mean_kwh, 4),
            'peak_kwh': round(peak_kwh, 4),
            'baseload_ratio': round(baseload_ratio, 6)}


def make_windows(building_id: str) -> tuple[list, list]:
    train_rows, val_rows = [], []
    for ptr in range(CONTEXT_LEN, TRAIN_END_HOUR, STRIDE):
        if ptr + PRED_LEN <= TRAIN_END_HOUR:
            train_rows.append({'building_id': building_id, 'seq_ptr': ptr})
    for ptr in range(TRAIN_END_HOUR, TOTAL_HOURS - PRED_LEN + 1, STRIDE):
        val_rows.append({'building_id': building_id, 'seq_ptr': ptr})
    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='변경 없이 통계만 출력')
    args = parser.parse_args()

    # 현재 catalog 로드
    cat = pd.read_csv(CATALOG_CSV)
    cat_ids = set(cat['building_id'])
    npy_bldgs = sorted(NPY_DIR.iterdir())

    missing = [d.name for d in npy_bldgs if d.is_dir() and d.name not in cat_ids]
    print(f"총 missing: {len(missing):,} 건물")

    # 아키타입별 분류
    from collections import Counter
    import re
    arch_counts = Counter()
    for b in missing:
        m = re.match(r'^([a-z_]+)_v', b)
        if m:
            arch_counts[m.group(1)] += 1
    print("아키타입별:")
    for a, c in sorted(arch_counts.items(), key=lambda x: -x[1]):
        print(f"  {a:>25}: {c:>5,}")

    if args.dry_run:
        print("\n[dry-run] 종료")
        return

    # 처리
    new_catalog_rows = []
    new_train_rows   = []
    new_val_rows     = []
    skip = 0
    timestamps = pd.date_range('2023-01-01', periods=TOTAL_HOURS, freq='h')

    for i, bldg in enumerate(missing):
        npy_path = NPY_DIR / bldg / 'hourly_electricity.npy'
        if not npy_path.exists():
            skip += 1
            continue

        info = parse_building_id(bldg)
        if info is None:
            print(f"  [WARN] 파싱 실패: {bldg}")
            skip += 1
            continue

        arch = info['archetype']
        city = info['city']
        if arch not in ARCH_TO_TYPE or city not in CITY_META:
            skip += 1
            continue

        elec = np.load(npy_path).astype(np.float32)
        if len(elec) != TOTAL_HOURS:
            skip += 1
            continue

        # 1. 개별 parquet 저장
        parq_path = INDIV_DIR / f'{bldg}.parquet'
        if not parq_path.exists():
            df_parq = pd.DataFrame({'power': elec}, index=timestamps)
            df_parq.to_parquet(parq_path, engine='pyarrow', compression='snappy')

        # 2. catalog 행 생성
        btype, btype_int = ARCH_TO_TYPE[arch]
        cm = CITY_META[city]
        stats = compute_stats(elec)
        new_catalog_rows.append({
            'building_id':      bldg,
            'archetype':        arch,
            'vintage':          info['vintage'],
            'city':             city,
            'climate_zone':     cm['climate_zone'],
            'weather_year':     'tmy',
            'schedule_class':   None,
            'variant_id':       info['variant_id'],
            'latitude':         cm['latitude'],
            'longitude':        cm['longitude'],
            'building_type':    btype,
            'building_type_int': btype_int,
            **stats,
        })

        # 3. 인덱스 윈도우 생성
        tr, va = make_windows(bldg)
        new_train_rows.extend(tr)
        new_val_rows.extend(va)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(missing)} 처리중...")

    print(f"\n처리 완료: {len(new_catalog_rows):,} 건물 추가 / {skip} 스킵")
    print(f"  신규 train 윈도우: {len(new_train_rows):,}")
    print(f"  신규 val 윈도우:   {len(new_val_rows):,}")

    # catalog 저장
    new_cat = pd.concat([cat, pd.DataFrame(new_catalog_rows)], ignore_index=True)
    new_cat.to_csv(CATALOG_CSV, index=False)
    print(f"catalog 저장: {len(new_cat):,} 건물")

    # train_weekly.csv 업데이트
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = pd.concat([train_df, pd.DataFrame(new_train_rows)], ignore_index=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    print(f"train_weekly.csv: {len(train_df):,} 윈도우")

    # val_weekly.csv 업데이트
    val_df = pd.read_csv(VAL_CSV)
    val_df = pd.concat([val_df, pd.DataFrame(new_val_rows)], ignore_index=True)
    val_df.to_csv(VAL_CSV, index=False)
    print(f"val_weekly.csv:   {len(val_df):,} 윈도우")

    print("\n완료!")


if __name__ == '__main__':
    main()
