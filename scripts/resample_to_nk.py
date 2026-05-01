"""아키타입별 N건물 균등 샘플링 → 학습 인덱스 재생성

방안 E: 아키타입별 max N건물로 캡핑하여 학습 분포 균형화.
  - 초과 아키타입(restaurant 13K 등)은 N개로 다운샘플
  - 부족 아키타입(university 2K 등)은 그대로 유지

--restaurant-cap: restaurant_full/quick만 별도 캡 적용 (다른 아키타입은 --n 적용)

Usage:
    python scripts/resample_to_nk.py --n 3000
    python scripts/resample_to_nk.py --n 4000
    python scripts/resample_to_nk.py --n 4000 --restaurant-cap 2000
    python scripts/resample_to_nk.py --n 3000 --seed 42
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / 'data' / 'korean_bb'
META_DIR = DATA_DIR / 'metadata'

RESTAURANT_ARCHS = {'restaurant_full', 'restaurant_quick'}


def resample(n: int, seed: int = 42, restaurant_cap: int = None, tag_override: str = None):
    rng = np.random.default_rng(seed)

    catalog = pd.read_csv(META_DIR / 'catalog.csv')
    train_idx = pd.read_csv(META_DIR / 'train_weekly.csv')
    val_idx   = pd.read_csv(META_DIR / 'val_weekly.csv')

    print(f"원본 catalog: {len(catalog):,} buildings")
    print(f"원본 train: {len(train_idx):,} windows")
    print(f"원본 val:   {len(val_idx):,} windows")
    if restaurant_cap is not None:
        print(f"restaurant cap: {restaurant_cap:,} / 기타 cap: {n:,}")
    print()

    # 아키타입별 캡 적용 샘플링
    sampled_bids = []
    summary = []

    for arch, group in catalog.groupby('archetype'):
        bids = group['building_id'].values
        cap = (restaurant_cap if (restaurant_cap is not None and arch in RESTAURANT_ARCHS) else n)
        k = min(len(bids), cap)
        if len(bids) > cap:
            chosen = rng.choice(bids, size=cap, replace=False)
            action = f"downsampled {len(bids)}→{cap}"
        else:
            chosen = bids
            action = f"kept all {len(bids)}"
        sampled_bids.extend(chosen.tolist())
        summary.append((arch, len(bids), k, action))

    print(f"{'Archetype':<25} {'원본':>6} {'샘플':>6}  {'액션'}")
    print('-' * 65)
    total_orig = 0
    total_new = 0
    for arch, orig, new, action in sorted(summary):
        print(f"  {arch:<23} {orig:>6,} {new:>6,}  {action}")
        total_orig += orig
        total_new += new
    print('-' * 65)
    print(f"  {'합계':<23} {total_orig:>6,} {total_new:>6,}")
    print()

    # 인덱스 필터링
    sampled_set = set(sampled_bids)
    new_train = train_idx[train_idx['building_id'].isin(sampled_set)].reset_index(drop=True)
    new_val   = val_idx[val_idx['building_id'].isin(sampled_set)].reset_index(drop=True)

    # 저장 파일명 태그
    if tag_override is not None:
        tag = tag_override
    elif restaurant_cap is not None:
        tag = f"r{restaurant_cap // 1000}k_{n // 1000}k"
    else:
        tag = f"{n // 1000}k"
    train_out = META_DIR / f'train_weekly_{tag}.csv'
    val_out   = META_DIR / f'val_weekly_{tag}.csv'
    new_train.to_csv(train_out, index=False)
    new_val.to_csv(val_out, index=False)

    print(f"저장 완료:")
    print(f"  {train_out.name}: {len(new_train):,} windows ({len(new_train)/len(train_idx)*100:.1f}% of original)")
    print(f"  {val_out.name}:   {len(new_val):,} windows")
    print()
    print(f"학습 명령 예시:")
    print(f"  python scripts/train.py \\")
    print(f"    --config configs/model/TransformerWithGaussian-M-v3-3k.toml \\")
    print(f"    --train_index train_weekly_{tag}.csv \\")
    print(f"    --val_index val_weekly_{tag}.csv \\")
    print(f"    --max_steps 40000 --note v3_{tag}_revin")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3000,
                        help='아키타입당 최대 건물 수 (default: 3000)')
    parser.add_argument('--restaurant-cap', type=int, default=None,
                        help='restaurant_full/quick 전용 캡. 미설정시 --n 사용.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, default=None,
                        help='파일명 태그 오버라이드 (예: 1k5). 미설정시 n//1000k 자동 생성.')
    args = parser.parse_args()
    resample(args.n, args.seed, args.restaurant_cap, args.tag)
