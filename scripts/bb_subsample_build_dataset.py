# -*- coding: utf-8 -*-
"""
BB Buildings-900K ComStock subset → Korean_BB pretraining format conversion.

Run on 5090 where E:\BuildingsBench\Buildings-900K is located.

Outputs:
  - data/bb_subset/buildings.npz (arrays of building time-series)
  - data/bb_subset/catalog.csv (building_id, puma, mean_load, ...)
  - data/bb_subset/train_weekly_Nxxx.csv / val_weekly_Nxxx.csv (index files)

Usage:
  python scripts/bb_subsample_build_dataset.py --n_buildings 700 --seed 42

The dataset can then be used with the existing train.py.
"""
import argparse
import os
import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- Paths ---
BB_ROOT = Path(r"E:\BuildingsBench\Buildings-900K\end-use-load-profiles-for-us-building-stock\2021")
COMSTOCK_DIR = BB_ROOT / "comstock_tmy3_release_1" / "timeseries_individual_buildings"
OUT_BASE = Path(r"C:\Korean_BB\data\bb_subset")


def gather_puma_dirs():
    """All puma directories under comstock_tmy3."""
    regions = [p for p in COMSTOCK_DIR.iterdir() if p.is_dir() and p.name.startswith("by_puma")]
    puma_dirs = []
    for region in regions:
        upgrade = region / "upgrade=0"
        if not upgrade.exists():
            continue
        for p in upgrade.iterdir():
            if p.is_dir() and p.name.startswith("puma="):
                puma_dirs.append(p)
    return puma_dirs


def read_puma_parquet(puma_dir):
    """Read first parquet in puma directory → dict {building_id: series}."""
    parquets = list(puma_dir.glob("*.snappy.parquet"))
    if not parquets:
        return {}
    df = pd.read_parquet(parquets[0])
    if "timestamp" not in df.columns:
        return {}
    df = df.sort_values("timestamp").reset_index(drop=True)
    series_dict = {}
    for col in df.columns:
        if col == "timestamp":
            continue
        try:
            int(col)
        except ValueError:
            continue
        series = df[col].values.astype(np.float32)
        # Pad to 8760 if 8759
        if len(series) == 8759:
            series = np.append(series, series[-1])
        elif len(series) != 8760:
            continue
        series_dict[col] = series
    return series_dict, df["timestamp"].values


def sample_buildings(n_buildings, seed):
    """Sample n_buildings randomly across all PUMAs."""
    puma_dirs = gather_puma_dirs()
    random.seed(seed)
    random.shuffle(puma_dirs)

    collected = {}  # building_id → (series, puma_id, timestamps)
    for puma in puma_dirs:
        if len(collected) >= n_buildings:
            break
        try:
            series_dict, timestamps = read_puma_parquet(puma)
        except Exception as e:
            print(f"[skip] {puma.name}: {e}")
            continue
        puma_id = puma.name.replace("puma=", "")
        # Sample a few buildings from each PUMA (not all to get more diversity)
        bids = list(series_dict.keys())
        random.shuffle(bids)
        # Take up to 10 per PUMA → more diversity
        for bid in bids[:10]:
            if len(collected) >= n_buildings:
                break
            unique_id = f"{puma_id}_{bid}"
            collected[unique_id] = (series_dict[bid], puma_id, timestamps)

    return collected


def build_index(catalog_df, n_train_ratio=0.98, seed=42):
    """Generate train/val weekly window index files compatible with our loader.

    Weekly = 168h context window, stride=168. Each row: (building_id, start_hour).
    """
    random.seed(seed)
    rows_train = []
    rows_val = []
    for _, row in catalog_df.iterrows():
        bid = row["building_id"]
        n_hours = 8760
        starts = list(range(0, n_hours - 192 + 1, 168))  # 168+24 = 192
        random.shuffle(starts)
        n_train = int(len(starts) * n_train_ratio)
        for s in starts[:n_train]:
            rows_train.append({"building_id": bid, "start_hour": s})
        for s in starts[n_train:]:
            rows_val.append({"building_id": bid, "start_hour": s})
    return pd.DataFrame(rows_train), pd.DataFrame(rows_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_buildings", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None, help="output tag (default: N)")
    args = ap.parse_args()

    tag = args.tag or str(args.n_buildings)
    out_dir = OUT_BASE / f"seed{args.seed}_n{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.n_buildings} buildings from ComStock TMY3 (seed={args.seed})...")
    collected = sample_buildings(args.n_buildings, args.seed)
    print(f"Collected: {len(collected)} buildings")

    # Save arrays
    bids = sorted(collected.keys())
    series_arr = np.stack([collected[b][0] for b in bids])  # (N, 8760)
    mean_loads = series_arr.mean(axis=1)
    np.savez_compressed(out_dir / "buildings.npz", series=series_arr, building_ids=np.array(bids))
    print(f"Saved: {out_dir / 'buildings.npz'} (shape={series_arr.shape})")

    # Catalog
    catalog = pd.DataFrame({
        "building_id": bids,
        "puma": [collected[b][1] for b in bids],
        "archetype": ["comstock"] * len(bids),
        "mean_load": mean_loads,
    })
    catalog.to_csv(out_dir / "catalog.csv", index=False)
    print(f"Saved: {out_dir / 'catalog.csv'}")

    # Index
    train_df, val_df = build_index(catalog, seed=args.seed)
    train_df.to_csv(out_dir / f"train_weekly_bb{tag}.csv", index=False)
    val_df.to_csv(out_dir / f"val_weekly_bb{tag}.csv", index=False)
    print(f"Saved train: {len(train_df)} windows, val: {len(val_df)} windows")
    print(f"All done. Dataset at {out_dir}")


if __name__ == "__main__":
    main()
