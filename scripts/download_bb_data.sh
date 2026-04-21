#!/bin/bash
# Download BuildingsBench evaluation data (CC-BY 4.0)
# Source: https://data.openei.org/submissions/5859
# Authors: Patrick Emami, Peter Graf (NREL)
#
# Uses AWS S3 public bucket (no credentials needed).
# Requires: aws cli (pip install awscli)
# Total download: ~17 GB (evaluation datasets + metadata + checkpoints)

set -e

DATA_DIR="${1:-external/BuildingsBench_data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading BuildingsBench evaluation data ==="
echo "License: CC-BY 4.0 (NREL)"
echo "Target: $DATA_DIR"
echo "Note: ~17 GB download. Requires AWS CLI (pip install awscli)."
echo ""

S3_BASE="s3://oedi-data-lake/buildings-bench/v1.0.0/BuildingsBench"

# Evaluation datasets + metadata
for dataset in BDG-2 Electricity Sceaux Borealis IDEAL LCL SMART metadata; do
  echo "Syncing $dataset ..."
  aws s3 sync --no-sign-request \
    "${S3_BASE}/${dataset}/" \
    "${DATA_DIR}/${dataset}/"
done

echo ""
echo "=== Download complete ==="
echo "Set environment variable:"
echo "  export BUILDINGS_BENCH=$DATA_DIR"
echo ""
echo "To also download BB official checkpoints:"
echo "  aws s3 sync --no-sign-request ${S3_BASE}/checkpoints/ ${DATA_DIR}/checkpoints/"
echo ""
echo "Contents:"
du -sh "$DATA_DIR"/*/ 2>/dev/null | sort -rh || ls "$DATA_DIR"/
