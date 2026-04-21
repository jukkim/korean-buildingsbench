#!/bin/bash
# Download BuildingsBench evaluation data (CC-BY 4.0)
# Source: https://data.openei.org/submissions/5859
# Authors: Patrick Emami, Peter Graf (NREL)
#
# This script downloads only the evaluation datasets (<1GB).
# For the full Buildings-900K pretraining data (~110GB), visit the source URL.

set -e

DATA_DIR="${1:-external/BuildingsBench_data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading BuildingsBench evaluation data ==="
echo "License: CC-BY 4.0 (NREL)"
echo "Target: $DATA_DIR"
echo ""

# Evaluation datasets
BASE_URL="https://oedi-data-lake.s3.amazonaws.com/buildings-bench"

echo "[1/2] Downloading evaluation data..."
wget -q --show-progress -O "$DATA_DIR/BuildingsBench.tar.gz" \
  "${BASE_URL}/BuildingsBench.tar.gz" 2>&1 || \
  curl -L -o "$DATA_DIR/BuildingsBench.tar.gz" "${BASE_URL}/BuildingsBench.tar.gz"

echo "[2/2] Extracting..."
tar -xzf "$DATA_DIR/BuildingsBench.tar.gz" -C "$DATA_DIR" --strip-components=1
rm "$DATA_DIR/BuildingsBench.tar.gz"

echo ""
echo "=== Download complete ==="
echo "Set environment variable:"
echo "  export BUILDINGS_BENCH=$DATA_DIR"
echo ""
echo "Expected structure:"
echo "  $DATA_DIR/BDG-2/          (611 commercial buildings)"
echo "  $DATA_DIR/Electricity/    (344 commercial buildings)"
echo "  $DATA_DIR/metadata/       (transforms, oov.txt)"
echo "  $DATA_DIR/Sceaux/         (residential)"
echo "  $DATA_DIR/Borealis/       (residential)"
echo "  $DATA_DIR/IDEAL/          (residential)"
echo "  $DATA_DIR/LCL/            (residential)"
echo "  $DATA_DIR/SMART/          (residential)"
