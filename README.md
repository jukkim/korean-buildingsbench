# Korean BuildingsBench

**Seven Hundred Simulations Suffice: Operational Diversity for Data-Efficient Zero-Shot Building Load Forecasting**

Jeong-Uk Kim, Department of Electrical Engineering, Sangmyung University

## Key Results

700 parametric EnergyPlus simulations with 12D Latin Hypercube Sampling and RevIN reach benchmark-level zero-shot forecasting on 955 real commercial buildings — without any geographic information.

### Zero-Shot NRMSE (%) on 955 Commercial Buildings

| Model | Training Data | N Buildings | RevIN | NRMSE (%) |
|-------|--------------|:-----------:|:-----:|:---------:|
| BB-900K (baseline) | BB 900K | 900,000 | OFF | 13.27 |
| BB+RevIN | BB 900K | 900,000 | ON | 13.89 |
| **K-700 (ours)** | **Korean sim** | **700** | **ON** | **13.11 ± 0.17** (5-seed) / **12.93** (best) |
| K-700 no RevIN | Korean sim | 700 | OFF | 14.72 ± 0.28 (3-seed) |
| **US-700 (ours)** | **US TMY sim** | **700** | **ON** | **13.64 ± 0.65** (5-seed) |
| BB-700 aug | BB subset | 700 | ON | 14.26 |
| BB-700 | BB subset | 700 | ON | 15.28 |
| BB-700 OFF | BB subset | 700 | OFF | 16.44 |

## Paper & Reproducibility

- **Paper (Applied Energy, submitted)**: [`docs/paper_final.md`](docs/paper_final.md) · [`docs/paper_ae.docx`](docs/paper_ae.docx)
- **Graphical Abstract**: [`docs/graphical_abstract.png`](docs/graphical_abstract.png)
- **File map for full reproduction**: [`docs/PAPER_FILES_MANIFEST.md`](docs/PAPER_FILES_MANIFEST.md)
- **Results provenance (per-table checkpoint mapping)**: [`results/RESULTS_REGISTRY.md`](results/RESULTS_REGISTRY.md)

## Installation

```bash
pip install -r requirements.txt
pip install -e external/BuildingsBench/
```

## Download Evaluation Data

```bash
bash scripts/download_bb_data.sh
```

The BuildingsBench evaluation data (CC-BY 4.0, NREL) will be downloaded to `external/BuildingsBench_data/`.

## Data Availability

The Korean EnergyPlus simulation training data and model checkpoints (`.pt`) are not distributed in this repository. Both can be regenerated from the provided simulation pipeline:

```bash
# 1. Generate parametric IDFs (12D LHS, 14 archetypes × 50 schedules)
python scripts/generate_parametric_idfs.py --version v3 --archetype office \
  --n-schedules 50 --cities seoul,busan,daegu,gangneung,jeju --vintages all \
  --use-pool --output-dir simulations/idfs_v3

# 2. Run EnergyPlus simulations
python scripts/run_simulations.py --idf-dir simulations/idfs_v3 \
  --result-dir simulations/results_v3 --workers 8

# 3. Post-process to Parquet + fit Box-Cox + build index
python scripts/postprocess.py --version v3 --append --fit-only --index-only
```

Requires EnergyPlus ≥ 9.6 installed and available on PATH.

## Training

```bash
python scripts/train.py \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --train_index train_weekly_50.csv \
  --val_index val_weekly_50.csv \
  --max_steps 18000 --warmup_steps 500 \
  --augment --bb_eval_interval 0 --seed 42 \
  --note n50_s18000_revin_on_seed42
```

## Evaluation

```bash
python scripts/evaluate_bb.py \
  --checkpoint checkpoints/TransformerWithGaussian-M-v3-3k_ms_n50_s18000_revin_on_seed42_best.pt \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --commercial_only
```

## Citation

```bibtex
@article{kim2026sevenhundred,
  title={Seven Hundred Simulations Suffice: Operational Diversity for Data-Efficient Zero-Shot Building Load Forecasting},
  author={Kim, Jeong-Uk},
  journal={Applied Energy},
  year={2026},
  note={Submitted}
}
```

## Acknowledgment

This work was supported by the Korea Institute of Energy Technology Evaluation and Planning (KETEP) grant funded by the Ministry of Trade, Industry and Energy, Republic of Korea (Grant No. RS-00238487).

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

The BuildingsBench evaluation data is licensed under CC-BY 4.0 by NREL (Patrick Emami, Peter Graf).
