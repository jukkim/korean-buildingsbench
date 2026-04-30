# Korean BuildingsBench

**Seven Hundred Simulations Suffice: Matching a 900,000-Building Foundation Model through Operational Diversity in Zero-Shot Load Forecasting**

Jeong-Uk Kim, Department of Electrical Engineering, Sangmyung University

## Key Results

700 parametric EnergyPlus simulations with RevIN match and in the best-seed result exceed the BuildingsBench SOTA (900K buildings) on zero-shot commercial load forecasting — without any geographic information.

### Zero-Shot NRMSE (%) on 955 Commercial Buildings

| Model | Training Data | N Buildings | RevIN | NRMSE (%) |
|-------|--------------|:-----------:|:-----:|:---------:|
| BB SOTA-M (reproduced) | BB 900K | 900,000 | OFF | 13.27 |
| BB 900K + RevIN | BB 900K | 900,000 | ON | 13.89 |
| **Korean-700 (ours)** | **Korean sim** | **700** | **ON** | **13.11 ± 0.17** (5-seed) / **12.93** (best) |
| Korean-700 | Korean sim | 700 | OFF | 14.72 ± 0.28 (3-seed) |
| BB-700 | BB subset | 700 | ON | 15.28 |
| BB-700 | BB subset | 700 | OFF | 16.44 |

### Zero-Shot on 218 Korean Convenience Stores

| Model | NRMSE (%) |
|-------|:---------:|
| Korean-700 (ours) | 12.30 |
| BB SOTA-M | 13.14 |

## Paper & Reproducibility

- **Paper (Applied Energy, submitted)**: [`docs/paper_final.md`](docs/paper_final.md) · [`docs/paper_final.docx`](docs/paper_final.docx)
- **Graphical Abstract**: [`docs/graphical_abstract.png`](docs/graphical_abstract.png)
- **File map for full reproduction**: [`docs/PAPER_FILES_MANIFEST.md`](docs/PAPER_FILES_MANIFEST.md)
- **Results provenance (per-table checkpoint mapping)**: [`results/RESULTS_REGISTRY.md`](results/RESULTS_REGISTRY.md)
- **Patent draft**: [`docs/patent_draft_v1.md`](docs/patent_draft_v1.md)

## Installation

```bash
pip install -r requirements.txt
```

## Download Evaluation Data

```bash
bash scripts/download_bb_data.sh
```

The BuildingsBench evaluation data (CC-BY 4.0, NREL) will be downloaded to `external/BuildingsBench_data/`.

## Training

```bash
python scripts/train.py \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --train_index train_weekly_50.csv \
  --val_index val_weekly_50.csv \
  --max_steps 18000 --warmup_steps 500 \
  --augment --bb_eval_interval 1 --seed 42
```

## Evaluation

```bash
python scripts/evaluate_bb.py \
  --checkpoint checkpoints/best.pt \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --commercial_only
```

## Citation

```bibtex
@article{kim2026sevenhundred,
  title={Seven Hundred Simulations Suffice: Matching a 900,000-Building Foundation Model through Operational Diversity in Zero-Shot Load Forecasting},
  author={Kim, Jeong-Uk},
  journal={Applied Energy},
  year={2026}
}
```

## Acknowledgment

This work was supported by the research project funded by the Ministry of Trade, Industry and Energy, Korea Institute of Energy Technology Evaluation and Planning (KETEP).

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

The BuildingsBench evaluation data is licensed under CC-BY 4.0 by NREL (Patrick Emami, Peter Graf).
