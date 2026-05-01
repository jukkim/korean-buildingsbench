# Korean_BB — Paper Files Manifest (논문 재현 가능 파일 지도)

> **작성일**: 2026-04-23
> **목적**: Applied Energy 논문 `docs/paper_final.md` 의 모든 수치·도표·주장 재현에 필요한 파일을 한 눈에 찾기 위한 SSOT.
> **원칙**: 이 문서에 명시된 파일은 **절대 삭제·이동 금지**. 여기 없는 파일은 `archive/` 이동 가능.

---

## 0. Quick Locator (자주 찾는 파일)

| 찾는 것 | 위치 |
|---------|------|
| 논문 최종본 | `docs/paper_final.md`, `docs/paper_final.docx` |
| 특허 명세서 | `docs/patent_draft_v1.md`, `docs/patent_draft_v1.docx` |
| 그림 4개 + Graphical Abstract | `docs/fig{1,2,3,4}_*.png/pdf`, `docs/graphical_abstract.*` |
| 특허 도면 | `docs/patent_fig{1,2,3,4,5}_*.png` |
| 수치 출처 전수 | `results/RESULTS_REGISTRY.md` |
| 논문 작성 계획 | `docs/PAPER_PLAN.md` |
| 평가 데이터 (BB 955건) | `external/BuildingsBench_data/` |
| 평가 스크립트 | `scripts/evaluate_bb.py` |
| Box-Cox 변환기 | `data/korean_bb/metadata/transforms/boxcox.pkl` |

---

## 1. 논문 문서 (Paper & Patent)

### 1.1 논문 본문
| 파일 | 용도 | 보존 |
|------|------|:----:|
| `docs/paper_final.md` | 최종 논문 (Applied Energy 투고용) | ⭐ |
| `docs/paper_final.docx` | Word 변환본 | ⭐ |
| `docs/PAPER_PLAN.md` | 논문 작성 SSOT (실험 계획·방어 전략) | ⭐ |
| `docs/PAPER_FILES_MANIFEST.md` | 이 파일 | ⭐ |

### 1.2 특허 문서
| 파일 | 용도 | 보존 |
|------|------|:----:|
| `docs/patent_draft_v1.md` | 특허 명세서 (청구항 13개) | ⭐ |
| `docs/patent_fig1_system.png` | 도 1 — 시스템 구성도 | ⭐ |
| `docs/patent_fig2_lhs.png` | 도 2 — LHS 파라미터 생성 | ⭐ |
| `docs/patent_fig3_revin.png` | 도 3 — RevIN + Transformer | ⭐ |
| `docs/patent_fig4_nscaling.png` | 도 4 — N-scaling 그래프 | ⭐ |
| `docs/patent_fig5_comparison.png` | 도 5 — 성능 비교 | ⭐ |

### 1.3 논문 Figure (본문용)
| 파일 | 논문 위치 | 보존 |
|------|----------|:----:|
| `docs/fig1_pipeline.png/pdf` | Fig 1 — 전체 파이프라인 | ⭐ |
| `docs/fig2_comparison.png/pdf` | Fig 2 — 성능 비교 bar chart | ⭐ |
| `docs/fig3_nscaling_new.png/pdf` | Fig 3 — N-scaling 곡선 | ⭐ |
| `docs/fig4_revin_asymmetry.png/pdf` | Fig 4 — RevIN 비대칭 효과 | ⭐ |
| `docs/graphical_abstract.png/pdf` | Applied Energy 투고용 | ⭐ |

### 1.4 보조 문서
| 파일 | 용도 | 보존 |
|------|------|:----:|
| `docs/EXPERIMENT_LOG.md` | 실험 전체 기록 | ⭐ |
| `docs/IMPROVEMENT_ROADMAP.md` | 전략 방향 SSOT | ⭐ |
| `docs/SIMULATION_DESIGN.md` | 시뮬레이션 설계 문서 | ⭐ |
| `docs/SIMULATION_REVIEW_v3.md` | v3 재검토 문서 | ⭐ |
| `docs/STATUS.md` | 10-cap 검증 결과 | ⭐ |
| `results/RESULTS_REGISTRY.md` | **모든 수치의 출처 · 체크포인트 · 재현 로그 SSOT** | ⭐⭐ |

---

## 2. 모델 체크포인트 (논문 테이블 → 체크포인트 매핑)

> **체크포인트 명명 규칙**: `*_best.pt` = val_loss 기준 (논문 공식). `_last.pt` / `_bb_best.pt` 는 재현에 불필요.

### 2.1 Table 3 (Main Comparison) — 10개 체크포인트

| 모델 | Checkpoint 파일 | NRMSE |
|------|----------------|:-----:|
| BB 900K+RevIN | `checkpoints/bb900k_revin_step590000.pt` | 13.89% |
| Korean-700 ON (seed42) | `checkpoints/TransformerWithGaussian-M-v3-3k_ms_n50_s18000_revin_on_seed42_best.pt` | 12.93% |
| Korean-700 ON (seed43) | `...ms_n50_s18000_revin_on_seed43_best.pt` | 13.06% |
| Korean-700 ON (seed44) | `...ms_n50_s18000_revin_on_seed44_best.pt` | 13.10% |
| Korean-700 ON (seed45) | `...retrain_on_45_best.pt` | 13.39% |
| Korean-700 ON (seed46) | `...ms_n50_s18000_revin_on_seed46_best.pt` | 13.07% |
| Korean-700 OFF (seed42) | `...retrain_off_42_best.pt` | 14.81% |
| Korean-700 OFF (seed43) | `...retrain_off_43_best.pt` | 14.94% |
| Korean-700 OFF (seed44) | `...ms_n50_s18000_revin_off_seed44_best.pt` | 14.40% |
| BB-700 (aug-matched) ON | `...bb700_aug_s18000_seed42_best.pt` | 14.26% |
| BB-700 (no aug) ON | `...bb700_s18000_revin_on_best.pt` | 15.28% |
| BB-700 OFF | `...bb700_s18000_revin_off_best.pt` | 16.44% |

### 2.2 Table 4 (N-Scaling) — 11개 체크포인트

| n | Buildings | Checkpoint | NRMSE |
|:-:|:---------:|-----------|:-----:|
| 1 | 14 | `...nscale_n1_s18000_valbest_best.pt` | 14.72% |
| 5 | 70 (5-seed) | `...nscale_n5_s18000_valbest_best.pt` + `n5_s18000_seed{43,44,45,46}_best.pt` | 13.28±0.12% |
| 6 | 84 | `...nscale_n6_s18000_valbest_best.pt` | 13.33% |
| 7 | 98 | `...nscale_n7_s18000_valbest_best.pt` | 13.21% |
| 8 | 112 | `...nscale_n8_s18000_valbest_best.pt` | 13.27% |
| 9 | 126 | `...nscale_n9_s18000_valbest_best.pt` | 13.26% |
| 50 | 700 | = Table 3 Korean-700 ON seed42 | 12.93% |

> n=2,3,4,10,20,30,40,60,70,80 결과는 **5090 원격 장비**(192.168.1.23)에서 학습됨 — 체크포인트는 `C:\Korean_BB\checkpoints\` (5090 로컬). 논문에는 수치만 인용.

### 2.3 Appendix (s=16000 multi-seed) — 8개 체크포인트

| Seed | RevIN ON | RevIN OFF |
|:----:|:--------:|:---------:|
| 42 | `ms_n50_s16000_revin_on_seed42_best.pt` | `ms_n50_s16000_revin_off_seed42_best.pt` |
| 43 | `ms_n50_s16000_revin_on_seed43_best.pt` | `ms_n50_s16000_revin_off_seed43_best.pt` |
| 44 | `ms_n50_s16000_revin_on_seed44_best.pt` | `ms_n50_s16000_revin_off_seed44_best.pt` |
| 45 | `ms_n50_s16000_revin_on_seed45_best.pt` | — |
| 46 | `ms_n50_s16000_revin_on_seed46_best.pt` | — |

### 2.4 체크포인트 보존 규칙

- **⭐ 보존**: `*_best.pt` (48개) + `bb900k_revin_step590000.pt` (1개) = **총 49개**
- **⬇ 아카이브**: `*_last.pt` (31개, resume용), `*_bb_best.pt` (17개, deprecated inline eval) = **총 48개** → `archive/checkpoints_redundant_20260423/`

---

## 3. 설정 파일 (Configs)

| 파일 | 용도 | 보존 |
|------|------|:----:|
| `configs/model/TransformerWithGaussian-M-v3-3k.toml` | Main M 모델 config (RevIN ON) | ⭐ |
| `configs/model/TransformerWithGaussian-M-v3-3k-norevin.toml` | M 모델 RevIN OFF | ⭐ |
| `configs/model/TransformerWithGaussian-M-v3-3k-ctx336.toml` | Ablation (context=336) | ⭐ |
| `configs/model/TransformerWithGaussian-M-v3-3k-decomp.toml` | Ablation (seasonal decomp) | ⭐ |
| `configs/model/TransformerWithGaussian-L-*.toml` | L 모델 variants | ⭐ |
| `configs/korean_buildings.yaml` | 14 archetype 정의 | ⭐ |
| `configs/schedule_params.yaml` | 12D LHS parameter 범위 | ⭐ |
| `configs/idf_mapping.yaml` | IDF 템플릿 매핑 | ⭐ |
| `configs/lhs_pool/` | Pre-generated LHS samples (중복 방지) | ⭐ |

---

## 4. 데이터 파일 (Data)

### 4.1 Korean 시뮬레이션 데이터
| 경로 | 내용 | 크기 | 보존 |
|------|------|------|:----:|
| `data/korean_bb/metadata/catalog.csv` | 192K building 카탈로그 | - | ⭐ |
| `data/korean_bb/metadata/transforms/boxcox.pkl` | Global Box-Cox (λ=-0.067) | - | ⭐ |
| `data/korean_bb/metadata/oov.txt` | OOV 건물 목록 | - | ⭐ |
| `data/korean_bb/metadata/summary.json` | 통계 요약 | - | ⭐ |
| `data/korean_bb/metadata/encoding_maps.json` | 카테고리 인코딩 | - | ⭐ |
| `data/korean_bb/metadata/train_weekly_{N}.csv` | n별 학습 인덱스 (n=1~90, 100~10k) | - | ⭐ |
| `data/korean_bb/metadata/val_weekly_{N}.csv` | n별 검증 인덱스 | - | ⭐ |
| `data/korean_bb/individual/*.parquet` | 198K parquet (전체 시뮬 데이터) | ~40GB | ⭐ |
| `simulations/npy_tier_a/` | NPY Tier A (17채널, 146K 건물) | ~80GB | ⭐ |

### 4.2 BB-700 평가용 subset
| 경로 | 내용 | 보존 |
|------|------|:----:|
| `data/bb_subset/seed42_n700/` | BB ComStock 700-building subset | ⭐ |

### 4.3 BB 평가 데이터 (외부)
| 경로 | 내용 | 보존 |
|------|------|:----:|
| `external/BuildingsBench/` | BB Python 패키지 | ⭐ |
| `external/BuildingsBench_data/` | BB 평가 CSV (BDG-2, Electricity, Residential) | ⭐ |

### 4.4 기상 데이터
| 경로 | 내용 | 보존 |
|------|------|:----:|
| `weather/` | 한국 5개 도시 TMY | ⭐ |

---

## 5. 핵심 스크립트 (논문 재현 필수)

### 5.1 데이터 생성 파이프라인
| 스크립트 | 용도 | 보존 |
|---------|------|:----:|
| `scripts/generate_parametric_idfs.py` | 12D LHS → IDF 생성 | ⭐ |
| `scripts/run_simulations.py` | EnergyPlus 병렬 실행 | ⭐ |
| `scripts/extract_npy.py` | CSV → NPY Tier A 변환 | ⭐ |
| `scripts/postprocess.py` | CSV→Parquet, Box-Cox fit, 인덱스 생성 | ⭐ |
| `scripts/resample_to_nk.py` | n별 train/val 인덱스 생성 | ⭐ |
| `scripts/add_npy_to_catalog.py` | NPY → catalog 등록 | ⭐ |

### 5.2 학습 / 평가
| 스크립트 | 용도 | 보존 |
|---------|------|:----:|
| `scripts/train.py` | 메인 학습 스크립트 | ⭐ |
| `scripts/evaluate_bb.py` | BB 955건 평가 (논문 공식) | ⭐ |
| `scripts/eval_korean_stores.py` | 편의점 218건 평가 (Table 6) | ⭐ |
| `scripts/train_bb900k_revin.py` | BB 900K + RevIN 학습 | ⭐ |
| `scripts/bb_subsample_build_dataset.py` | BB-700 subset 빌드 | ⭐ |

### 5.3 시각화
| 스크립트 | 용도 | 보존 |
|---------|------|:----:|
| `scripts/generate_graphical_abstract.py` | Graphical Abstract 생성 | ⭐ |
| `scripts/plot_scaling.py` | Fig 3 N-scaling 곡선 생성 | ⭐ |

---

## 6. 소스 코드 (모델/데이터/평가 구현)

| 경로 | 내용 | 보존 |
|------|------|:----:|
| `src/models/` | TransformerWithGaussian + RevIN 구현 | ⭐ |
| `src/data/` | KoreanBBPretrainingDataset 로더 | ⭐ |
| `src/idf/` | IDFModifier (외피/HVAC/스케줄 수정) | ⭐ |
| `src/schedules/` | 확률적 스케줄 생성기 | ⭐ |
| `src/buildings/` | 건물 archetype 정의 | ⭐ |

> `src/eval/` 및 `src/transforms/` 는 별도 모듈 없이 `scripts/evaluate_bb.py` 및 `scripts/postprocess.py` 에 직접 구현됨.

---

## 7. 결과 파일 (Metrics CSV)

| 파일 | 논문 용도 | 보존 |
|------|----------|:----:|
| `results/bb_metrics_TransformerWithGaussian-M-v3-3k.csv` | 논문 Main M 결과 | ⭐ |
| `results/bb_metrics_TransformerWithGaussian-M-v3-3k-norevin.csv` | RevIN OFF 결과 | ⭐ |
| `results/bb_metrics_TransformerWithGaussian-M-v3-3k-ctx336.csv` | Ablation ctx=336 | ⭐ |
| `results/bb_metrics_TransformerWithGaussian-M-v3-3k-decomp.csv` | Ablation seasonal decomp | ⭐ |
| `results/bb_metrics_TransformerWithGaussian-L-*.csv` | L 모델 supplementary | ⭐ |
| `results/RESULTS_REGISTRY.md` | 수치 출처 SSOT | ⭐⭐ |

---

## 8. 로그 파일 (재현 검증용)

| 파일 | 내용 | 보존 |
|------|------|:----:|
| `logs/valbest_all.log` | Korean-700 seed42~46 평가 로그 | ⭐ |
| `logs/n1_n5_valbest_main.log` | N-scaling n=1, n=5 | ⭐ |
| `logs/n5_multiseed_main.log` | n=5 multi-seed 학습 | ⭐ |
| `logs/n6789_valbest_main.log` | n=6,7,8,9 학습 | ⭐ |
| `logs/retrain_broken_main.log` | broken seed 재학습 | ⭐ |
| `logs/korean_stores_valbest.log` | 편의점 218건 평가 | ⭐ |
| `logs/s16000_valbest_eval.log` | Steps sweep s=16000 | ⭐ |

---

## 9. 아카이브된 파일 (2026-04-23 정리)

> 아래 파일은 `archive/` 로 이동됨. 재현이 필요하면 archive에서 복원.

### 9.1 `archive/checkpoints_redundant_20260423/`
- `*_last.pt` 31개 — 학습 재개용 (논문 재현에는 불필요)
- `*_bb_best.pt` 17개 — deprecated inline BB eval 체크포인트

### 9.2 `archive/runs_tensorboard_20260423/`
- `runs/` 386개 — TensorBoard 로그 (최종 결과는 이미 `checkpoints/`와 `results/`에 저장됨)

### 9.3 `archive/scratch_wsl_20260423/`
- `scratch/` 전체 — WSL/cloud 실험 중간 파일 (bb_fresh, github_datasets 등)

### 9.4 `archive/simulations_validation_test_20260423/`
- `simulations/validation_test/` 65MB — v1/v3/v5 초기 smoke test

### 9.5 `archive/docs/` (기존)
- PAPER_DRAFT.md, paper_v1.md, paper_v2.md, paper_summary_kr.md — 이전 버전
- BOUNTY_HUNTER_REVIEW_v1.md, GPT_REVIEW_EXP031.md — 과거 리뷰
- CLEANUP_LOG.md, GITHUB_CLEANUP.md — 과거 정리 기록
- AI_COMPETITION_BRIEF.md, RESEARCH_VALLOSS_VS_BBEVAL.md — 참고

### 9.6 `archive/scripts/` (기존)
- run_n*_valbest*.py, retrain_broken_seeds.py — 과거 실험 스크립트

---

## 10. 재현 Quick Start

```bash
# 1. 환경 준비
conda activate bb_repro  # torch 2.0.1, sklearn 1.1.3

# 2. BB SOTA 재현 (13.27%)
python scripts/evaluate_bb.py \
  --checkpoint external/BuildingsBench_data/checkpoints/Transformer_Gaussian_M.pt \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --commercial_only

# 3. Korean-700 재현 (12.93% best seed)
python scripts/evaluate_bb.py \
  --checkpoint checkpoints/TransformerWithGaussian-M-v3-3k_ms_n50_s18000_revin_on_seed42_best.pt \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --commercial_only

# 4. N-scaling (n=5, 70 buildings)
python scripts/evaluate_bb.py \
  --checkpoint checkpoints/TransformerWithGaussian-M-v3-3k_nscale_n5_s18000_valbest_best.pt \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --commercial_only

# 5. 편의점 218건 (Table 6)
# --store-dir: 편의점별 서브디렉토리 + *_hourly_labeled.csv 포함 디렉토리 (비공개 데이터)
python scripts/eval_korean_stores.py \
  --store-dir /path/to/korean_stores \
  --checkpoint checkpoints/TransformerWithGaussian-M-v3-3k_ms_n50_s18000_revin_on_seed42_best.pt
```

---

## 11. 참조

| 관련 문서 | 경로 |
|----------|------|
| 논문 본문 | `docs/paper_final.md` |
| 논문 계획 | `docs/PAPER_PLAN.md` |
| 수치 출처 | `results/RESULTS_REGISTRY.md` |
| 실험 기록 | `docs/EXPERIMENT_LOG.md` |
| 시뮬 설계 | `docs/SIMULATION_DESIGN.md` |
| 프로젝트 개요 | `CLAUDE.md` |

---

*최초 작성: 2026-04-23 | 작성: Claude Code (정리 작업 2026-04-23)*
