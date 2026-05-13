# Korean_BB — 한국형 BuildingsBench

한국 건물 환경에서 EnergyPlus 시뮬레이션으로 학습하여 에너지 사용량을 예측하는 프로젝트.
BuildingsBench(NeurIPS'23) SOTA와 공정 비교 + 한국 특화 벤치마크 구축.

## 현황 조회 규칙 (세션 공통)

사용자가 **"현황"** 이라고 하면 4090과 5090 **양쪽 모두** 확인한다.

### 4090 (로컬, 192.168.50.1 또는 localhost)
```bash
# 진행 중인 실험 로그 (최신 exp034 로그 기준)
grep "step \|BB Commercial\|Cell " logs/exp034_cell6_runner.log | tail -5
# 또는 현재 실행 중인 로그 파일 확인 후 tail
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### 5090 (원격, IP: **192.168.1.23**, SSH: `user@192.168.1.23`)
> ⚠️ 192.168.50.54 는 잘못된 IP — 반드시 192.168.1.23 사용
```bash
ssh -o ConnectTimeout=8 user@192.168.1.23 "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader && tail -10 C:\\Korean_BB\\logs\\exp034_cells79_runner.log"
```
- 5090 SSH 불응 시: 절전/종료 상태로 표시 (ping으로 재확인)
- 5090 프로젝트 경로: `C:\Korean_BB\`
- 5090 로그 경로: `C:\Korean_BB\logs\`
- 5090 체크포인트: `C:\Korean_BB\checkpoints\`

### 현황 출력 형식
| 항목 | 머신 | 상태 |
|------|------|------|
| Cell N (nXk, sYK) | 4090 / 5090 | step X/Y, ETA Z / 완료 / 오류 |

### 프로세스 안전 실행 규칙
- **CMD 창 닫으면 학습 프로세스가 죽는다** (forrtl error 200)
- 안전한 실행: `python launch_hidden.py <bat파일> <로그파일>` — DETACHED_PROCESS로 창 없이 실행
- schtasks로 실행된 창도 닫으면 안전하지 않음 (interactive session)
- 모니터링: 언제든 로그 파일 tail로 확인
- **5090 주의**: DETACHED_PROCESS + stdout redirect = 0-byte 로그. SSH background(`&`)로 실행해야 로그 기록됨

## 평가 기준 (절대 규칙 — 세션 변경 시에도 준수)

### 체크포인트 기준: `_best.pt` (val_loss 기준) 사용
- **`_best.pt`** = Korean val split에서 val_loss 최저 시점 저장 → **공식 체크포인트**
- `_bb_best.pt` (inline BDG-2 기준) 는 test-set leakage 위험 → **논문에 사용 금지**
- `_last.pt` 는 재개용. BB 평가에 사용 금지
- **학습 시 `--bb_eval_interval 0`** 사용 (inline BB eval 비활성화)

### 평가: evaluate_bb.py로 사후 평가
- **평가 데이터**: Commercial **955건** (BDG-2:611 + Electricity:344)
- `python scripts/evaluate_bb.py --checkpoint <_best.pt> --commercial_only`
- 인라인 eval 결과는 참고용일 뿐, 논문 보고에 사용 금지

### 로그에서 결과 읽는 법
```bash
grep "Best BB Commercial CVRMSE" logs/<실험명>.log
# 출력 예: Best BB Commercial CVRMSE: 14.73% (gap vs SOTA: +1.42%p)
```

### 예외 처리 (인라인 소실 시)
- 0-byte 로그 등으로 인라인 소실 → `_bb_best.pt`로 `evaluate_bb.py` 실행 (BDG-2 611건)
- `_bb_best.pt` 도 없으면: 해당 셀 결과 **결측(n/a)** 처리. `_best.pt` 결과는 별도 표기

### 평가 단계 요약
| 단계 | 시점 | 평가셋 | 플래그 | 목적 |
|------|------|-------|-------|------|
| 인라인 | 매 epoch 학습 중 | BDG-2 611건 | (자동) | `_bb_best` 저장 + 실험 비교 공식값 |
| 예외 eval | 인라인 소실 시만 | BDG-2 611건 | `--bdg2_only` | 결측 셀 채우기 |
| 최종 full eval | 전체 실험 완료 후 best 1개만 | **전체 1908건** (commercial+residential) | (플래그 없음) | 논문 보고 |

## 현재 상태 (2026-05-08)

| 항목 | 상태 |
|------|------|
| 시뮬 데이터 | ✅ v1(5,750)+v2(2,097)+v3(3K~7K) + US-TMY(699) 완료 |
| 실험 | ✅ M/L 모델 n×steps 스윕, N-scaling, Ablation, Multi-seed 전부 완료 |
| **Exp-039 US-TMY** | ✅ 5-seed: **13.64 ± 0.65%** (LHS > weather 확인) |
| **논문** | ⏳ **Energy and AI 투고 준비 중** (APEN→E&B→BS 연속 desk reject, GPT 4-round 리뷰 반영 완료 2026-05-13) |
| **논문 제목** | "Operational Diversity by Design: A Parametric Simulation Methodology for Zero-Shot Building Load Forecasting" |
| **특허** | ✅ 2026-05-01 변리사 제출 완료 |
| **GitHub** | ✅ `jukkim/korean-buildingsbench` public, README·figures 최신 |
| **다음** | 리뷰 결과 대기 (EM Funding Info → KETEP 정정 필요) |

### 논문 공식 수치 (955건 Commercial, val_best, s=18000)
*수치 출처: `results/RESULTS_REGISTRY.md`. 모델명은 논문 Table 3 기준.*

| Model | N Buildings | NRMSE | Δ baseline | 비고 |
|-------|:---:|:---:|:---:|------|
| **BB-900K (baseline)** | 900K | **13.27%** | — | 논문 기준값 (원본 13.28%) |
| BB+RevIN | 900K | 13.89% | +0.62%p | RevIN asymmetry |
| **K-700 (ours, 5-seed)** | **700** | **13.11 ± 0.17%** | **-0.16%p** | **논문 공식 (best 12.93%)** |
| K-700 no RevIN (3-seed) | 700 | 14.72 ± 0.28% | +1.45%p | |
| **US-700 (ours, 5-seed)** | **700** | **13.64 ± 0.65%** | **+0.37%p** | **Exp-039 결과** |
| BB-700 aug | 700 | **14.26%** | +0.99%p | aug-matched, seed42 |
| BB-700 | 700 | 15.28% | +2.01%p | no aug |
| BB-700 OFF | 700 | 16.44% | +3.17%p | |
| Persist. | — | 16.68% | +3.41%p | |

### v2 시뮬레이션 결과
| 아키타입 | OK | FAIL | 비고 |
|---------|---:|---:|------|
| office | 300 | 0 | |
| retail | 300 | 0 | |
| school | 300 | 0 | |
| hotel | 300 | 0 | SCALE_MULT_CAP=1.8 |
| hospital | 297 | 3 | SCALE_MULT_CAP=1.5, 3건 plant temp runaway |
| apartment_midrise | 300 | 0 | |
| apartment_highrise | 300 | 0 | SCALE_MULT_CAP=2.0 |
| **합계** | **2,097** | **3** | **99.9% 성공** |

### 체크포인트
| 파일 | 용도 |
|------|------|
| `TransformerWithGaussian-M_v2.1_final_last.pt` | **최종 모델** (epoch 49, 2,097건) |
| `TransformerWithGaussian-M_v2.1_stage4_best.pt` | Stage 4 best (epoch 39, val_loss=-4.12520) |
| `TransformerWithGaussian-M_v2.1_stage1_best.pt` | Stage 1 best (epoch 26) |

## 프로젝트 구조

```
Korean_BB/
├── src/                  # 모델 코드
│   ├── models/           # TransformerWithGaussian (BB Transformer 재현)
│   ├── data/             # KoreanBBPretrainingDataset 로더
│   ├── eval/             # BB 프로토콜 평가
│   ├── idf/              # IDFModifier (외피/HVAC/스케줄 수정)
│   ├── schedules/        # 확률적 스케줄 생성기
│   └── transforms/       # Box-Cox, RevIN
├── scripts/
│   ├── generate_parametric_idfs.py  # v2 파라메트릭 IDF 생성 (8D LHS)
│   ├── run_simulations.py           # EnergyPlus 배치 시뮬
│   ├── postprocess.py               # CSV→Parquet, Box-Cox, 인덱스
│   ├── train.py                     # 학습 (resume, filter, num_workers)
│   ├── evaluate_bb.py               # BB 공식 평가
│   ├── check_progress.py            # 시뮬 진행 확인
│   └── generate_korean_idfs.py      # v1 IDF 생성 (레거시)
├── simulations/
│   ├── idfs_v2/          # v2 파라메트릭 IDF (2,100건)
│   ├── results_v2/       # v2 시뮬 결과
│   ├── idfs/             # v1 IDF (5,750건, 레거시)
│   └── results/          # v1 시뮬 결과
├── data/korean_bb/       # 학습 데이터 (Parquet)
├── checkpoints/          # 모델 체크포인트
├── results/              # BB 평가 결과 CSV
├── configs/              # 모델/빌딩 설정 TOML/YAML
├── docs/                 # 분석 문서 + 전략
├── external/             # BB 패키지 + 평가 데이터
│   ├── BuildingsBench/   # BB 패키지 (Python)
│   └── BuildingsBench_data/  # 평가 CSV (BDG-2, Electricity 등)
└── weather/              # TMY 기상 파일
```

## v3 파라메트릭 스케줄 시스템 (최신)

### 12D 파라미터 공간 (LHS)
| 파라미터 | 범위 | 의미 | v2→v3 변경 |
|---------|------|------|-----------|
| `op_start` | 0~12h | 운영 시작 시간 | 동일 |
| `op_duration` | 8~24h | 운영 지속 시간 | 동일 |
| `baseload_pct` | 25~98% | 비운영시 부하 비율 | UP: 10→25% |
| `weekend_factor` | 0~1.2 | 주말/주중 비율 | 동일 |
| `ramp_hours` | 0.5~4h | 전환 경사도 | 동일 |
| `equip_always_on` | 30~95% | 상시 가동 장비 비율 | UP: 20→30, 85→95 |
| `daily_noise_std` | 5~35% | 일별 노이즈 | UP: 25→35 |
| `scale_mult` | 0.3~3.0 | 부하 밀도 스케일 | 동일 |
| `night_equipment_frac` | 30~95% | **야간 장비 잔류 비율** | NEW (Fix A) |
| `weekly_break_prob` | 0~25% | **주별 패턴 깨짐 확률** | NEW (Fix B) |
| `seasonal_amplitude` | 0~30% | **계절별 부하 진폭** | NEW (Fix B) |
| `process_load_frac` | 0~50% | **상시 공정부하 비율** | NEW (Fix C) |

### 패턴 갭 수정 결과 (100건 검증)
| 지표 | v2 | v3 | BB(실측) | 판정 |
|------|:---:|:---:|:---:|:---:|
| Night/Day ratio | 0.512 | **0.852** | 0.803 | OK |
| Autocorr 168h | 0.938 | **0.784** | 0.751 | OK |
| Baseload P95 | 0.51 | **0.905** | 0.725 | OK |
| CV P5 | 0.21 | **0.032** | 0.105 | OK |

### 다중 도시 지원
```python
CITIES = {
    'seoul':     {'climate_zone': 'central_2', 'weather': 'KOR_Seoul.epw'},
    'busan':     {'climate_zone': 'southern',  'weather': 'KOR_Busan.epw'},
    'daegu':     {'climate_zone': 'southern',  'weather': 'KOR_Daegu.epw'},
    'gangneung': {'climate_zone': 'central_1', 'weather': 'KOR_Gangneung.epw'},
    'jeju':      {'climate_zone': 'jeju',      'weather': 'KOR_Jeju.epw'},
}
```

### Data Augmentation (학습 시)
- **Window jitter**: context window ±1~6h shift
- **Gaussian noise**: Box-Cox 후 N(0, 0.02) 추가
- **Amplitude scaling**: U(0.85, 1.15) 곱
- 활성화: `python scripts/train.py --augment`

## v2 파라메트릭 스케줄 시스템 (레거시, 8D LHS)

SCALE_MULT_CAP: office/retail/school=3.0, hotel=1.8, hospital=1.5, midrise=2.5, highrise=2.0
파라미터: op_start(0~12h), op_duration(8~24h), baseload_pct(5~90%), weekend_factor(0~1.2),
ramp_hours(0.5~4h), equip_always_on(0~0.8), daily_noise_std(0~0.15), scale_mult(0.3~3.0)

## ems_transformer와의 차이

| | ems_transformer | Korean_BB |
|---|---|---|
| **목표** | EMS 전략 → 에너지 절감량 산정 | 한국 건물 → 에너지 사용량 예측 |
| **과제** | CFEE (반사실 EMS 효과) | 168h→24h 시계열 예측 |
| **비교 대상** | 없음 (새 벤치마크) | BuildingsBench SOTA |
| **시뮬 데이터** | 기존 956K (EMS 포함) | v2 파라메트릭 2,097건 |
| **모델** | PatchTST + Projector + LLM | TransformerWithGaussian-M |
| **평가** | 절감률 MAE, ROUGE-L | NRMSE, CRPS (BB 프로토콜) |

## 절대 규칙

### BB 평가 코드 작성 규칙 (위반 금지)
- **새 평가 코드 작성 전에 BB 공식 코드 먼저 읽기**: `external/BuildingsBench/buildings_bench/evaluation/` + `zero_shot.py` 직접 확인 후 구현
- 가정으로 구현 금지 — 원본 프로토콜에서 건물 수·로딩 방식·필터 기준 직접 확인
- 파일 수정 후 연관 파일 전수 확인 (evaluate_bb.py ↔ train.py ↔ MachineLearning/benchmark_ts.py ↔ CLAUDE.md 동기화)

## BB 공식 프로토콜 (준수 필수)
```
per-building: CVRMSE = sqrt(mean(SE)) / mean(|actual|)
aggregation: median across buildings
CI: bootstrap 50,000 reps (rliable library)
10-cap: 데이터셋당 최대 10건물
OOV: metadata/oov.txt 제외
normalization: Box-Cox (global fitted)
loss: Gaussian NLL
```

## 주요 커맨드

### v3 파이프라인
```bash
# Phase 1: 패턴 검증 (100건 테스트)
python scripts/generate_parametric_idfs.py --version v3 --archetype office --n-schedules 50 --cities seoul --vintages v1_pre1990,v5_2018_plus --output-dir simulations/idfs_v3_test --dry-run

# Phase 2a: IDF 생성 (아키타입별)
python scripts/generate_parametric_idfs.py --version v3 --archetype office --n-schedules 400 --cities seoul,busan,daegu,gangneung,jeju --vintages all
python scripts/generate_parametric_idfs.py --version v3 --archetype retail --n-schedules 400 --cities seoul,busan,daegu,gangneung,jeju --vintages all

# Phase 2b: 시뮬레이션
python scripts/run_simulations.py --idf-dir simulations/idfs_v3 --result-dir simulations/results_v3 --workers 10 --filter office

# Phase 2c: 후처리
python scripts/postprocess.py --version v3 --filter office --append
python scripts/postprocess.py --version v3 --fit-only
python scripts/postprocess.py --version v3 --index-only

# Phase 3: 학습 (augmentation + curriculum)
python scripts/train.py --config configs/model/TransformerWithGaussian-M-v3.toml --epochs 20 --augment --note v3_stageA --num_workers 0
python scripts/train.py --config configs/model/TransformerWithGaussian-M-v3.toml --epochs 20 --augment --resume <ckpt> --lr 3e-5 --note v3_stageB
```

### v3 S6+ 파이프라인 (새 표준 — 2026-02-24~)
```bash
# Phase 1: IDF 생성 — 글로벌 LHS 풀 사용 (중복 방지)
python scripts/generate_parametric_idfs.py \
  --version v3 --archetype office --n-schedules 200 \
  --cities seoul,busan,daegu,gangneung,jeju --vintages all \
  --use-pool --pool-target 10000 \
  --output-dir simulations/idfs_v3_s6

# Phase 2: 시뮬 (로컬 또는 AWS)
python scripts/run_simulations.py --idf-dir simulations/idfs_v3_s6 \
  --result-dir simulations/results_v3_s6 --workers 10

# 또는 AWS Batch (cloud/)
python cloud/orchestrate_batch.py --step s6 --bucket korean-bb-v3-{ACCOUNT}

# Phase 3: 후처리 (NPY 내보내기 포함)
python scripts/postprocess.py --version v3 \
  --result-dir simulations/results_v3_s6 \
  --idf-dir simulations/idfs_v3_s6 \
  --append --export-npy

# Phase 4: 학습
python scripts/train.py --config configs/model/TransformerWithGaussian-M-v3.toml \
  --augment --resume checkpoints/TransformerWithGaussian-M-v3_v3_s4_scratch_bb_best.pt \
  --lr 3e-5 --note v3_s6 --num_workers 2 --bb_eval_interval 5

# Phase 5: BB 평가
python scripts/evaluate_bb.py --checkpoint checkpoints/TransformerWithGaussian-M-v3_v3_s6_best.pt \
  --config configs/model/TransformerWithGaussian-M-v3.toml
```

### v2 파이프라인 (레거시)
```bash
python scripts/check_progress.py
python scripts/postprocess.py --result-dir simulations/results_v2 --filter <arch> --append
python scripts/train.py --config configs/model/TransformerWithGaussian-M.toml --epochs 50 --resume <ckpt>
python scripts/evaluate_bb.py --checkpoint <ckpt> --config configs/model/TransformerWithGaussian-M.toml
```

## 핵심 참고문헌
- [BuildingsBench (NeurIPS 2023)](https://arxiv.org/abs/2307.00142) — SOTA-M 13.28% / SOTA-L 13.31%
- Buildings-900K: NREL EULP 550K ResStock + 350K ComStock

## 제출 파일 (Energy and AI 투고 준비)
| 파일 | 용도 |
|------|------|
| **[paper/main.tex](paper/main.tex)** | **LaTeX 원본 (elsarticle, GPT 4-round 리뷰 반영)** |
| **[paper/main.pdf](paper/main.pdf)** | **컴파일된 PDF (29p)** |
| **[docs/highlights_energy_ai.md](docs/highlights_energy_ai.md)** | **Highlights (5개, ≤85 chars)** |
| **[docs/cover_letter_energy_ai.md](docs/cover_letter_energy_ai.md)** | **커버레터** |
| [docs/paper_final.md](docs/paper_final.md) | SSOT (편집용, 비제출) |

### 이전 제출본 (아카이브)
| 파일 | 저널 |
|------|------|
| [docs/blinded_manuscript.docx](docs/blinded_manuscript.docx) | Building Simulation (double-blind) |
| [docs/title_page.docx](docs/title_page.docx) | Building Simulation |
| [docs/cover_letter_building_simulation.docx](docs/cover_letter_building_simulation.docx) | Building Simulation |
| [docs/paper_final_enb_rejected_20260507.md](docs/paper_final_enb_rejected_20260507.md) | E&B |
| [docs/paper_final_backup_20260429.md](docs/paper_final_backup_20260429.md) | Applied Energy |

## 문서
| 문서 | 내용 |
|------|------|
| **[docs/PAPER_FILES_MANIFEST.md](docs/PAPER_FILES_MANIFEST.md)** | **⭐ 논문 재현 파일 지도 SSOT — 체크포인트·데이터·스크립트 위치 (2026-05-04)** |
| **[results/RESULTS_REGISTRY.md](results/RESULTS_REGISTRY.md)** | **⭐ 논문 수치 출처 SSOT — Table별 체크포인트·재현 로그** |
| **[docs/PAPER_PLAN.md](docs/PAPER_PLAN.md)** | **논문 작성 SSOT — 실험 계획·구조·방어 전략·일정** |
| **[docs/REVISION_LOG_20260513.md](docs/REVISION_LOG_20260513.md)** | GPT 4-round 리뷰 반영 이력 (2026-05-13) |
| **[docs/IMPROVEMENT_ROADMAP.md](docs/IMPROVEMENT_ROADMAP.md)** | 전략 방향 SSOT — 성능 갭 분해·의사결정 트리 (2026-03-16~) |
| **[docs/SIMULATION_REVIEW_v3.md](docs/SIMULATION_REVIEW_v3.md)** | 2026-02-24 표준 재검토 — sim_id·Tier A·LHS풀·마이그레이션 |
| [docs/STATUS.md](docs/STATUS.md) | (2026-02 시점, 논문 공식 수치는 RESULTS_REGISTRY 참조) |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | 실험 기록 (EXP-001~038) |
| [docs/SIMULATION_DESIGN.md](docs/SIMULATION_DESIGN.md) | 시뮬레이션 설계 문서 (2026-02) |
| [archive/docs/AI_COMPETITION_BRIEF.md](archive/docs/AI_COMPETITION_BRIEF.md) | AI 경진대회/논문 브리프 (archived) |
| [docs/RESIDENTIAL_SIMULATION_SURVEY.md](docs/RESIDENTIAL_SIMULATION_SURVEY.md) | 주거 부문 시뮬레이션 서베이 — 방법론·공개 데이터셋·IDF·성능 분석 |

## 폴더 정리 이력 (2026-04-23)

논문 완성 후 대규모 정리 수행. **필요 파일은 `docs/PAPER_FILES_MANIFEST.md` 참조**.

| 이동 | From | To | 파일 수 |
|------|------|-----|:------:|
| 재개용 체크포인트 | `checkpoints/*_last.pt` | `archive/checkpoints_redundant_20260423/` | 31 |
| deprecated 체크포인트 | `checkpoints/*_bb_best.pt` | 〃 | 17 |
| TensorBoard 로그 | `runs/` (폴더 제거) | `archive/runs_tensorboard_20260423/` | 386 |
| WSL/cloud 중간 | `scratch/` (폴더 제거) | `archive/scratch_wsl_20260423/` | - |
| Smoke test | `simulations/validation_test/` | `archive/simulations_validation_test_20260423/` | 65MB |

**결과**: checkpoints 80→32개 (-8.5GB), 루트 19→17 폴더. 논문 재현에 필요한 48개 `_best.pt` + `bb900k_revin_step590000.pt` 모두 유지.

## 관련 프로젝트
| 프로젝트 | 경로 | 관계 |
|----------|------|------|
| ems_transformer | `../ems_transformer/` | EMS 절감 모델 (별개 목적) |
| BB 평가 데이터 | `external/BuildingsBench_data/` | 로컬 보유 |
| BB 패키지 | `external/BuildingsBench/` | Python 패키지 |
