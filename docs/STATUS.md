# BB_comparison 프로젝트 현황 (2026-02-18) — ⚠ SUPERSEDED

> **이 문서는 2026-02-18 시점의 역사적 기록입니다.** 논문 공식 수치는 `docs/paper_final.md` 및 `results/RESULTS_REGISTRY.md` 참조.
> 현재 공식: Korean-700 ON 5-seed **13.11 ± 0.16%** (best 12.93%) — BB SOTA 13.27%와 동등/초과.
> 이하 10-cap 검증 결과는 초기 PatchTST 실험(21.31% 갭 +8%p) 기반이므로 폐기됨.

## 핵심 발견: 10-cap 검증 결과

BB SOTA 13.31%는 데이터셋당 10건물만 평가한 결과 (`zero_shot.py` line 154: `if count == 10: break`).
동일 조건으로 우리 PatchTST를 평가한 결과:

| Model | Commercial NRMSE | 평가 건물수 | 비고 |
|-------|:---:|:---:|------|
| **BB Transformer-M** | **13.28%** | — | **BB SOTA (GitHub 공식)** |
| BB Transformer-L | 13.31% | ~50 (10-cap) | |
| BB Transformer-S | 13.97% | — | GitHub 공식 |
| Persistence Ensemble | 16.68% | ~50 (10-cap) | BB baseline |
| Previous Day Persistence | 21.09% | ~50 (10-cap) | |
| **우리 PatchTST (10-cap)** | **21.31%** | **50** | BB 프로토콜 |
| 우리 PatchTST (Full) | 23.33% | 742 | 기존 보고 |

### 데이터셋별 10-cap 결과

| Dataset | 10-cap | Full | N(10) | N(all) | 특이사항 |
|---------|:---:|:---:|:---:|:---:|------|
| bdg-2:fox | **13.42%** | 17.33% | 10 | 132 | SOTA 수준! |
| bdg-2:bear | 18.84% | 15.18% | 10 | 78 | |
| bdg-2:rat | 21.31% | 23.69% | 10 | 277 | |
| bdg-2:panther | 23.34% | 19.54% | 10 | 105 | |
| electricity | 35.76% | 33.11% | 10 | 150 | 포르투갈 전력, 도메인 갭 큼 |

**bdg-2:fox 10-cap = 13.42%** — BB SOTA(13.31%)에 근접! 하지만 10건물이라 분산이 큼 [10.37%, 18.54%].

### 결론

- **SOTA 갭: +8.00%p** (21.31% vs 13.31%)
- 10-cap은 full보다 약간 나은 경향 (-2.02%p)
- 데이터셋별 편차 큼 (13.42% ~ 35.76%)
- **순수 시계열 예측으로 BB SOTA를 이기기는 현실적으로 어려움**
  - BB: 900K 건물 학습 (350K 상업건물, 고유 스케줄)
  - 우리: 20K 시뮬 (4-16 건물 유형, 고정 스케줄)
  - 데이터 다양성 격차가 핵심 (물리 파라미터 아님)

## 전략적 방향: 경쟁 회피 → 차별화

**순수 시계열 예측 (168h→24h)은 BB의 영역.**
우리의 강점은 **EMS 전략 효과 예측** — BB가 할 수 없는 과제.

→ 상세: [STRATEGY.md](STRATEGY.md) 참조

## 관련 데이터 현황

→ 상세: [DATA_STATUS.md](DATA_STATUS.md) 참조

| 데이터셋 | 시뮬레이션 수 | Q&A 수 | 상태 |
|----------|:---:|:---:|------|
| QA (기본, EMS포함) | 956,081 | 503,807 | 20K 로컬 + 936K 클라우드 완료 (90만원) |
| QB (EMS+세부미터링) | 20,000 | 계획중 | 로컬 완료, 클라우드 ~1.5M 예정 |
| EnergyGPT 멀티모달 | - | 5,360,000 | v5 학습 진행중 |

## 파일 목록

| 파일 | 설명 |
|------|------|
| `scripts/verify_10cap.py` | 10-cap 검증 스크립트 (완료) |
| `docs/STATUS.md` | 이 문서 |
| `docs/DATA_STATUS.md` | Q1/Q2 데이터 현황 상세 |
| `docs/STRATEGY.md` | EMS 효과 예측 전략 |
| `docs/EXPERIMENT_LOG.md` | 실험 결과 기록 |
