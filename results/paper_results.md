# Korean_BB v3 — Incremental Training Results — ⚠ SUPERSEDED
*Updated: 2026-02-23 23:00 (955건 기준)*

> **이 문서는 2026-02-23 시점의 중간 결과입니다.** 논문 공식 수치는 `RESULTS_REGISTRY.md` (동일 폴더) 및 `../docs/paper_final.md` 참조.
> 여기의 v2.1/S1~S3 수치(19.31%, 16.86%, 17.43%, 18.21% 등)는 n=50 s=18000 val_best 체크포인트 방식으로 교체됨.

## 평가 기준
- **Commercial**: 955건 (BDG-2: bear 91 + fox 135 + panther 105 + rat 280 = 611, Electricity 344)
- **Residential**: 953건 (IDEAL 219, LCL 713, SMART 5, Sceaux 1, Borealis 15)
- **총합**: 1,908건
- **로딩 방식**: per-year 독립 로딩 (빈 파일 스킵, 연도별 NaN>50% 만 제외)
- **10-cap**: 사용 안 함 (BB README: "eval over all real buildings for all available years")

## BB Evaluation (Commercial CVRMSE, median)
| Model / Step | N Buildings | Commercial | Electricity | BDG-2 | Residential | Best Epoch | Gap vs SOTA |
|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BB Transformer-L** | 900,000 | 13.31% | 13.95% | — | 40.80% | — | — |
| **BB Transformer-M** | 900,000 | 13.95% | — | — | — | — | — |
| **Persistence Ensemble** | 900,000 | 16.68% | — | — | 37.58% | — | — |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Korean_BB v2.1 (baseline) | 2,097 | 19.31% | 10.84% | 20.99% | 85.71% | 50 | +6.00%p |
| Korean_BB v3 S1 | 3,017 | **16.86%** | 13.40% | 19.81% | 86.81% | 6 ✓ | +3.55%p |
| Korean_BB v3 S2 | 3,407 | **17.43%** | 13.83% | 20.12% | 83.24% | 14 | +4.12%p |
| Korean_BB v3 S3 | 3,867 | **18.21%** | 14.19% | 20.88% | 90.25% | — | +4.90%p |

*✓ = early stopped*
*BB eval: 955 commercial + 953 residential = 1,908건 총합 (per-year 독립 로딩)*

## Data Composition per Step
| Step | Total | office | retail | school | hotel | hospital | apt_mid | apt_high |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2.1 | 2,097 | 300 | 300 | 300 | 300 | 297 | 300 | 300 |
| v3 S1 | 3,017 | 760 | 760 | 300 | 300 | 297 | 300 | 300 |
| v3 S2 | 3,407 | 760 | 760 | 430 | 430 | 427 | 300 | 300 |
| v3 S3 | 3,867 | 760 | 760 | 430 | 430 | 427 | 300 | 300 |

## Training Details
| Step | Max Epochs | Actual Epochs | Best Epoch | Best Val Loss | Early Stop |
|:---|:---:|:---:|:---:|:---:|:---:|
| v2.1 | 50 | 50 | 39 | -4.125 | No |
| v3 S1 | 20 | 16 | 6 | -3.41170 | Yes |
| v3 S2 | 20 | 20 | 14 | -3.52280 | No |
| v3 S3 | 20 | — | — | — | — |

## 관찰 사항
- S1 (16.86%): SOTA 대비 +3.55%p — v2.1(+6.00%p) 대비 크게 개선
- S2→S3 상승 추이: 다양한 아키타입 추가 시 단기적 성능 하락 → 데이터 누적 필요
- Electricity S1=13.40%: BB Transformer-M(13.95%) 대비 우수!
- 목표: S10 이후 Commercial **14% 이하** (현재 +3.55~4.90%p 범위)
