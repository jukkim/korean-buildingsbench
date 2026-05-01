# Applied Energy 논문 작성 계획 (SSOT)

> 이 문서가 논문 작성의 **유일한 기준 문서**. 최종 갱신: 2026-04-21
> 논문 최종본: `docs/paper_final.md`
> 특허 초안: `docs/patent_draft_v1.md`

---

## 1. 논문 기본 정보

| 항목 | 내용 |
|------|------|
| **타겟 저널** | Applied Energy (Elsevier, IF 12.81) 또는 Energy & AI (IF 21.0) |
| **제목** | Seventy Simulations Suffice: Matching a 900,000-Building Foundation Model through Operational Diversity in Zero-Shot Load Forecasting |
| **저자** | Jeong-Uk Kim (단독) |
| **소속** | Dept. of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea |
| **단어 수** | ~7,200 단어 |
| **상태** | **초안 완료 (paper_final.md)** |

---

## 2. 확정된 실험 결과 (2026-04-21 최종)

모든 결과는 bb_repro 환경 (torch 2.0.1+cu118, sklearn 1.1.3, autocast, no clip)에서 통일 검증 완료.

### Main Results (Table 3)

| Model | Training Data | N Buildings | RevIN | CVRMSE (%) | vs SOTA |
|-------|--------------|:-----------:|:-----:|:----------:|:-------:|
| BB SOTA-M (reproduced) | BB 900K | 900,000 | OFF | 13.27 | — |
| BB 900K + RevIN | BB 900K | 900,000 | ON | 13.89 | +0.62 |
| **Korean-700 (ours)** | **Korean sim** | **700** | **ON** | **13.11 ± 0.16** (best 12.93) | **-0.16** |
| Korean-700 | Korean sim | 700 | OFF | 14.72 ± 0.28 | +1.45 |
| BB-700 | BB subset | 700 | ON | 15.28 | +2.01 |
| BB-700 | BB subset | 700 | OFF | 16.44 | +3.17 |
| Persistence Ensemble | — | — | — | 16.68 | +3.41 |

> 기준: val_best 체크포인트, 재평가 완료 (`results/RESULTS_REGISTRY.md` §Table 3).

### Data Design vs RevIN Decomposition (Table 4)

| Comparison | RevIN ON (5-seed) | RevIN OFF (3-seed) | RevIN Effect |
|------------|:-----------------:|:------------------:|:------------:|
| Korean-700 | 13.11% | 14.72% | -1.61 pp |
| BB-700 | 15.28% | 16.44% | -1.16 pp |
| **Data Design Effect** | **-2.17 pp** | **-1.72 pp** | — |

### N-Scaling (Table 5, seed=42, s=18000, _best.pt)

| n | Buildings | CVRMSE (%) |
|:-:|:---------:|:----------:|
| 1 | 14 | 14.72 |
| 3 | 42 | 13.47 |
| 5 | 70 | 13.28 ± 0.12 (5-seed) |
| 10 | 140 | 13.18 |
| 20 | 280 | 13.23 |
| 50 | 700 | 12.93 |
| 70 | 980 | 13.20 |
| 80 | 1,120 | 13.15 |

> 전체 n=1~80 곡선(17포인트): `results/RESULTS_REGISTRY.md` §Table 4.

### Ablation (Table 6)

| Experiment | CVRMSE | Change |
|------------|:------:|:------:|
| Korean-700 RevIN ON (best seed=42) | 12.93% | — |
| RevIN OFF (3-seed mean) | 14.72% | +1.79 |
| BB Box-Cox for training | 16.24% | +3.31 |
| 4x training tokens | 16.02% | +3.09 |
| 5K cap per archetype | 15.35% | +2.42 |
| Seasonal decomposition | 16.65% | +3.72 |
| Context 336h | 13.29% | +0.36 |
| lat/lon = 0 | 12.93% | 0.00 |

### Korean Convenience Store (Table 7)

| Model | 100 stores | 120 stores | All 218 |
|-------|:----------:|:----------:|:-------:|
| Korean-700 (ours) | 17.42% | 10.22% | 12.30% |
| BB SOTA-M | — | — | 13.14% |

### Multi-Seed Details (val_best, 재평가 후)

**s=18000 RevIN ON (5 seeds, 42~46)**: 12.93, 13.06, 13.10, 13.39, 13.07 → **13.11 ± 0.16%**
**s=18000 RevIN OFF (3 seeds, 42~44)**: 14.81, 14.94, 14.40 → **14.72 ± 0.28%**
**s=16000 RevIN ON (5 seeds)**: 12.89, 13.24, 13.16, 13.26, 13.12 → **13.13 ± 0.15%**
**s=16000 RevIN OFF (3 seeds)**: 14.25, 14.34, 14.46 → **14.35 ± 0.11%**

> 기준: `_best.pt` (val_loss) 체크포인트 평가. 개별 seed 근거: `results/RESULTS_REGISTRY.md` §Table 3.

---

## 3. 핵심 발견

1. **700건 ≈ 900K**: Korean-700 RevIN ON 5-seed 13.11 ± 0.16% (best 12.93%) ≈ BB SOTA 13.27% — 0.08% 데이터로 SOTA 동등/초과
2. **데이터 설계 > RevIN**: data effect (2.17 pp @ ON / 1.72 pp @ OFF) > RevIN effect (1.61 pp)
3. **RevIN은 소규모에서만 유효**: BB 900K + RevIN (13.89%) > BB 900K (13.27%) — 대규모에서 오히려 악화
4. **위도/경도 불필요**: lat=lon=0으로 동일 성능
5. **기후 차이는 강점**: 한국 기후로 학습 → 미국 건물 예측이 더 정확 → operational diversity가 기후를 초월
6. **N-scaling 포화**: n=5(70건)부터 SOTA 동등, n=10(140건)부터 plateau
7. **실측 검증**: 한국 편의점 218건 zero-shot 12.30% vs BB 13.14%

---

## 4. 파일 위치

| 파일 | 용도 |
|------|------|
| `docs/paper_final.md` | **논문 최종본 (Applied Energy 투고용 Markdown SSOT)** |
| `docs/paper_final.docx` | Word 변환본 |
| `docs/graphical_abstract.png/pdf` | Graphical Abstract |
| `docs/fig{1,2,3,4}_*.png/pdf` | 논문 본문 Figure 4개 |
| `docs/patent_draft_v1.md` | 특허 명세서 |
| `docs/patent_fig{1,2,3,4,5}_*.png` | 특허 도면 5개 |
| `docs/PAPER_FILES_MANIFEST.md` | **⭐ 재현 필요 파일 지도 SSOT (2026-04-23)** |
| `results/RESULTS_REGISTRY.md` | **⭐ 수치 출처 SSOT** |
| `archive/docs/` | 이전 버전 (paper_v1/v2, PAPER_DRAFT, CLEANUP_LOG 등) |

---

## 5. 제출 전 체크리스트

- [x] BB SOTA 재현 (13.27%)
- [x] 통일 재평가 (bb_repro torch 2.0.1) — val_best 기준
- [x] BB 900K + RevIN (13.89%)
- [x] BB-700 ON/OFF
- [x] 편의점 218건 (Korean-700: 12.30% / BB: 13.14%)
- [x] N-scaling sweep (n=1~80)
- [x] Ablation 8개
- [x] Multi-seed (5-seed ON: 13.11±0.16, 3-seed OFF: 14.72±0.28)
- [x] 논문 초안 완성 (`docs/paper_final.md`)
- [x] 특허 초안 + 도면 5개
- [x] Graphical Abstract (`docs/graphical_abstract.png/pdf`)
- [x] Figure 생성 (Fig 1~4, matplotlib)
- [x] References 보강 (21→30개)
- [x] 폴더 정리 + MANIFEST 작성 (2026-04-23)
- [ ] 특허 출원 (논문 제출 전)
- [ ] 저널 포맷 변환 (LaTeX/Word)
- [ ] Supplementary material 준비
- [ ] Applied Energy 투고
