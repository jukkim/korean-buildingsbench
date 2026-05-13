# 논문 작성 계획 (SSOT)

> 이 문서가 논문 작성의 **유일한 기준 문서**. 최종 갱신: 2026-05-04
> 논문 최종본: `docs/paper_final.md` | LaTeX: `paper/main.tex`
> Highlights: `docs/highlights_energy_ai.md` | Cover Letter: `docs/cover_letter_energy_ai.md`

---

## 1. 논문 기본 정보

| 항목 | 내용 |
|------|------|
| **타겟 저널** | ~~Applied Energy~~ → ~~Energy and Buildings~~ → ~~Building Simulation~~ → **Energy and AI** (Elsevier, Gold OA) |
| **제목** | Seven Hundred Simulations Suffice: Operational Diversity for Data-Efficient Zero-Shot Building Load Forecasting |
| **저자** | Jeong-Uk Kim (단독) |
| **소속** | Dept. of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea |
| **단어 수** | ~6,400 단어 (~195 word abstract) |
| **상태** | ⏳ **Energy and AI 투고 준비 중** (APEN→E&B→BS 연속 desk reject 후). GPT 4-round 리뷰 반영 완료 (2026-05-13) |

---

## 2. 확정된 실험 결과 (2026-05-04 최종)

모든 결과는 bb_repro 환경 (torch 2.0.1+cu118, sklearn 1.1.3, autocast, no clip)에서 통일 검증 완료.

### Main Results (Table 3)

| Model | Training Data | N Buildings | RevIN | NRMSE (%) | Δ baseline |
|-------|--------------|:-----------:|:-----:|:----------:|:-------:|
| BB-900K (baseline) | BB 900K | 900,000 | OFF | 13.27 | — |
| BB+RevIN | BB 900K | 900,000 | ON | 13.89 | +0.62 |
| **K-700 (ours)** | **Korean sim** | **700** | **ON** | **13.11 ± 0.17** (best 12.93) | **-0.16** |
| K-700 no RevIN | Korean sim | 700 | OFF | 14.72 ± 0.28 | +1.45 |
| **US-700 (ours)** | **US TMY sim** | **700** | **ON** | **13.64 ± 0.65** | **+0.37** |
| BB-700 aug | BB subset | 700 | ON | 14.26 | +0.99 |
| BB-700 | BB subset | 700 | ON | 15.28 | +2.01 |
| BB-700 OFF | BB subset | 700 | OFF | 16.44 | +3.17 |
| Persist. | — | — | — | 16.68 | +3.41 |

> 기준: val_best 체크포인트, 재평가 완료 (`results/RESULTS_REGISTRY.md` §Table 3).

### Data Design vs RevIN Decomposition (Table 4)

| Comparison | RevIN ON (5-seed) | RevIN OFF (3-seed) | RevIN Effect |
|------------|:-----------------:|:------------------:|:------------:|
| Korean-700 | 13.11% | 14.72% | -1.61 pp |
| BB-700 | 15.28% | 16.44% | -1.16 pp |
| **Data Design Effect** | **-2.17 pp** | **-1.72 pp** | — |

### N-Scaling (Table 5, seed=42, s=18000, _best.pt)

| n | Buildings | NRMSE (%) |
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

| Experiment | NRMSE | Change |
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
| BB-900K baseline | — | — | 13.14% |

### Multi-Seed Details (val_best, 재평가 후)

**s=18000 RevIN ON (5 seeds, 42~46)**: 12.93, 13.06, 13.10, 13.39, 13.07 → **13.11 ± 0.17%**
**s=18000 RevIN OFF (3 seeds, 42~44)**: 14.81, 14.94, 14.40 → **14.72 ± 0.28%**
**US-TMY ON (5 seeds, 42~46)**: 13.21, 13.03, 14.56, 13.78, 13.63 → **13.64 ± 0.65%**

> 기준: `_best.pt` (val_loss) 체크포인트 평가. 개별 seed 근거: `results/RESULTS_REGISTRY.md` §Table 3.

---

## 3. 핵심 발견

1. **Data sufficiency through design**: 700 LHS 시뮬이 benchmark-level 성능 달성 (13.11 ± 0.17% vs baseline 13.27%)
2. **데이터 설계 > RevIN**: data effect (2.17 pp @ ON / 1.72 pp @ OFF) > RevIN effect (1.61 pp)
3. **RevIN은 regime-dependent**: 소규모(700)에서 유효(-1.61pp), BB 900K에서 악화(+0.62pp)
4. **위도/경도 불필요**: lat=lon=0으로 동일 성능
5. **Cross-climate transfer**: Korean weather → US 건물 zero-shot 가능. US-TMY ablation도 BB-700 대비 우위
6. **N-scaling 포화**: n=5(70건)부터 benchmark-level, n=10(140건)부터 plateau
7. **실측 검증**: 한국 편의점 218건 zero-shot 12.30% vs BB 13.14%

---

## 4. 파일 위치

| 파일 | 용도 |
|------|------|
| `docs/paper_final.md` | **논문 최종본 (Markdown SSOT)** |
| `docs/paper_ae.docx` | **투고용 DOCX (Applied Energy 포맷)** |
| `docs/Highlights.docx` | **별도 Highlights 파일 (4 bullets, ≤85자)** |
| `docs/cover_letter.docx` | **Cover Letter** |
| `docs/graphical_abstract.png/pdf` | Graphical Abstract |
| `docs/fig{1,2,3,4}_*.png/pdf` | 논문 본문 Figure 4개 |
| `docs/PAPER_FILES_MANIFEST.md` | **재현 필요 파일 지도 SSOT** |
| `results/RESULTS_REGISTRY.md` | **수치 출처 SSOT** |

---

## 5. 제출 전 체크리스트

- [x] BB baseline 재현 (13.27%)
- [x] 통일 재평가 (bb_repro torch 2.0.1) — val_best 기준
- [x] BB 900K + RevIN (13.89%)
- [x] BB-700 ON/OFF + aug-matched (14.26%)
- [x] 편의점 218건 (Korean-700: 12.30% / BB: 13.14%)
- [x] N-scaling sweep (n=1~80)
- [x] Ablation 8개
- [x] Multi-seed (5-seed ON: 13.11±0.17, 3-seed OFF: 14.72±0.28)
- [x] **Exp-039: US-TMY/LHS ablation** (5-seed: 13.64±0.65%)
- [x] 논문 최종본 (`docs/paper_final.md`)
- [x] Graphical Abstract
- [x] Figure 4개 (BB-900K baseline 범례 통일)
- [x] References 30개
- [x] 특허 출원 (2026-05-01 변리사 제출)
- [x] 저널 포맷 DOCX (`docs/paper_ae.docx`, A4/TNR 12pt/더블스페이싱/행번호)
- [x] Highlights.docx (별도 파일)
- [x] Cover Letter (`docs/cover_letter.docx`)
- [x] 논문 리뷰 (수치 밀도 축소, SOTA→baseline 통일, 프레이밍 data-sufficiency 중심)
- [x] GitHub 공개 (`jukkim/korean-buildingsbench`, public)
- [x] GitHub 정리 (CLAUDE.md·특허·내부문서 제거, README 갱신)
- [ ] 편의점 서면 허가 확보
- [x] ~~Applied Energy 투고~~ (desk reject APEN-D-26-10043, scope mismatch)
- [x] **Energy and Buildings 이관** (Elsevier Transfer Service, 2026-05-06)
- [ ] E&B submission 완료 이메일 확인 → Subscription 선택

---

## 6. Exp-039: U.S.-TMY/LHS Ablation (완료)

**목적:** Korean weather와 LHS schedule의 confound 해소

**결과 (5-seed mean):** US-700 = **13.64 ± 0.65%** (vs BB-700 aug 14.26%)
- LHS 설계 효과 확인 (US weather에서도 BB-700 대비 0.62pp 우위)
- Korean-700 대비 +0.53pp → residual weather sensitivity 존재하나 주된 요인은 LHS 설계

---

## 7. 원고 수정 이력

### 2026-05-04 (최종 polish)
| 위치 | 변경 |
|------|------|
| 전체 | "SOTA" → "baseline"/"benchmark-level" (Section 2.2 정의 제외 전부 제거) |
| Abstract | Highlights 줄 제거 (별도 파일로 분리) |
| Abstract | "supporting operational diversity as a major driver" 반복 해소 |
| Section 2.2 | "state-of-the-art (SOTA)" → "benchmark result" |
| 전체 | 수치 밀도 축소 (Abstract 14→4개 핵심 수치, Discussion 교차참조 활용) |
| Figure 범례 | "BB SOTA-M" → "BB-900K baseline" (Fig 2, 3, 4, Graphical Abstract) |
| Conclusion | "outperform" → "reach benchmark-level performance ... sufficient for high-accuracy zero-shot forecasting" |
| Section 5.1 | RevIN 일반화 완화: "In this benchmark setting..." |
| Table 3 | 모델명 압축 (K-700, US-700, BB+RevIN, BB-700 aug, Persist.) |

### 2026-05-02
| 위치 | 변경 |
|------|------|
| Title | "Matching a 900,000-Building Foundation Model" → "Operational Diversity for Data-Efficient" |
| §1 | "unmetered buildings" → "buildings with limited historical data" |
| §5.1 | "1.61 pp" → "approximately 1.6 pp in the available multi-seed comparison" |
| §5.2 | "opposite of what power-law scaling would predict" → "differs from monotonic gains..." |
| §5.3 | "outperform U.S. ones" → "remain competitive on U.S. and Portuguese buildings" |
| §6 | "most seeds surpass" → "four of five seeds surpass the 900K baseline" |
| App. C | 통합 문장 → 4개 failure mode 별도 기계론적 설명 |
