# Exp-039: U.S.-TMY/LHS Ablation

> 생성: 2026-05-02 | 완료: 2026-05-03 | 상태: ✅ 완료 — Case A 확인 (13.15%)
> 논문 SSOT: `docs/paper_final.md` | 수치 SSOT: `results/RESULTS_REGISTRY.md`

---

## 목적

Korean_BB 논문의 핵심 약점—**Korean weather와 LHS schedule design의 confound**—을 직접 실험으로 해소한다.

현재 원고에서 "operational diversity"가 핵심 기여라고 주장하지만, 실제로는 다음이 동시에 다름:
- Korean weather vs. U.S. weather
- LHS schedule vs. stock-model schedule

이 실험은 **LHS schedule을 고정하고 weather만 U.S.로 교체**하여 weather 기여를 분리한다.

---

## 핵심 질문

> **같은 LHS 스케줄 설계를 U.S. 날씨로 훈련해도 Korean-700과 비슷한 성능이 나오는가?**

| 결과 | 의미 | 논문 영향 |
|------|------|---------|
| US-TMY ≈ Korean-TMY (±0.3 pp) | LHS operational diversity가 weather보다 중요 | 핵심 주장 강화, major revision 방어 가능 |
| US-TMY > Korean-TMY (+0.5 pp↑) | Korean climate 기여 있음 | confound 솔직히 해소, 주장 소폭 약화 |
| US-TMY < Korean-TMY (−0.5 pp↓) | Korean climate이 유리하게 작용 | 흥미로운 발견, 추가 분석 필요 |

---

## 실험 설계

### 격리 원칙

| 변수 | Korean-700 | US-TMY-700 (이번 실험) |
|------|-----------|----------------------|
| 건물 아키타입 | 14종 × 50개 | **동일** |
| LHS 파라미터 | 12D, 기존 pool | **동일 (pool 재사용)** |
| 건물 코드 | Korean code | **동일** (weather만 교체) |
| 날씨 파일 | 5개 Korean TMY | **5개 U.S. TMY** |
| 모델 아키텍처 | Transformer-M 15.8M | **동일** |
| 학습 설정 | 18K steps, RevIN ON, aug ON | **동일** |
| Box-Cox | Korean-fitted boxcox.pkl | **동일 (재사용)** |
| 평가 | 955-building BB protocol | **동일** |

> HVAC autosizing 유의: Korean IDF를 U.S. weather로 돌릴 때 cooling/heating capacity가
> 안 맞을 수 있음. Phase 1 sanity check에서 에너지 프로파일 확인 필수.

### 도시 매핑

| Korean city (현재) | ASHRAE | U.S. city (이번 실험) | ASHRAE | 비고 |
|-------------------|--------|----------------------|--------|------|
| Seoul | 4A | Washington D.C. | 4A | 기후대 일치 (BB 평가셋 포함) |
| Busan | 3A | Atlanta, GA | 3A | 습윤아열대 일치 |
| Daegu | 3A | Charlotte, NC | 3A | 습윤아열대 |
| Gangneung | 4A coastal | Boston, MA | 5A | 근사 매핑 |
| Jeju | 2A subtropical | Miami, FL | 1A | 아열대 (가장 더운 쪽) |

---

## 단계별 실행 계획

### Phase 0 — 날씨 파일 준비 (1h)

```bash
mkdir -p weather/us
# EnergyPlus weather 다운로드: https://energyplus.net/weather
# 필요 파일 5개:
#   USA_DC_Washington.724050_TMY3.epw
#   USA_GA_Atlanta.722190_TMY3.epw
#   USA_NC_Charlotte.723140_TMY3.epw
#   USA_MA_Boston.725090_TMY3.epw
#   USA_FL_Miami.722020_TMY3.epw
```

### Phase 1 — run_simulations.py 수정 (1~2h)

`scripts/run_simulations.py`에 `--weather-override` 옵션 추가:

```python
# 추가 인자
parser.add_argument('--weather-override', type=str, default=None,
                    help='EPW 파일 경로로 IDF 내 weather 참조를 override')
parser.add_argument('--city-filter', type=str, default=None,
                    help='특정 city의 IDF만 처리 (e.g., "seoul")')
```

또는 더 단순하게: **city별로 result-dir를 분리해서 5번 실행**

```bash
# city별 결과 디렉토리에 각각 U.S. weather로 시뮬레이션
for CITY_KOR, CITY_US in [
    ("seoul",     "weather/us/USA_DC_Washington.724050_TMY3.epw"),
    ("busan",     "weather/us/USA_GA_Atlanta.722190_TMY3.epw"),
    ("daegu",     "weather/us/USA_NC_Charlotte.723140_TMY3.epw"),
    ("gangneung", "weather/us/USA_MA_Boston.725090_TMY3.epw"),
    ("jeju",      "weather/us/USA_FL_Miami.722020_TMY3.epw"),
]:
    python scripts/run_simulations.py \
      --idf-dir simulations/idfs_v3 \
      --result-dir simulations/results_us_tmylhs \
      --workers 10 \
      --weather-override $CITY_US \
      --city-filter $CITY_KOR
```

### Phase 2 — Sanity Check (30min)

아래 지표를 Korean-700 결과와 비교:
- 연간 총 전력 소비 분포 (kWh/yr)
- Night/Day ratio
- 계절별 부하 패턴

```bash
python scripts/check_progress.py --result-dir simulations/results_us_tmylhs
```

이상이 없으면 Phase 3 진행. 에너지 값이 비정상이면 (0 또는 무한대 다수) HVAC sizing 문제 → 설계 재검토.

### Phase 3 — 후처리 (30min)

```bash
python scripts/postprocess.py \
  --version v3 \
  --result-dir simulations/results_us_tmylhs \
  --append \
  --train-index-output data/korean_bb/metadata/train_weekly_us_tmylhs.csv
# Box-Cox: 기존 data/korean_bb/metadata/transforms/boxcox.pkl 재사용 (--no-refit)
```

### Phase 4 — 학습 (2h, 4090)

```bash
python scripts/train.py \
  --config configs/model/TransformerWithGaussian-M-v3.toml \
  --train-index data/korean_bb/metadata/train_weekly_us_tmylhs.csv \
  --augment \
  --note exp039_us_tmylhs_seed42 \
  --seed 42 \
  --num_workers 0 \
  --bb_eval_interval 0
```

seed 42 먼저. 결과가 13.xx% 범위면 seed 43, 44 추가 → 3-seed mean.

### Phase 5 — 평가 (30min)

```bash
python scripts/evaluate_bb.py \
  --checkpoint checkpoints/TransformerWithGaussian-M-v3_exp039_us_tmylhs_seed42_best.pt \
  --commercial_only
```

---

## 예상 소요 시간

| 단계 | 시간 |
|------|------|
| Phase 0 날씨 파일 준비 | 1h |
| Phase 1 코드 수정 | 1~2h |
| Phase 2 시뮬레이션 700건 (workers=10) | 4~6h |
| Phase 3 후처리 | 30min |
| Phase 4 학습 seed 42 | 2h |
| Phase 5 평가 | 30min |
| seed 43, 44 추가 (필요 시) | +4h |
| **합계 (seed 42만)** | **~1일** |
| **합계 (3-seed)** | **~1.5일** |

---

## 코드 수정 범위

| 파일 | 수정 내용 | 규모 |
|------|---------|------|
| `scripts/run_simulations.py` | `--weather-override`, `--city-filter` 옵션 추가 | ~20줄 |
| `scripts/postprocess.py` | `--no-refit` (Box-Cox 재사용), `--train-index-output` 옵션 | ~10줄 |
| `configs/` | `TransformerWithGaussian-M-v3.toml` 복사 후 train-index 경로만 변경 | 1줄 |

---

## 논문 반영 계획

### Table 3에 추가할 행

```
| US-TMY-700 (LHS) | U.S. TMY sim | 700 | ON | ON | TBD | TBD | TBD |
```

### 본문 수정 범위

| 위치 | 수정 내용 |
|------|---------|
| §4.2 (Decomposing) | Exp-039 결과 1~2문장 추가 |
| §5.2 (Why Diversity) | "U.S.-TMY/LHS ablation" 결과로 causal 주장 강화 |
| §5.4 (Limitations) | "U.S.-TMY/LHS ablation 실시; confound 해소" 로 업데이트 |

### 결과에 따른 주장 조정

**Case A (US-TMY ≈ Korean-TMY, ±0.3 pp)**
- §5.4 limitation "U.S.-TMY/LHS ablation was not conducted" → 삭제
- §5.2에 "U.S.-TMY/LHS ablation yields XX%, confirming that weather origin is not the primary driver" 추가
- Conclusion: "operational diversity generalizes across climate origins" 강화

**Case B (US-TMY < Korean-TMY, +0.5 pp↑)**
- §5.4에 "U.S.-TMY experiment yields XX%, suggesting Korean climate provides an additional XX pp advantage"
- 주장 완화: "LHS-designed operational diversity contributes substantially, with additional benefit from Korean climate alignment"

---

## 판단 기준 (실험 완료 후)

| US-TMY 결과 | 판단 |
|-------------|------|
| ≤ 13.5% | Case A: 논문 주장 강함 → 투고 준비 완료 |
| 13.5~14.0% | Case B: 논문 주장 약화, 표현 조정 후 투고 |
| > 14.0% | Korean climate 기여 크다 → §5 전면 재검토 필요 |

---

## 관련 파일

| 파일 | 용도 |
|------|------|
| `simulations/idfs_v3/` | 기존 Korean LHS IDF (재사용) |
| `simulations/results_us_tmylhs/` | 이번 실험 시뮬 결과 |
| `data/korean_bb/metadata/train_weekly_us_tmylhs.csv` | 이번 실험 train index |
| `checkpoints/TransformerWithGaussian-M-v3_exp039_*.pt` | 이번 실험 체크포인트 |
| `docs/paper_final.md` | 결과 반영 대상 논문 |
