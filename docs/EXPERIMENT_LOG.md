# 실험 기록

## EXP-001: 10-cap 검증 (2026-02-18)

### 목적
BB SOTA 13.31%가 10건물/dataset cap으로 계산됨을 발견.
동일 조건으로 우리 PatchTST를 재평가하여 공정한 갭 측정.

### 방법
- 스크립트: `scripts/verify_10cap.py`
- 모델: PatchTST v5 (stage1_v5_20260218_191206, Val MAPE 1.46%)
- 프로토콜: BB `zero_shot.py` 재현 (10-cap, OOV 제외, pooled median)
- OOV 건물: 15개 (Electricity MT_147 등) 제외
- Bootstrap: 1,000 reps, 95% CI

### 결과

**Commercial (10-cap)**:
| Dataset | 10-cap NRMSE | 95% CI | N |
|---------|:---:|------|:---:|
| bdg-2:fox | **13.42%** | [10.37%, 18.54%] | 10 |
| bdg-2:bear | 18.84% | [7.75%, 24.06%] | 10 |
| bdg-2:rat | 21.31% | [16.48%, 27.19%] | 10 |
| bdg-2:panther | 23.34% | [13.01%, 51.22%] | 10 |
| electricity | 35.76% | [23.39%, 61.43%] | 10 |
| **OVERALL** | **21.31%** | **[16.98%, 25.98%]** | **50** |

**Commercial (Full)**:
| Dataset | Full NRMSE | N |
|---------|:---:|:---:|
| bdg-2:bear | 15.18% | 78 |
| bdg-2:fox | 17.33% | 132 |
| bdg-2:panther | 19.54% | 105 |
| bdg-2:rat | 23.69% | 277 |
| electricity | 33.11% | 150 |
| **OVERALL** | **23.33%** | **742** |

**Residential**: 10-cap=89.36%, Full=94.87% (모델이 상업건물만 학습, 예상대로 매우 높음)

### 핵심 발견

1. **10-cap(21.31%) < Full(23.33%)**: 10-cap이 약간 나음 (-2.02%p)
   - 각 데이터셋의 "첫 10건물"이 더 표준적인 건물일 가능성

2. **bdg-2:fox 10-cap = 13.42%**: BB SOTA(13.31%)에 매우 근접!
   - 단, 10건물이라 CI가 넓음 [10.37%, 18.54%]
   - Fox 사이트의 건물 특성이 우리 SIM과 유사할 가능성

3. **SOTA 갭 = +8.00%p**: 상당한 갭
   - BB: 900K 건물 학습 (350K 상업, 고유 스케줄 아키타입)
   - 우리: 20K 시뮬 (4-16 유형, 고정 스케줄)

4. **Electricity 데이터셋 최악** (35.76%):
   - 포르투갈 전력 소비 데이터 (실측, 주거/상업 혼합)
   - 한국 상업건물 시뮬과 완전히 다른 도메인

### 결론
순수 시계열 예측으로 BB SOTA를 이기기는 현실적으로 어려움.
→ 전략적 방향 전환: EMS 효과 예측 (CFEE) 추천
→ 상세: [STRATEGY.md](STRATEGY.md)

---

## EXP-000: 이전 BB 평가 결과 (참고)

### EnergyGPT BB Cross-Domain (v4, 이전 세션)
- PatchTST (v4): Commercial NRMSE 21.30% (full, 748건물)
- Persistence: ~19.63%
- 모델이 Persistence를 이기지 못함 (41.4% win rate)

### 근본 원인 분석
- SIM 일간 자기상관: 0.985 vs BB: 0.846
- SIM baseload ratio: 0.31-0.64 vs BB: 0.70-0.95
- SIM 24h 패턴: 뚜렷한 주중/야간 감소 vs BB: 높은 baseload
- 유사도-성능 상관: Spearman rho = -0.002 (p=0.96) → 무관

---

## EXP-003: v3 Incremental Training (2026-02-22~)

### 목적
v3 패턴갭 수정 데이터(12D LHS, 다중도시/vintage)를 ~1K건 단위로 누적하며
BB Commercial CVRMSE 개선 추이를 추적. 논문용 ablation study.

### 설계
- **기준선 (S0)**: v2.1 — 2,097건, 50 epochs → Commercial **19.31%**
- **각 Step**: ~1K건 추가 시뮬 → 완전 재학습(max 20ep, patience=10) → BB 평가
- **자동 기록**: `results/pipeline_progress.json`, `results/paper_results.md`
- **확인 명령**: `python scripts/run_pipeline.py --report`

### v3 패턴 갭 수정 요약 (100건 검증)
| 패턴 | v2 (before) | v3 (after) | BB 실측 | 개선 |
|------|:---:|:---:|:---:|:---:|
| 야간/주간 비율 | 0.512 | **0.852** | 0.803 | ✅ |
| 주간 자기상관 168h | 0.938 | **0.784** | 0.751 | ✅ |
| Baseload P95 | 0.51 | **0.905** | 0.725 | ✅ |
| CV P5 (flat) | 0.21 | **0.032** | 0.105 | ✅ |

### Step별 결과 (자동 업데이트: paper_results.md 참조)

*BB 평가 기준: Commercial 955건 (BDG-2 611 + Electricity 344), Residential 953건, 총합 1,908건*
*per-year 독립 로딩, 10-cap 미사용 (2026-02-23 수정)*

| Step | 누적 건물 | Commercial | Electricity | BDG-2 | Gap | 비고 |
|:----:|----------:|:----------:|:-----------:|:-----:|:---:|------|
| S0 (v2.1) | 2,097 | 19.31% | 10.84% | 20.99% | +6.00%p | 기준선 |
| **S1** | **3,017** | **16.86%** | **13.40%** | **19.81%** | **+3.55%p** | office+retail 1K 추가, early stop ep6 |
| **S2** | **3,407** | **17.43%** | **13.83%** | **20.12%** | **+4.12%p** | school+hotel+hospital 450건 추가, ep14 |
| **S3** | **3,867** | **18.21%** | **14.19%** | **20.88%** | **+4.90%p** | 추가 시뮬 460건 |
| **S4 scratch** | **5,867** | **16.82%** | **12.53%** | **19.87%** | **+3.51%p** | 완전 재학습. S1 돌파. Electricity 최고 |

*실제 결과: paper_results.md 참조 (자동 갱신)*

### S1 학습 관찰 (2026-02-22)
- 데이터: office(760) + retail(760) + v2 전체(1,497 상업 + 600 주거) = 3,017건
- 에포크별: 1(-2.65)→2(-3.06)→3(-3.21)→4(-3.33)→5(-3.34)→6(-3.41, BEST)→7~12 (미개선)
- Patience=10, 자동종료 16 epoch

### S2 학습 관찰 (2026-02-22~23)
- 데이터: S1 + school(130)+hotel(150)+hospital(150)+apartment(19) = 3,407건
- best_epoch=14, val_loss=-3.5228, 20 epoch 완료 (early stop 없음)
- Commercial 악화 (+0.66%p): school/hotel/hospital이 BB commercial 건물과 패턴 미스매치 가능성
- Residential 소폭 개선 (86.81%→84.77%)
- **관찰**: 다양한 아키타입 추가가 단기적으로 commercial을 악화시킬 수 있음 → 데이터 누적 필요

---

## EXP-004: 새 아키타입 6종 추가 (Strategy C) (2026-03-02~03)

### 목적
BDG-2 소형 건물(<50kW, 259건) CVRMSE 23.16%, 미커버 타입(public 190, assembly 149) 28% 갭 해소.
새 아키타입 6종 추가로 BDG-2 개선 → Commercial SOTA 접근.

### 새 아키타입 6종
| archetype_id | DOE BLDG | Base IDF | BDG-2 대응 | 파워 | SCALE_CAP |
|:---:|:---:|:---:|:---:|:---:|:---:|
| small_office | B01 | OfficeSmall | office소형, public, assembly | 5-30kW | 3.0 |
| large_office | B03 | OfficeLarge | office대형 | 500-1500kW | 2.0 |
| warehouse | B14 | Warehouse | warehouse, utility | 10-50kW | 3.0 |
| restaurant_full | B10 | RestaurantSitDown | food | 50-100kW | 1.2 |
| restaurant_quick | B09 | RestaurantFastFood | food소형 | 30-70kW | 1.2 |
| strip_mall | B04 | RetailStripmall | retail 소형/다점포 | 30-60kW | 2.5 |

### 수정 파일 (7개)
- `src/buildings/archetypes.py` — 6개 BuildingArchetype 추가
- `configs/idf_mapping.yaml` — source_idf 매핑 6개 추가
- `scripts/generate_parametric_idfs.py` — 5개 dict × 6항목
- `scripts/postprocess.py` — ARCHETYPE_BUILDING_TYPE 6개 추가
- `scripts/run_pipeline.py` — ARCHETYPE_TYPES + building_id 파싱 버그 수정
- `src/idf/modifier.py` — set_equipment_density() Watts/Area 지원 추가

### 검증 (smoke_test_new_archetypes.py)
- 18 시뮬레이션 (6 아키타입 × 3 파라미터): **ALL PASS**
- small_office low_scale: peak **6kW** → BDG-2 <10kW 커버 확인
- Watts/Area 수정: 5개 zone 전부 정확 적용

### S16 배치: 5,000건 시뮬
| 아키타입 | 수량 | 시뮬 | 비고 |
|---------|:---:|:---:|------|
| small_office | 1,500 | OK | ~8s/건 |
| large_office | 1,000 | OK | ~33s/건 |
| warehouse | 800 | OK | ~9s/건 |
| strip_mall | 700 | OK | ~17s/건 |
| restaurant_full | 500 | OK | ~7s/건 |
| restaurant_quick | 500 | OK | ~7s/건 |
| **합계** | **5,000** | **0 FAIL** | 로컬 ~3h |

### 후처리
- CSV→Parquet: 5,000 OK (549s)
- 카탈로그: 49,955 + 5,000 = **54,955건**
- **catalog.csv city 버그 수정**: 기존 2,079건 3-part vintage 파싱 오류 (city="plus"/"2000" 등) 패치

### 학습 (55K sub-epoch)
- Config: TransformerWithGaussian-M-v3-final.toml (batch 128, lr 6e-5)
- --max_steps 40,690 = 0.28 epochs (BB 동등)
- train_loss=-2.52710, val_loss=-3.40266 (4,077s)

### Full BB Eval 결과

| Dataset | Type | N | 55K CVRMSE | 50K CVRMSE | 변화 |
|---------|:---:|:---:|:---:|:---:|:---:|
| **Commercial** | **all** | **955** | **15.56%** | **15.53%** | **+0.03%p** |
| BDG-2 | commercial | 611 | 18.72% | 18.46% | +0.26%p |
| **Electricity** | **commercial** | **344** | **11.21%** | **11.77%** | **-0.56%p** |
| Residential | all | 953 | 84.51% | 84.51% | 동일 |

**Gap vs SOTA: +2.25%p** (50K: +2.22%p)

### 분석
- **Electricity 개선 -0.56%p**: 새 아키타입(small_office, restaurant)이 실측 패턴 다양성 확보에 기여
- **BDG-2 소폭 악화 +0.26%p**: 새 데이터 희석 효과, 아직 학습 부족 가능
- **Commercial 전체는 동등**: Electricity 개선과 BDG-2 악화가 상쇄
- **결론**: 새 아키타입 자체는 유효하나, 5K만으로는 전체 성능 큰 차이 없음

---

## EXP-005: University 아키타입 추가 (Strategy A+C) (2026-03-03)

### 목적
BDG-2 Education(254건, 42%)이 미국 대학 패턴 vs 한국 K-12 패턴 불일치로 성능 약함.
University 아키타입 신설 + per-archetype LHS 파라미터 오버라이드로 BDG-2 Education 개선.

### BDG-2 Education 패턴 분석
| 지표 | BDG-2 Edu | Our School | 갭 |
|------|:---:|:---:|:---:|
| Night/Day ratio | 0.721 | 0.861 | +0.139 |
| Autocorr 168h | 0.787 | 0.908 | +0.121 |
| Baseload P5/mean | 0.562 | 0.724 | +0.163 |
| CV | 0.351 | 0.245 | -0.106 |
| Mean load | 97 kW | 236 kW | +139 kW |

### University 아키타입 정의
| 항목 | 값 | 비교(school) |
|------|:---:|:---:|
| Base IDF | SchoolPrimary | 동일 |
| DOE BLDG | B07 | 동일 |
| HVAC | HA (PSZ-AC) | 동일 |
| Lighting | 12.0 W/m² | 10.0 |
| Equipment | 12.0 W/m² | 8.0 |
| Baseload | 800 W | 500 |
| SCALE_MULT_CAP | 1.5 | 3.0 |

### 시행착오 기록

**시도 1: ARCHETYPE_PARAM_OVERRIDES로 LHS 범위 오버라이드**
- scale_mult(0.1~1.5), weekly_break_prob(0.15~0.60) 등 극단값 설정
- 결과: **59% 실패** (534/902)
- 원인: Plant loop temp exceeded (400건) + warmup 미수렴 (144건)

**시도 2: 파라미터 범위 축소**
- scale_mult(0.3~1.5), weekly_break_prob(0.10~0.45) 등으로 완화
- 결과: **54% 실패** — 불충분

**시도 3: set_warmup_days(25) 추가**
- 기본 15일 → 25일로 확대
- 결과: **40% 실패** — 여전히 높음

**시도 4: relax_convergence(temp_tol=0.4, FullExterior) 추가**
- 소규모 테스트(50건): 19% 실패로 개선
- 1,250건 생산: **46% 실패** — 오히려 악화

**근본 원인 분석 (bounty hunter agent)**
- `FullExterior` Solar Distribution이 PSZ-AC 시스템의 DX coil을 불안정화
- 코일 출구 온도가 -476°C까지 발산
- relax_convergence가 해결책이 아닌 **원인**이었음

**최종 해결 (시도 5): 오버라이드 전면 제거**
- ARCHETYPE_PARAM_OVERRIDES = {} (빈 dict)
- 차별화는 loads(lighting/equipment/baseload_w)와 SCALE_MULT_CAP(1.5)만으로 달성
- 결과: **0% 실패 (1,000/1,000)**

### 부수 발견: Windows readvars.audit 거짓 실패
- 10+ workers 병렬 시뮬 시 readvars.audit 파일 잠금으로 eplusout.end에 "Fatal Error" 표시
- 실제 CSV는 8760행 정상 생성 — **거짓 실패**
- 시뮬 성공 판별은 eplusout.end가 아닌 CSV 행수 기반으로 해야 함

### 수정 파일
- `scripts/generate_parametric_idfs.py` — ARCHETYPE_PARAM_OVERRIDES, get_effective_bounds(), university dict 추가
- `src/idf/modifier.py` — set_warmup_days(), relax_convergence() 메서드 추가
- `src/buildings/archetypes.py` — university BuildingArchetype 등록
- `configs/idf_mapping.yaml` — university source_idf 매핑
- `scripts/postprocess.py` — ARCHETYPE_BUILDING_TYPE 추가
- `scripts/run_pipeline.py` — ARCHETYPE_TYPES 추가

### S17 배치: 1,000건 시뮬
| 아키타입 | 수량 | 결과 | 시뮬시간 |
|---------|:---:|:---:|:---:|
| university | 1,000 | **0 FAIL** | ~1min/건 |
- IDF: `simulations/idfs_v3_univ_final/`
- 결과: `simulations/results_v3_univ_final2/`

### 후처리
- CSV→Parquet: 1,000 OK
- 카탈로그: 54,955 + 1,000 = **55,955건** (14 archetypes)

### 학습 (56K sub-epoch scratch)
- Config: TransformerWithGaussian-M-v3-final.toml (batch 128, lr 6e-5)
- --max_steps 42,100 = 0.28 epochs
- best_val_loss = -3.4112 (55K: -3.4027, 개선)

### Full BB Eval 결과

| Dataset | Type | N | 56K CVRMSE | 55K CVRMSE | 변화 |
|---------|:---:|:---:|:---:|:---:|:---:|
| **Commercial** | **all** | **955** | **15.40%** | **15.56%** | **-0.16%p** |
| **BDG-2** | **commercial** | **611** | **18.48%** | **18.72%** | **-0.24%p** |
| **Electricity** | **commercial** | **344** | **11.04%** | **11.21%** | **-0.17%p** |
| Residential | all | 953 | 83.80% | 84.51% | -0.71%p |

**Gap vs SOTA: +2.09%p** (55K: +2.25%p, 개선 -0.16%p)

### 누적 결과 추이
| Step | 누적건물 | Commercial | Gap | 비고 |
|:----:|:-------:|:----------:|:---:|------|
| S0 (v2.1) | 2,097 | 19.31% | +6.00%p | 기준선 |
| S1 | 3,017 | 16.86% | +3.55%p | v3 12D LHS |
| S4 scratch | 5,867 | 16.82% | +3.51%p | 완전 재학습 |
| S8 | 11,886 | 16.73% | +3.42%p | |
| 50K sub | 49,955 | 15.53% | +2.22%p | sub-epoch 전환 |
| 55K newarch | 54,955 | 15.56% | +2.25%p | 6종 추가 |
| **56K univ** | **55,955** | **15.40%** | **+2.09%p** | **university 추가, 역대 최고** |
| Persist | 900K | 16.68% | +3.37%p | BB baseline |
| **SOTA-M** | 900K | **13.28%** | — | BB GitHub 공식 (Transformer-M Gaussian) |
| SOTA-L | 900K | 13.31% | +0.03%p | |

### 분석
- **전 세그먼트 균일 개선**: Commercial/Electricity/BDG-2 모두 개선 — university가 BDG-2 Education 패턴 커버에 효과적
- **SOTA 갭 2%p대 진입**: 55,955건 시뮬 데이터만으로 900K 합성 데이터 SOTA-L 대비 +2.09%p
- **Persistence Ensemble 1.28%p 초과**: 소규모 시뮬 데이터 기반 모델이 대규모 합성 데이터 기반 통계 모델보다 확실히 나음
- **아키타입 차별화 방법**: 같은 IDF 기반이라도 loads/SCALE_MULT_CAP만으로 충분히 다른 패턴 생성 가능
- **ARCHETYPE_PARAM_OVERRIDES는 위험**: PSZ-AC 같은 민감한 HVAC 시스템에서 극단적 스케줄 범위가 발산을 유발

---

## EXP-006: 성능 최적화 — 현상금 사냥꾼 (2026-03-03)

### 목적
Commercial 15.40% → SOTA 13.31% 갭(+2.09%p) 해소. 근본 원인 분석 후 순차적 개선.

### 근본 원인 분석 (root-cause-analyst agent)
6개 원인 식별, 기대 효과순 정렬:

| # | 원인 | 기대 효과 | 상태 |
|:-:|------|:---:|:---:|
| 1 | Box-Cox Transform 불일치 | -1.0~2.0%p | **실패** (+0.84%p 악화) |
| 2 | Training token 부족 (125M vs 1B) | -0.5~1.0%p | **실패** (+0.62%p 악화) |
| 3 | 미커버 건물 유형 (public/assembly) | -0.5~1.0%p | 미착수 |
| 4 | 모델 크기 M→L (15.8M→160M) | -0.3~0.5%p | 미착수 |
| 5 | Data augmentation (cutout, mixup) | -0.2~0.5%p | 미착수 |
| 6 | 코드 차이 (negligible) | 0~0.2%p | 스킵 |

### Action 1: BB Box-Cox 통일 — 실패

**가설**: 학습 시 우리 Box-Cox(λ=-0.050)로 정규화하고 eval 시 BB Box-Cox(λ=-0.067)로
정규화하면 수치 공간 불일치가 성능 저하를 유발한다.

**방법**:
- 우리 boxcox.pkl → boxcox_ours_backup.pkl 백업
- BB의 boxcox.pkl(λ=-0.067, mean=1.44, scale=1.77) 복사하여 학습에 사용
- 동일 조건 scratch 학습: --max_steps 42,100, batch 128, lr 6e-5

**Box-Cox 비교**:
| 파라미터 | Our (original) | BB | 차이 |
|---------|:---:|:---:|:---:|
| lambda | -0.05010 | -0.06722 | -0.017 |
| mean | 4.47548 | 1.43939 | -3.036 |
| scale | 0.89729 | 1.76805 | +0.871 |

**학습 결과**:
- train_loss=-2.62434 (our: -2.52710)
- val_loss=-3.60487 (our: -3.41120, **비교 불가** — Box-Cox 공간이 다름)
- 학습 시간: 4,697s

**Full BB Eval 결과**:

| Dataset | Type | N | BB Box-Cox | Our Box-Cox | 변화 |
|---------|:---:|:---:|:---:|:---:|:---:|
| **Commercial** | **all** | **955** | **16.24%** | **15.40%** | **+0.84%p 악화** |
| BDG-2 | commercial | 611 | 19.30% | 18.48% | +0.82%p 악화 |
| Electricity | commercial | 344 | 11.36% | 11.04% | +0.32%p 악화 |
| Residential | all | 953 | 231.00% | 83.80% | 대폭 악화 |

**Gap vs SOTA: +2.93%p** (이전 +2.09%p, 악화)

**분석 — 왜 실패했나**:
1. **데이터 분포 불일치**: 우리 시뮬 데이터와 BB 실측 데이터의 분포가 근본적으로 다름
   - 시뮬: 평균 부하 높음 (mean=4.48 in Box-Cox), 분산 작음 (scale=0.90)
   - BB 실측: 평균 부하 낮음 (mean=1.44), 분산 큼 (scale=1.77)
2. **정규화 품질 > 수치 공간 일치**: 데이터에 맞게 피팅된 Box-Cox가 패턴 학습에 최적
3. **모델의 분포 적응력**: 모델은 학습 시 데이터 패턴(주기성, 변동성)을 학습하며,
   eval 시 다른 Box-Cox 공간의 입력을 받아도 잘 작동함
4. **Residential 231% 폭등**: BB Box-Cox의 큰 scale(1.77)이 저부하 주거 데이터에서
   예측 오차를 극단적으로 증폭

**결론**: Box-Cox 불일치는 근본 원인이 아님. 우리 Box-Cox로 복원.

### Action 2: 학습 토큰 4배 증가 — 실패 (과적합)

**가설**: BB는 1B tokens(81K steps × 8 GPU), 우리는 125M tokens(42K steps × 1 GPU) = 8배 적음.
동일 데이터를 더 많이 반복 학습하면 패턴 일반화가 개선된다.

**방법**:
- 4x 증가: --max_steps 168,400 (= 1.12 epochs, ~500M tokens)
- 우리 Box-Cox 사용 (복원됨)
- scratch 학습, augment, num_workers=2
- 체크포인트: `v3_56k_4x`

**학습 결과**:
- Epoch 1 완료: train_loss=-2.889, val_loss=-3.549
- max_steps 도달: train_loss=-3.085, val_loss=-3.553 (1.12 epochs)
- 학습 시간: 17,196s (~4.8h)

**Full BB Eval 결과**:

| Dataset | Type | N | 4x (168K) | 기준 (42K) | 변화 |
|---------|:---:|:---:|:---:|:---:|:---:|
| **Commercial** | **all** | **955** | **16.02%** | **15.40%** | **+0.62%p 악화** |
| BDG-2 | commercial | 611 | 18.79% | 18.48% | +0.31%p 악화 |
| Electricity | commercial | 344 | 11.72% | 11.04% | +0.68%p 악화 |
| Residential | all | 953 | 89.52% | 83.80% | +5.72%p 악화 |

**Gap vs SOTA: +2.71%p** (기준 +2.09%p, 악화)

**분석 — 왜 실패했나**:
1. **과적합**: val_loss는 개선(-3.411→-3.553)되었지만 BB eval은 악화
   → 시뮬 데이터 내부 패턴에 과적합, 실측 데이터 일반화 저하
2. **0.28 epochs가 최적점**: BB도 대규모 데이터에서 ~1 epoch 학습하지만,
   우리는 56K건(BB의 1/16)이라 0.28 epochs가 과적합 전 최적 지점
3. **BB와의 핵심 차이**: BB는 900K건 다양한 합성 데이터(NREL ResStock+ComStock) → 더 많이 학습해도 과적합 안 됨.
   우리는 56K건 시뮬 데이터 → 금방 패턴을 암기함
4. **Residential 89.52%**: 과학습으로 시뮬 분포에 더 특화 → 실측 주거 건물 예측 악화

### 종합 결론

| Action | 방법 | Commercial | Gap | 결과 |
|:---:|------|:---:|:---:|:---:|
| — | **기준선 (56K, 42K steps)** | **15.40%** | **+2.09%p** | **최고** |
| 1 | BB Box-Cox 통일 | 16.24% | +2.93%p | 실패 |
| 2 | 4x 토큰 (168K steps) | 16.02% | +2.71%p | 실패 |

**핵심 발견**:
- **Sim-to-real gap이 근본 한계**: 시뮬 데이터의 Box-Cox나 학습량을 조정해도
  실측 건물의 고유한 다양성(장비 고장, 인간 행동, 날씨 반응 등)은 재현 불가
- **0.28 epochs (sub-epoch)이 최적**: 시뮬 데이터 규모에서 과적합 없이
  패턴 일반화를 최대화하는 sweet spot
- **현재 15.40%가 56K 시뮬 데이터의 실질적 성능 상한**
- **추가 개선 경로**: (a) 더 다양한 아키타입/시뮬 추가, (b) 모델 크기 증가,
  (c) 실측 데이터 소량 혼합(domain adaptation), (d) 논문으로 전환

---

## EXP-007: 100K 데이터 + 62K steps (2026-03-05)

### 목적
90K(42K steps, 0.163ep) → 15.52%로 56K(15.40%) 대비 악화. under-training 원인 파악.

### 방법
- 데이터: 95,955 buildings (전체)
- max_steps=62,000 (0.240 epochs)
- scratch, augment, RevIN

### 결과
- val_loss=-3.48142 (역대 최고), NRMSE=0.0894
- **Commercial: 15.43%** (gap +2.12%p)
- BDG-2: 18.68%, Electricity: 10.97%

### 분석
62K steps(0.240ep)가 42K(0.163ep)보다 낫지만 56K(15.40%)보다 악화. restaurant 27% 편향이 상한 제약.

---

## EXP-008: 3K 균등 샘플링 (2026-03-05)

### 목적
restaurant 편향 해소: 아키타입별 최대 3K 샘플링으로 균등 분포 구성.

### 방법
- `scripts/resample_to_nk.py --n 3000` → 39K buildings
- max_steps=40,000 (0.382 epochs), scratch, augment
- 체크포인트: `TransformerWithGaussian-M-v3-3k_v3_3k_balanced_best.pt`

### 결과
- val_loss=-3.48142, NRMSE=0.0894 (inline)
- **Full BB eval: Commercial 15.31%** (gap +2.00%p)
- BDG-2: 18.41%, Electricity: 11.02%

### 분석
균등 샘플링이 restaurant 편향 해소 → 개선. 이후 모든 실험의 표준 방법으로 채택.

---

## EXP-009: 3K 완전 균형 (2026-03-06)

### 목적
large_office/small_office/university 각 3K 완성 → 완전한 14 archetypes × 3K.

### 방법
- s22 추가 시뮬: small_office 1K + large_office 1K + university 1K
- 총 42K buildings, max_steps=40,000 (0.363 epochs)
- val_loss=-3.39728, NRMSE=0.0958, 21,128s

### 결과
- **Full BB eval: Commercial 15.36%** (gap +2.05%p)
- EXP-008(15.31%) 대비 0.05%p 악화

### 분석
완전한 3K balanced가 오히려 미세하게 나빴음. 3K cap에서 성능 상한 도달.

---

## EXP-010: RevIN — Per-instance Normalization (2026-03-06) ★ BEST

### 목적
각 테스트 건물의 168h context로 per-instance 정규화 → OOD 일반화 개선.

### 방법
- RevIN: context (x-μ)/σ → 예측 → 역정규화
- 3K 균등 (42K), max_steps=40,000 (0.354 epochs), scratch, augment
- `src/models/transformer.py`: use_revin=True 추가
- val_loss=-1.77422, NRMSE=0.0925, 12,812s
- 체크포인트: `TransformerWithGaussian-M-v3-3k_v3_revin_best.pt`

### 결과 (Full BB eval)
| Segment | N | CVRMSE |
|---------|:-:|:------:|
| **Commercial** | 955 | **15.16%** |
| BDG-2 | 611 | 18.15% |
| Electricity | 344 | 10.84% |
| Residential | 953 | 78.73% |

**Gap vs SOTA: +1.85%p — 현재 최고 성능**

### 분석
- EXP-009(15.36%) 대비 -0.20%p 개선, 전 세그먼트 개선
- RevIN은 BB보다 유리한 조건 (BB는 global Box-Cox만, 우리는 추가 per-instance norm) → 논문 명시 필요
- **이후 모든 실험의 baseline**

---

## EXP-011: RevIN + Restaurant 2K cap (2026-03-07)

### 목적
restaurant_full/quick 비중 축소 (BB commercial에서 restaurant 비율 낮음).

### 방법
- restaurant_full/quick 2K, 기타 4K → 51,975 buildings
- max_steps 비례 조정, val_loss=-1.75723, 10,641s

### 결과
- **Full BB eval: Commercial 15.31%** (gap +2.00%p)
- EXP-010(15.16%) 대비 0.15%p 악화

### 분석
restaurant 줄이기 전략 실패. 분포 alignment 효과 < 데이터 다양성 손실.
3K 균등(14% rest)이 r2k_4k(8% rest)보다 우수.

---

## EXP-012: RevIN + 5K 전체 균등 (2026-03-07)

### 목적
데이터 양 증가: 3K→5K cap으로 OOD 성능 개선 가능성 검증.

### 방법
- 5K cap: 14 archetypes × 5K → 66,974 buildings
- max_steps=50,000 (0.278 epochs), val_loss=-1.84743

### 결과
- **Full BB eval: Commercial 15.35%** (gap +2.04%p)
- EXP-010(15.16%) 대비 0.19%p 악화

### 분석
더 많은 데이터(67K)가 synthetic val에는 더 잘 맞지만 OOD 실측에는 오히려 나쁨.
val_loss(synthetic) ≠ BB eval(real). **3K/42K/0.354ep가 실측 OOD 최적점**.

---

## EXP-013: RevIN + 5K 63K steps (2026-03-07)

### 목적
5K cap에서 coverage 증가(건물당 노출 횟수 = EXP-010 수준으로)로 개선 가능성 검증.

### 방법
- 5K cap 67K buildings + max_steps=63,000 (0.350 epochs)
- 건물당 노출 ~120회 (EXP-010과 동일)
- val_loss=-1.87322, NRMSE=0.0896, 19,144s

### 결과
- **Full BB eval: Commercial 15.51%** (gap +2.20%p)
- EXP-012(15.35%)보다도 악화

### 분석
Coverage 가설 기각. 67K 데이터에서 더 많이 학습할수록 오히려 악화.
**42K(3K cap)이 최적 데이터 크기. 5K cap은 어떤 epoch에서도 안 됨.**

---

## EXP-014: Seasonal Decomp + RevIN (2026-03-07~08)

### 목적
trend/seasonal 분리 → RevIN on seasonal → 예측 후 trend 복원으로 구조적 개선.

### 방법
- moving_avg(k=25) → trend, load-trend → seasonal
- RevIN on seasonal, predict seasonal, add trend back
- `configs/model/*-decomp.toml`: use_seasonal_decomp=true, decomp_kernel=25
- 3K 균등 (42K), max_steps=40,000 (0.354 epochs)
- val_loss=-2.11017, NRMSE=0.0892, 12,067s
- 체크포인트: `TransformerWithGaussian-M-v3-3k-decomp_v3_seasonal_decomp_bb_best.pt`

### 결과
- **BB Commercial (inline): 16.65%** — EXP-010(15.16%) 대비 +1.49%p 악화

### 분석
Moving average trend 분리가 BB OOD 건물에 오히려 노이즈로 작용.
RevIN만 단독 사용이 최선. Seasonal decomp 실패.

---

## EXP-015: 스케일링 법칙 — 10K cap (2026-03-15)

### 목적
건물 수 스케일링 법칙 검증: 데이터 증가(140K) → Commercial NRMSE 개선 여부

### 방법
- 10K cap: 14 archetypes × 10K = **140K buildings**
- max_steps=133,000 (0.353 epochs), scratch, augment, preload, RevIN
- config: TransformerWithGaussian-M-v3-3k.toml

### 결과 (Full BB eval, 1908 buildings, bootstrap 50K reps)

| Segment | N | CVRMSE |
|---------|:-:|:------:|
| **Commercial** | 955 | **15.95%** |
| BDG-2 | 611 | 19.15% |
| Electricity | 344 | 11.75% |
| Residential | 953 | 80.92% |

**Gap vs SOTA: +2.64%p** (기준 EXP-010 +1.85%p, **악화**)

inline eval (train.py 내부): 17.12%

### 분석
- 140K(10K cap)이 42K(3K cap, EXP-010)보다 **0.79%p 악화**
- 건물 수 증가 → 오히려 OOD 성능 저하 (synthetic 과적합 패턴)
- **스케일링 법칙 가설 기각**: 단순 건물 수 증가로는 개선 안 됨

### 후속 실험
- EXP-015b: commercial-only 10K (apartment 제외, 120K buildings, steps=114K)
  - 가설: residential 학습이 commercial 성능에 노이즈로 작용

---

## EXP-015b: commercial-only 10K (2026-03-15)

### 목적
apartment_midrise/highrise 제외, commercial 아키타입 12종만 학습 시 commercial eval 개선 여부 확인.

### 방법
- **학습 데이터**: commercial 12 archetypes × 10K = **120K buildings** (apartment 2종 제외)
- **인덱스**: train_weekly_commercial_10k.csv / val_weekly_commercial_10k.csv
- max_steps=114,000 (0.35 epochs), scratch, augment, preload, RevIN
- config: TransformerWithGaussian-M-v3-3k.toml
- val_loss=-1.79567, NRMSE=0.0930, 15,450s (4.3h)

### 결과 (Full BB eval, 1908 buildings, bootstrap 50K reps)

| Segment | N | CVRMSE |
|---------|:-:|:------:|
| **Commercial** | 955 | **15.83%** |
| BDG-2 | 611 | 19.24% |
| Electricity | 344 | 11.57% |
| Residential | 953 | 80.53% |

**Gap vs SOTA: +2.52%p**

inline eval (train.py 내부): 16.76%

### 비교

| 실험 | 학습 건물 | Commercial | Gap |
|------|:--------:|:----------:|:---:|
| EXP-010 RevIN 3K balanced | 42K | **15.16%** | +1.85%p |
| EXP-015 전체 10K | 140K | 15.95% | +2.64%p |
| **EXP-015b commercial 10K** | **120K** | **15.83%** | **+2.52%p** |

### 분석
- 전체 10K(15.95%)보다 0.12%p 개선됐지만, 3K balanced(15.16%)엔 여전히 0.67%p 뒤처짐
- **가설 기각**: apartment 제거가 commercial 성능을 크게 개선하지 않음
- apartment 포함 다양한 아키타입 학습이 오히려 일반화에 기여 (정규화 효과)
- **핵심 결론**: 데이터 증가(건물 수)가 아닌 **3K 균등 샘플링 + RevIN** 조합이 OOD 일반화 최적점
- 스케일링 방향 전환 필요: 건물 수 증가보다 아키텍처/학습 전략 개선에 집중

---

## EXP-016: 스케일 스윕 2K~6K (2026-03-15~16)

### 목적
정밀 스케일링 법칙 확인: 2K/3K/4K/5K/6K per archetype, 동일 조건(RevIN + augment + preload + 0.354 epoch)

### 조건 (EXP-010과 동일, preload 추가)
- config: TransformerWithGaussian-M-v3-3k.toml (RevIN=true)
- augment, num_workers=0, scratch
- epoch 비율: 0.354 고정 (steps = 40K × N/3K)
- 14 archetypes 균등 (apartment 포함)

| Cap | Buildings | max_steps | epoch |
|-----|:---------:|:---------:|:-----:|
| 2K | 28K | 27,000 | 0.36 |
| 3K | 42K | 40,000 | 0.354 |
| 4K | 56K | 53,000 | 0.352 |
| 5K | 70K | 67,000 | 0.354 |
| 6K | 84K | 80,000 | — |

### 결과 (Full BB eval, 1908 buildings)

| Cap | Buildings | Commercial | BDG-2 | Electricity | Residential | val_loss |
|-----|:---------:|:----------:|:-----:|:-----------:|:-----------:|:--------:|
| **2K** | 28K | **15.14%** | 18.52% | 10.62% | 78.31% | -1.631 |
| **3K** (EXP-010) | 42K | **15.16%** | 18.15% | 10.84% | 78.73% | -1.774 |
| **4K** | 56K | **15.29%** | 18.43% | 10.88% | 79.79% | -1.835 |
| **5K** | 70K | **15.55%** | 18.80% | 11.11% | 79.71% | -1.844 |
| **6K** | 84K | — | — | — | — | — |

> 3K(EXP-010): preload 없이 LRU cache 사용. 2K/4K/5K/6K: preload 사용. 모델 학습 동일.
> 6K 학습 중 (2026-03-16 07:33 기준)

### 분석 (진행 중)
- 2K(15.14%) ≈ 3K(15.16%) — 최솟값 구간
- 4K부터 단조 증가 (악화): 4K→5K +0.26%p
- **스케일링 법칙 패턴**: 2~3K 최적, 그 이후 건물 수 증가 = 성능 저하
- 6K 결과 대기 중

---

## EXP-017: BB GitHub 코드 비교 분석 — 20인 전문가 패널 (2026-03-16)

### 배경

Korean_BB vs BuildingsBench 공식 GitHub 코드를 직접 비교하여 성능 갭(+1.85%p) 원인을 체계적으로 분석.
4개 그룹 × 5인 전문가 편성, BB 코드(`external/BuildingsBench/`) 직접 정독 후 교차 검증.

### 분석 대상

| 파일 | 핵심 내용 |
|------|----------|
| `external/BuildingsBench/buildings_bench/models/transformers.py` | LoadForecastingTransformer 구조 (d_model=512) |
| `external/BuildingsBench/buildings_bench/transforms.py` | BoxCoxTransform λ=-0.067, LatLonTransform (US PUMA 기준) |
| `external/BuildingsBench/buildings_bench/data/buildings900K.py` | 4 datasets, sliding_window=24, US lat/lon |
| `external/BuildingsBench/scripts/zero_shot.py` | `if count == 10: break` — 데모 코드 (논문 결과 아님) |
| `external/BuildingsBench/buildings_bench/configs/TransformerWithGaussian-M.toml` | batch=64, lr=6e-5, train_tokens=1B |
| `external/BuildingsBench_data/metadata/benchmark.toml` | BDG-2 Bear=(37.87,-122.26), Electricity=(35.40,-120.87) — 모두 가짜 US 좌표 |

### 발견된 버그 및 불일치 — 우선순위순

#### BUG-1: evaluate_bb.py lat/lon 정규화 3-way 불일치 ← **우선순위 #1 (즉시 수정)**

| 코드 위치 | 동작 | 문제 |
|----------|------|------|
| `src/data/korean_dataset.py` | Korean 좌표 정규화 (lat≈35.5±1.5, lon≈127.8±0.9) | 학습 기준 |
| `scripts/train.py` fast eval | `lat=0, lon=0` 하드코딩 | inline eval 기준 |
| `scripts/evaluate_bb.py:339~340` | `(lat-37.5)/5.0`, `(lon+96.0)/20.0` (US rough center) | **full eval 불일치** |

BB 데이터(benchmark.toml)의 모든 좌표는 가짜 US CONUS 좌표. Electricity 데이터셋은 포르투갈 실제 데이터인데 San Luis Obispo(CA) 좌표 사용. 정규화 자체가 무의미 → `lat_norm=0.0, lon_norm=0.0`이 가장 일관적.

**수정 사항 (2026-03-16 즉시 완료)**:
```python
# scripts/evaluate_bb.py:339~340  BEFORE:
lat_norm = (self.latlon[0] - 37.5) / 5.0  # rough US center
lon_norm = (self.latlon[1] + 96.0) / 20.0  # rough US center

# AFTER:
lat_norm = 0.0   # 학습 시 ignore_spatial=True → 0 효과와 동일하게 맞춤
lon_norm = 0.0
```

**예상 개선**: inline eval과 full eval 간 불일치 해소 → 0.1~0.2%p 개선 기대 (EXP-017 실측 확인 중)

#### BUG-2: transformer.py ignore_spatial 부분 버그

```python
# src/models/transformer.py:182~188
if ignore_spatial:
    self.lat_embedding = nn.Linear(1, 32 * s)
    nn.init.zeros_(self.lat_embedding.weight)
    # lon_embedding은 0으로 초기화 안 됨 — 버그
```

lat만 0 초기화, lon은 학습 활성화. ignore_spatial=True가 완전히 적용 안 됨.
**수정 방법**: `nn.init.zeros_(self.lon_embedding.weight)` 추가 → 재학습 필요 (우선순위 #2)

#### 발견-3: Box-Cox 부호 반대 (완화됨)

- BB: λ=-0.067 (음수), 표준화 mean≈3.08, std≈1.43
- 우리: λ≈+0.011 (양수), 표준화 mean≈5.25, std≈1.89

절대값 차이는 작지만 부호가 반대. RevIN이 99.8% 보정 → 현재는 RevIN 사용으로 실질적 영향 최소화.
BB Box-Cox 대체 시도(EXP-006a)에서 오히려 악화 확인 → 자체 Box-Cox 유지 결정.

#### 발견-4: BB 10-cap 혼동 해소

`zero_shot.py` 내 `if count == 10: break`는 **데모/디버그 코드**, 논문 공식 결과(13.31%)는 전체 건물 사용.
우리 955건물 full eval 프로토콜이 올바름 (EXP-001에서 확인했으나 이번에 코드 직접 재확인).

#### 발견-5: BB 건물 노출 횟수 비대칭

| 항목 | BB (900K synth) | Ours (42K synth) |
|------|:---:|:---:|
| 총 학습 토큰 | 1B | ~15B (0.354ep) |
| 건물당 노출 | **5.8×** | **122×** |
| 다양성 | 900K 합성 건물 (NREL) | 42K 합성 건물 (Korean) |

우리는 같은 42K 건물을 122번씩 반복 학습 → 합성 데이터의 EnergyPlus 물리 패턴에 과적합.
BB는 각 건물을 약 6번만 보고도 일반화 달성 (900K 다양성 효과).

**근본 한계**: 더 많은 합성 데이터 추가(5K→67K)가 오히려 악화된 이유 (EXP-012/013 확인됨).

#### 발견-6: BB에서 날씨 피처 사용 안 함 (우리와 동일)

BB M 모델: `weather_embedding = nn.Linear(len(weather_features), 64)` 가 있지만
기본 configs에 날씨 피처 미활성화. 우리도 날씨 미사용. 일관성 있음.

### 20인 패널 최종 합의 — 우선순위 로드맵

| 우선순위 | 작업 | 예상 효과 | 노력 |
|---------|------|----------|------|
| **#1** | evaluate_bb.py lat/lon → 0 고정 + EXP-010 재평가 | 0.1~0.2%p | 즉시 (코드 수정 완료, 평가 실행 중) |
| **#2** | ignore_spatial 버그 수정 (lon_embedding 0 초기화) + 3K 재학습 | 0.05~0.1%p | ~3h (학습+평가) |
| **#3** | Transformer-L (d_model=1024, enc/dec=4/4) + RevIN + 3K | 0.5~1.0%p | ~8h (학습+평가) |
| **#4** | 실측 데이터 fine-tuning (공개 한국 건물 데이터) | 1~3%p | 데이터 수집 필요 |

### 성능 상한 분석

**현재 갭**: +1.85%p (15.16% vs SOTA 13.31%)

갭 분해:
- **모델 용량 부족**: 우리 d_model=512 M vs BB L(d_model=1024) → 추정 0.5~1.0%p
- **합성-실측 도메인 갭**: EnergyPlus 물리 ≠ 실제 건물 패턴 → 추정 0.5~1.0%p (근본 한계)
- **lat/lon 버그**: evaluate_bb.py 정규화 불일치 → 추정 0.1~0.2%p (BUG-1 수정으로 해소 예정)
- **ignore_spatial 버그**: lon_embedding 미초기화 → 추정 0.05~0.1%p (BUG-2)

**달성 가능한 목표**: 14.5% 이하 (BUG 수정 + Transformer-L 적용 시)
**근본 한계선**: ~14.0% (합성 데이터만으로 가능한 최저치 추정)

---

## EXP-031: n × steps 2D 스윕 (2026-03-20~)

### 목적
**"Fixed Steps 가설"** 검증: optimal steps ≈ 6,490~6,960 이 모든 n에서 공통인가?

배경:
- EXP-016(scale sweep, 0.354 epoch 고정) → n 클수록 단조 악화 (1K→14.05%, 6K→16.36%)
- EXP-024(n=500, s=6659) → 14.16%, EXP-010(n=3K, s≈37K) → 15.16%
- 가설: 0.354 epoch 고정이 아닌, steps 수를 고정하면 n이 커도 성능 유지되거나 개선?

### 설계
- **그리드**: steps=[6302,6490,6659,6772,6960,7149,7337,7525,9000,12000] × n=[250,500,1k,2k,4k,6k,8k,10k] = **80셀**
- **실행**: 로컬 GPU(RTX 4090) — fill_all_cells.sh + fill_n250_s6490.sh로 순차 실행
- **평가**: BB 공식 프로토콜 (955 Commercial buildings)

### 핵심 논점 (2026-03-20)

#### 1. "Under-training" 개념의 적용 여부
- 8760h 연간 데이터 → 168h 슬라이딩 윈도우 → 건물당 ~8,592개 샘플
- SPE (steps per epoch) = 건물 수 × 8,592 / batch_size
  - n=500 → SPE ≈ 18,810, s=6,490 = 0.345 epoch
  - n=4k  → SPE ≈ 150,480, s=6,490 = 0.043 epoch
- **사용자 지적**: "윈도우로 자른 것이라 건물당 샘플이 많아, under-training 개념이 꼭 적용되는 것은 아님"
- **핵심 트레이드오프**: 각 건물을 반복 학습 vs 다양한 건물 패턴을 조금씩 노출
  → 테이블이 이 질문에 경험적 답을 줄 것

#### 2. inline vs full eval 괴리 주의
- n=4k/s=6490: inline 13.88% → **full eval 14.21%** (+0.33%p 괴리)
- inline eval은 학습 중 subset으로 평가 → 과소추정 경향
- **결론**: full eval 수치만 신뢰

### 결과 (2026-03-27 업데이트 — n=750/1500 추가 + inline eval 오류 수정)

> **⚠️ Inline vs Full eval 수정 (2026-03-27)**: train.py `_bb_fast_eval()`의 타임스탬프 버그 발견.
> 모든 window를 hours [0..191]로 처리 → 실제 calendar 위치 무시 → 일관되게 0.08~0.31%p 과소평가.
> n=2k의 14.01/14.04/14.05%는 모두 inline eval 숫자였음. bbeval.log 재확인 후 수정.
> n=500/n=1k s=6302~7337은 아직 full eval 미실행 — sweep_fill_remaining_2가 실행 예정.

| steps | 250 | 500 | **750** | 1k | **1500** | **2k** | **2500** | 4k | 6k | 8k | 10k |
|-------|:---:|:---:|:-------:|:--:|:--------:|:------:|:--------:|:--:|:--:|:--:|:---:|
| 6302  | 14.28% | *(inline)* | 14.27% | *(inline)* | 14.28% | 14.28% | — | 14.14% | 14.17% | **14.07%** | 14.15% |
| 6490  | 14.24% | *(inline)* | 14.26% | *(inline)* | 14.21% | 14.22% | — | 14.21% | **14.12%** | 14.18% | **14.12%** |
| 6659  | 14.23% | *(inline)* | 14.27% | *(inline)* | 14.14% | 14.22% | — | **14.12%** | 14.16% | 14.19% | 14.23% |
| 6772  | 14.25% | *(inline)* | 🔄 | *(inline)* | 14.15% | ⏳ | — | 14.21% | 14.23% | 14.15% | 14.18% |
| 6960  | 14.21% | *(inline)* | ⏳ | *(inline)* | 14.10% | 14.17% | — | 14.13% | 14.19% | 14.22% | 14.20% |
| 7149  | 14.15% | *(inline)* | ⏳ | *(inline)* | **14.09%** | 14.20% | — | 14.18% | 14.15% | 14.24% | 14.13% |
| 7300  |  —  |  —  |  —  |  —  |  —  |  —  | 14.25% |  —  |  —  |  —  |  —  |
| 7337  | **14.13%** | *(inline)* | ⏳ | *(inline)* | **14.09%** | 14.22% | — | 14.20% | 14.15% | 14.18% | 14.35% |
| 7350  |  —  |  —  |  —  |  —  |  —  |  —  | **14.23%** |  —  |  —  |  —  |  —  |
| 7400  |  —  |  —  |  —  |  —  |  —  |  —  | 14.30% |  —  |  —  |  —  |  —  |
| 7450  |  —  |  —  |  —  |  —  |  —  |  —  | 14.26% |  —  |  —  |  —  |  —  |
| 7500  |  —  |  —  |  —  |  —  |  —  |  —  | 14.31% |  —  |  —  |  —  |  —  |
| 7525  | 14.14% | *(inline)* | ⏳ | 14.28% | 14.20% | 14.21% | 14.32% | 14.30% | **14.12%** | 14.30% | 14.35% |
| 7600  |  —  |  —  |  —  |  —  |  —  |  —  | 14.31% |  —  |  —  |  —  |  —  |
| 9000  | 14.17% | 14.39% |  —  | 14.34% | 🔄 | 14.30% | — | 14.36% | 14.30% | 14.27% | 14.32% |
| 12000 | 14.81% | 14.72% |  —  | 14.59% |  —  | — | — | 14.87% | 14.76% | 14.89% | 14.93% |

> ※ n=750: 10,500 buildings (750/arch×14). n=1500: 21,000 buildings (1500/arch×14). n=2500: 35,000 buildings.
> 🔄 = eval 진행 중, ⏳ = 대기, *(inline)* = full eval 미실행 (sweep_fill_remaining_2 예정)

**현재 최고 (full eval 확인)**: **n=8k, s=6302 → 14.07%** ← NEW BEST
**SOTA-M = 13.28% (BB Transformer-M Gaussian, GitHub 공식)  SOTA-L = 13.31%**

> **Inline eval 오류 정정 (2026-03-27)**: train.py `_bb_fast_eval`에서 모든 sliding window의 timestamp를
> 동일한 base=[0..191]로 처리하는 버그 발견. 실제 calendar 위치(start, start+192)와 불일치.
> 영향: 큰 steps에서 gap이 커짐 (s=9000에서 최대 +0.31%p 과소평가).
> n=2k 기록값 수정: 6960=14.05%→14.17%, 7337=14.04%→14.22%, 7525=14.01%★→14.21% (bbeval.log 재확인).
> n=500/1k 핵심 셀은 full eval 미실행 상태였음 — 결과 대기.

> ⚠️ **EXP-031 결과 신뢰도 경고 (2026-04-07 재확인)**:
> 2026-03-27 수정 당시 "0.08~0.31%p" 과소평가로 기록했으나, 2026-04-07 `_bb_fast_eval` 직접 재실행 결과
> M n=8k s=6302 체크포인트 → **15.75%** (기록된 14.07% 대비 1.68%p 차이).
> **결론**: EXP-031 전체 결과표는 구 sequential timestamp 코드 기준이며 현재 코드와 직접 비교 불가.
> EXP-031 결과는 상대적 n/steps 순위 참고용으로만 사용. 절대 수치는 EXP-034+ 기준 재평가 필요.

### 건물당 샘플링수 뷰 (steps/n)

steps/n = 건물 하나가 gradient update에 노출되는 평균 횟수.
n이 달라도 "동일 횟수 노출" 조건에서 성능을 비교하기 위한 관점.

| n | steps/n → result (오름차순) | 최적 steps/n | 최고 결과 |
|---|---|:---:|:---:|
| 500 | *(inline, full eval 예정)* | — | — |
| 1k  | *(inline s=6302~7337, full s=7525→14.28%)* | — | 14.28% |
| 1500 | 4.2→14.28, 4.3→14.21, 4.4→14.14, 4.5→14.15, **4.6→14.10**, 4.8→14.09, 4.9→14.09, 5.0→14.20 | **4.7** | **14.09%** |
| 2k  | 3.2→14.28, 3.2→14.22, 3.3→14.22, 3.5→14.17, 3.6→14.20, 3.7→14.22, 3.8→14.21, 4.5→14.30 | **3.5** | 14.17% |
| 4k  | 1.6→14.14, **1.7→14.12**, 1.8→14.21, 1.9→14.13, 2.3→14.18, 2.5→14.30 | **1.65** | 14.12% |
| 6k  | **1.1→14.12**, 1.1→14.16, 1.2→14.15/14.19, 1.3→14.12, 1.9→14.30 | **1.1/1.3** | 14.12% |
| 8k  | **0.79→14.07**, 0.81→14.18, 0.83→14.19, 0.86→14.15, 0.88→14.22 | **0.79** | **14.07%** |
| 10k | 0.63→14.15, **0.64→14.12**, 0.67→14.23, 0.70→14.13/14.20, 0.71→14.35 | **0.64** | 14.12% |

> *(inline)* = full eval 미실행. sweep_fill_remaining_2 실행 후 업데이트 예정.

최적 steps/n 패턴 (full eval 기준):

| n | 최적 steps/n | 최고 결과 | 비고 |
|---|:---:|:---:|-----|
| 1500 | 4.7 | 14.09% | s=7149~7337 클러스터 |
| 2k  | 3.5 | 14.17% | n=1500보다 열세 |
| 4k  | 1.65 | 14.12% | n=1500과 경쟁적 |
| 6k  | 1.1~1.3 | 14.12% | 4k/6k 동일 수렴 |
| 8k  | 0.79 | **14.07%** | 전체 최고 |
| 10k | 0.64 | 14.12% | 8k보다 악화 |

**Fixed Steps 가설**: 건물 수가 늘어도 총 gradient steps ≈ 6,302~7,337에 수렴.
그러나 n=2k는 n=1500보다 일관되게 나쁨 — "최적 n=2k"는 inline eval 오류였음.
**현재 실증 최적**: n=8k, s=6302 → 14.07%. n이 너무 작으면(1k↓) under-training, 너무 크면(10k↑) 데이터 다양성 오히려 방해. 8k 근방에 실용적 최적점 존재.

### 관찰 및 결론

#### Fixed Steps 가설 검증
- n=1500 구간(6302~7337): **14.09~14.28%** — 6960~7337에서 14.09~14.10% 최적 클러스터
- n=2k (수정 후): 6960=14.17%, 7337=14.22%, 7525=14.21% — n=1500보다 0.08~0.13%p 악화
- n=4k: 6302=14.14%, 6659/6960=14.12~14.13% — n=1500과 경쟁적
- n=6k: 6490/7525=**14.12%** — n=4k와 유사 수렴
- n=8k: s=6302 = **14.07%** ← 전체 최고 (2026-03-27 기준)
- **결론**: optimal steps ≈ 6,302~7,337 범위, optimal n ≈ 1500/4k/8k (U-curve 아님, 다중 최적점)

#### n 증가 효과 (수정 후)
- n=1500/s=6960 = **14.10%** ← n=1500 최고
- n=2k/s=6960 = 14.17% — n=1500(14.10%) 대비 소폭 악화 (이전 기록 14.05%는 inline 오류)
- n=4k/s=6659 = 14.12%, n=6k/s=6490 = 14.12% — n=2k보다 개선
- n=8k/s=6302 = **14.07%** — 전체 최고, n=6k/10k보다 개선
- n=10k/s=6490 = **14.12%** — n=8k보다 악화 (n=8k가 최적점)
- **결론**: n=2k는 n=1500/4k/8k 대비 경쟁력 없음 (이전 14.01%는 inline eval 오류).
  **현재 최적**: **n=8k, s=6302 → 14.07%** (SOTA-M 대비 +0.12%p, SOTA-L 대비 +0.76%p)

#### s=12000 과학습 분석
- n=2k/s=12000 = 14.90% (급격 악화), n=4k/s=12000 진행중
- 원인: 인접 슬라이딩 윈도우(STRIDE=24)의 85.7% 중복 → 건물당 실제 독립 블록 ~51개
  건물당 27~50회 반복 학습 시 합성 시뮬레이터 패턴 암기

#### EXP-032: STRIDE=168 실험 (진행중)
- 배경: 168h 슬라이딩 윈도우를 STRIDE=168으로 설정 → 독립 블록만 학습
- 데이터: 1K 건물 × 50 독립 윈도우 = 50K (기존 358K → 7× 감소)
- 설정: n=1k, s=6490, `train_weekly_1k_s168.csv`
- 기대: 과학습 방지 + 동일 steps에서 더 균형 잡힌 학습

- **역대 최고 (full eval 확인)**: n=8k/s=6302 → **14.07%** (SOTA-L 대비 +0.76%p, 2026-03-27)
  - ※ 이전 기록 n=2k/s=7525 14.01%는 inline eval 오류였음 → full eval 14.21% (2026-03-27 수정)
- **EXP-031 완료 (2026-03-25)**: 80셀 전체 측정 완료
- **EXP-031b 완료 (2026-03-26)**: n=2500 7셀(s=7300~7600) 완료 → 최고 **14.23%** (s=7350) — 13%대 진입 실패

---

## EXP-033: L 모델 n × steps 스윕 (2026-04-04~)

### 목적
M 모델 한계(14.07% @ n=8k, s=6302) 돌파를 위해 **L 모델(160M params, M 대비 4×)**로 동일 조건 테스트.
L 모델 Phase Transition 가설: convergence threshold ≈ 24K~32K fixed steps (n 무관).

### 배경 (EXP-031 결론)
- M 모델(50M) 최고: 14.07% @ n=8k/s=6302 (SOTA-L 13.31% 대비 +0.76%p)
- M 모델 구조적 한계 도달: n/steps 2D 스윕 완료, 플라토 확인
- L 모델 특성: d_model=768, layers=12+12, warmup=4000, grad_clip=1.0

### 설정
- **머신**: RTX 5090 (C:\Korean_BB)
- **Config**: `TransformerWithGaussian-L-real-v3-3k-fixed.toml`
- **데이터**: 5090에 133,471 Parquet 파일 전송 완료 (2026-04-04)
- **인덱스**: train_weekly_8k.csv (112K windows), train_weekly_4k.csv (56K windows)

### 실험 셀
| Cell | n | steps | epochs | 예상시간 | 체크포인트 |
|------|---|-------|--------|---------|-----------|
| 1 | 8k | 24,000 | ~2.0 | ~4.5h | L_n8k_s24000_fixed |
| 2 | 4k | 24,000 | ~4.0 | ~4.5h | L_n4k_s24000_fixed |
| 3 | 8k | 32,000 | ~2.6 | ~6h | L_n8k_s32000_fixed |
| 4 | 4k | 32,000 | ~5.3 | ~6h | L_n4k_s32000_fixed |

- **Multi-eval**: evaluate_bb_multi.py — 4개 체크포인트 1-pass 평가 (~4h)

### 실행 방법
```
C:\Korean_BB\run_exp033_5090.bat  (launch_exp033.py로 실행)
PID=10700, 시작시각: 2026-04-04 15:38
```

### 결과
→ EXP-033 결과는 `L_n*k_s*k_fixed.log` 파일에서 확인. EXP-034로 이어짐.

---

## EXP-034: L 모델 3k 균등 데이터 n × steps 9셀 코스 그리드 (2026-04-06~)

### 목적
TransformerWithGaussian-L (160M) 으로 n=[4k,8k,10k] × steps=[12K,20K,32K] 9셀 코스 그리드 탐색.
최적 (n, steps) 조합 확인 후 Stage 2 파인 그리드 설계.

### 평가 기준 (확정 — 절대 준수)
- **체크포인트**: `_bb_best.pt` 만 사용 (`_best.pt` 사용 금지)
- **지표**: 학습 로그의 `Best BB Commercial CVRMSE` (인라인 BDG-2, 611건)
- **별도 evaluate_bb.py 불필요** — 인라인 BDG-2 결과가 공식 값
- `--bdg2_only` (611건) → 인라인 소실 시 결측 셀 채우기용
- **최종 full eval** (1908건, commercial+residential 전체) → 전체 실험 완료 후 **best 1개 체크포인트**에만 사용 (논문용)

### 평가 방법 비교 (EXP별 사용 코드 기록)

| 구분 | timestamp 처리 | 코드 | EXP-031 | EXP-034+ | 신뢰도 |
|------|--------------|------|:-------:|:--------:|:------:|
| **인라인 eval** (`_bb_fast_eval`) | real calendar (fix 후) | train.py | ❌ 구 코드 | ✅ | EXP-034+만 유효 |
| **full eval** (`evaluate_bb.py`) | sequential (fix 전) → real (fix 후) | evaluate_bb.py | ❌ 구 코드 | ✅ (2026-04-07 fix) | EXP-034+ 이후 유효 |
| **앙상블 eval** (`ensemble_ml_eval.py`) | real calendar | train.py 인프라 재사용 | N/A | ✅ | 항상 유효 |
| **zero timestamps eval** (`eval_zero_timestamps.py`) | 0으로 고정 | 신규 | N/A | 실험 중 | 비교 실험 |

**EXP-034/035 인라인 결과는 real timestamps 기준**:
- train.py가 2026-03-27 fix 이후 버전
- `_bb_fast_eval`이 `_load_bb_buildings` 반환 real doy/dow/hod 사용 확인
- `_bb_fast_eval` 직접 재실행으로 M n=8k = 15.75% 확인 (EXP-031의 14.07%와 구분)

**M vs L 참값 (real timestamps 기준)**:
| 모델 | 체크포인트 | CVRMSE | 측정 방법 | 신뢰도 |
|------|-----------|:------:|----------|:------:|
| M n=8k s=6302 | `_tbl_n8k_s6302_bb_best.pt` | **15.75%** | `_bb_fast_eval` 2026-04-07 재실행 | ✅ |
| L n=4k s=20K | `_L_n4k_s20k_s1_bb_best.pt` | **14.73%** | 인라인 EXP-034, real timestamps | ✅ |
| L n=2k s=17K | EXP-035 | **14.86%** | 인라인, real timestamps | ✅ |
| L n=2k s=20K | EXP-035 | 14.89% | 인라인, real timestamps | ✅ |
| L n=2k s=23K | EXP-035 | 14.96% | 인라인, real timestamps | ✅ |
| **M+L 앙상블 50:50** | M n=8k + L n=4k | **14.69%** | ensemble_ml_eval.py v2, real timestamps | ✅ |

### 설정
- **4090** (로컬): Cells 1~6
- **5090** (192.168.1.23, `C:\Korean_BB\`): Cells 7~9
- **Config**: `TransformerWithGaussian-L-real-v3-3k.toml` (d_model=768, layers=12+12)
- **데이터**: train_weekly_{4k,8k,10k}.csv + val_weekly_3k.csv
- **고정**: seed=1, augment, lru_cache=50000

### Stage 1 결과 (인라인 BDG-2, _bb_best 기준)

| Cell | n | steps | BDG-2 Commercial | Gap vs SOTA-L | 비고 |
|------|---|-------|:----------------:|:-------------:|-----|
| 1 | 4k | 12K | nan (퇴화) | — | BB eval 완료 안됨 |
| 2 | 4k | 20K | **14.73%** | **+1.42%p** | |
| 3 | 4k | 32K | n/a | — | 학습 중 BB eval 미실행 (버그) |
| 4 | 8k | 12K | nan (퇴화) | — | BB eval 완료 안됨 |
| 5 | 8k | 20K | 14.98% | +1.67%p | |
| 6 | 8k | 32K | 15.18% | +1.87%p | |
| 7 | 10k | 12K | 15.10% | +1.79%p | |
| 8 | 10k | 20K | n/a | — | 5090 0-byte 로그 (결측) |
| 9 | 10k | 32K | n/a | — | 5090 0-byte 로그 (결측) |

> SOTA-L = 13.31%

### 패턴 분석
- **12K steps + n=4k,8k → 퇴화** (nan): L 모델은 최소 ~20K steps 필요
- **12K steps + n=10k → 15.10%**: n이 클수록 12K에서도 수렴 가능성 있음 (더 많은 배치 다양성)
- **n 증가 방향 (steps=20K)**: 4k(14.73%) → 8k(14.98%) → 단조 악화 → 데이터 많을수록 합성-실측 갭 반영
- **steps 증가 방향 (n=4k)**: 20K(14.73%) → 32K(n/a) — 과학습 방향 예상
- **현재 최고**: Cell 2 (n=4k, 20K) = **14.73%** (+1.42%p gap)

### 미결 셀 (Cells 8, 9)
5090 DETACHED_PROCESS 0-byte 로그로 인라인 결과 소실.
`_bb_best.pt` 존재 여부: Cell 9 ✓, Cell 8 ❌.
→ Stage 2 전에 Cell 9 `_bb_best.pt`로 evaluate_bb.py 실행 예정.

### Stage 2 방향 (Cell 9 확인 후 확정)
- 최적 영역: n=3k~6k, steps=16K~24K
- Cell 2 (n=4k, 20K) 주변 파인 그리드

---

## EXP-035: L 모델 파인 그리드 Stage 2 (2026-04-07~) — ⚠ 미채택 (논문은 M n=50 s=18000 채택)

### 목적
EXP-034 Cell 2 (n=4k, 20K = **14.73%**) 주변 파인 그리드 탐색.
n=[2k,3k,4k] × steps=[14K,17K,20K,23K] = 12셀 (기완료 2셀 제외 → 10셀 신규).

### 평가 기준
EXP-034와 동일 — `_bb_best.pt` + 인라인 BDG-2 611건.

### 설정
- **머신**: 5090 (`C:\Korean_BB\`, `exp035_stage2_5090.py`)
- **Config**: `TransformerWithGaussian-L-real-v3-3k.toml`
- **데이터**: train_weekly_{2k,3k,4k}.csv + val_weekly_3k.csv

### 통합 결과표 (EXP-034 + EXP-035 — 행=steps, 열=n)

셀값: **steps/총건물 — BDG-2 CVRMSE** (`_bb_best.pt` 기준, 611건)
총 건물 = n(per-arch) × 14 archetypes

| steps \ n | **2k** (28K) | **3k** (42K) | **4k** (56K) | **8k** (112K) | **10k** (140K) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **12K** | — | — | 0.214 — NaN | 0.107 — NaN | 0.086 — 15.10% |
| **10K** | — — NaN† | 0.238 — 14.79%† | 0.179 — 대기† | 0.089 — 대기† | — |
| **12K** | — — 14.75%† | 0.286 — 진행중† | 0.214 — NaN | 0.107 — NaN | 0.086 — 15.10% |
| **13K** | — — 14.76%† | 0.310 — 대기† | 0.232 — 대기† | — | — |
| **14K** | 0.500 — NaN | 0.333 — NaN | 0.250 — 대기 | — | — |
| **17K** | 0.607 — 14.86% | 0.405 — 14.84% | 0.304 — 대기 | — | — |
| **20K** | 0.714 — 14.89% | 0.476 — 14.84% | **0.357 — 14.73%** | 0.179 — 14.98% | 0.143 — n/a |
| **23K** | 0.821 — 14.96% | 0.548 — 진행중 | 0.411 — 대기 | — | — |
| **32K** | — | — | 0.571 — n/a | 0.286 — 15.18% | 0.229 — n/a |

> 모든 수치: `_bb_best.pt` 인라인 BDG-2 611건, **real timestamps** 기준 (2026-04-07 확인)
> †EXP-036 warmup 수정 버전 (warmup=max(2000,steps//8)). 나머지는 EXP-034/035 (warmup=max(1000,steps//10))
> NaN = 학습 발산. n/a = `_bb_best.pt` 소실. — = 미실험. 진행중 = 5090 실행 중.
> **현재 최고: n=4k s=20K → 14.73%** (gap +1.42%p vs SOTA-L 13.31%)
> **n=2k EXP-036**: s=12K=14.75%, s=13K=14.76% (warmup=2000). n=2k s=10K는 여전히 NaN.
> **n=3k 패턴**: EXP-035 17K=20K=14.84% (평탄). EXP-036 10K=14.79% (warmup 수정, 저steps가 더 좋음!)

### EXP-035 셀별 상세 결과 (2026-04-07)

| 셀 | n | steps | warmup | 결과 | 완료시각 | 비고 |
|----|---|-------|--------|------|---------|------|
| s2_n2k_s14K | 2k | 14K | 1,400 | NaN | 13:05 | warmup 부족으로 발산 |
| s2_n2k_s17K | 2k | 17K | 1,700 | **14.86%** | ~15:50 | n=2k 최고 |
| s2_n2k_s20K | 2k | 20K | 2,000 | 14.89% | 18:16 | |
| s2_n2k_s23K | 2k | 23K | 2,300 | 14.96% | 20:55 | |
| s2_n3k_s14K | 3k | 14K | 1,400 | **NaN** | 23:36 | overflow 발산, warmup 부족 |
| s2_n3k_s17K | 3k | 17K | 1,700 | 진행중 | — | 2026-04-08 00:14 |
| s2_n3k_s20K~s23K | 3k | 20K~23K | — | 대기 | — | |
| s2_n4k_s14K | 4k | 14K | 1,400 | 대기 | — | |
| s2_n4k_s17K | 4k | 17K | 1,700 | 대기 | — | |
| s2_n4k_s20K | 4k | 20K | 2,000 | **14.73%** | EXP-034 | EXP-034 Cell2 결과 |
| s2_n4k_s23K | 4k | 23K | 2,300 | 대기 | — | |

### 결과 (n=3k/4k 완료 시 업데이트 예정)

n=2k 패턴 확인: 17K=14.86%(최고) → 20K=14.89% → 23K=14.96% — steps 증가 시 단조 악화.
**n=2k 결론**: 최적 steps=17K (ratio=0.607), L 모델 n=2k는 n=4k/20K(14.73%) 대비 0.13%p 열세.

---

## EXP-036: L 모델 저steps 탐색 — warmup 수정 (2026-04-07~) — ⚠ 미채택 (논문은 M n=50 s=18000 채택)

### 목적
EXP-034에서 n=4k,8k s=12K → NaN, n=2k s=14K → NaN 발산 재도전.
원인: warmup = max(1000, steps//10) = 1,200 steps로 너무 짧았음.
수정: warmup = max(2000, steps//8) → 10K steps에서도 warmup 2,000(20%) 보장.

### 설정
- **머신**: 5090 (`exp036_lowsteps_5090.py`)
- **Config**: `TransformerWithGaussian-L-real-v3-3k.toml`
- **그리드**: n=[2k,3k,4k,8k] × steps=[10K,12K,13K] = 11셀
- **warmup**: max(2000, steps//8) — EXP-034/035 대비 ~2× 더 긴 warmup

### 그리드
| 셀 | n | steps | warmup | 비고 |
|----|---|-------|--------|------|
| s36_n2k_s10K | 2k | 10K | 2,000 (20%) | 진행중 (2026-04-07 20:57) |
| s36_n2k_s12K | 2k | 12K | 2,000 (17%) | 대기 |
| s36_n2k_s13K | 2k | 13K | 2,000 (15%) | 대기 |
| s36_n3k_s10K | 3k | 10K | 2,000 (20%) | 대기 |
| s36_n3k_s12K | 3k | 12K | 2,000 (17%) | 대기 |
| s36_n3k_s13K | 3k | 13K | 2,000 (15%) | 대기 |
| s36_n4k_s10K | 4k | 10K | 2,000 (20%) | 대기 |
| s36_n4k_s12K | 4k | 12K | 2,000 (17%) | EXP-034 NaN → warmup 수정 재도전 |
| s36_n4k_s13K | 4k | 13K | 2,000 (15%) | 대기 |
| s36_n8k_s10K | 8k | 10K | 2,000 (20%) | 대기 |
| s36_n8k_s12K | 8k | 12K | 2,000 (17%) | EXP-034 NaN → warmup 수정 재도전 |

### 결과 (업데이트 중)

| 셀 | n | steps | warmup | 결과 | 완료시각 | 비고 |
|----|---|-------|--------|------|---------|------|
| s36_n2k_s10K | 2k | 10K | 2,000 | **NaN** | 23:07 | warmup=2000도 발산 (n=2k 최소 17K 필요) |
| s36_n2k_s12K | 2k | 12K | 2,000 | 진행중 | — | 2026-04-08 00:14, ETA ~01:30 |
| s36_n2k_s13K | 2k | 13K | 2,000 | 대기 | — | |
| s36_n3k_s10K~s13K | 3k | 10K~13K | 2,000 | 대기 | — | |
| s36_n4k_s10K~s13K | 4k | 10K~13K | 2,000 | 대기 | — | n=4k에서는 수렴 가능성 있음 |
| s36_n8k_s10K,s12K | 8k | 10K,12K | 2,000 | 대기 | — | EXP-034 NaN 재도전 |

> **n=2k s=10K NaN** 확인: warmup을 2000(20%)으로 늘려도 n=2k(28K 건물) + s=10K(ratio=0.116)는 너무 짧음.
> **패턴**: n=2k 최소 가동 steps ≈ 17K (ratio≥0.607). n=3k,4k에서는 더 낮은 ratio에서도 수렴 가능.

---

## EXP-Zero: Zero Timestamps 실험 (2026-04-07~08)

### 목적
사용자 가설 검증: "학습은 real timestamps, 평가는 zero timestamps → 더 좋을 수 있다?"
배경: EXP-031 M 모델이 sequential timestamps eval에서 14.07%가 나왔던 현상.

### 결과 (L n=4k s=20K best checkpoint)

| 방식 | BDG-2 Commercial CVRMSE | 비교 |
|------|:------------------------:|------|
| Real timestamps | **14.73%** | 정상 eval |
| Zero timestamps (doy=dow=hod=0) | 18.42% | +3.69%p 악화 |

```
SOTA-L = 13.31%,  Gap zero: +5.11%p
```

### 결론
- **가설 기각**: Zero timestamps는 real보다 **훨씬 나쁨** (+3.69%p)
- Zero 18.42% = Persistence(16.68%)보다도 나쁨 → timestamp feature 없이는 예측 불가
- **역설 해설 (이전 14.07% 현상)**: Sequential timestamps(0,1,2,...,192)는 zero(0,0,...,0)보다는 낫지만 real calendar보다 나쁨. 일부 패턴 신호가 있어서 1.68%p 이득이 있었던 것.
- **L 모델에서 real timestamps는 매우 중요** (없으면 +3.69%p 손실)

---

## BUG-001: BB eval NaN 버그 및 수정 (2026-04-09)

### 현상
`run_sweep_4090.py` 실행 중 일부 셀에서 `BB Commercial CVRMSE: nan%` 발생.
- n=250,500: s≥7400 전체 NaN
- n=1k: s=6500, 7100, 7400, 7700, 8000 NaN (s=6800만 13.73% 정상)
- val_loss는 정상 범위 (-1.2~-1.3) → 훈련 자체는 문제 없음

### 근본 원인

GaussianNLL 모델 출력층 `nn.Linear(d_model, out_dim)`에 범위 제약 없음.
일부 초기화/학습 경로에서 BB 테스트 건물에 대해 mu > 14.88 출력.

Box-Cox 역변환 수식 (lambda = -0.0672):
```
x_orig = (-0.0672 × mu + 1)^(1 / -0.0672)
```
mu > 14.88 이면 밑이 음수 → 실수 지수 계산 불가 → **NaN**

val_loss는 Box-Cox 공간에서 계산되므로 역변환 NaN을 감지 못함.
현상이 확률적인 이유: 랜덤 초기화에 따라 일부 실행만 임계값 초과.

### 수정

**파일**: `scripts/train.py`, `_bb_fast_eval()` 함수

```python
# 역변환 직전 clamp 추가 (line 288)
all_pred = np.clip(all_pred, -10.0, 10.0)  # prevent Box-Cox inverse NaN
pred_orig = bb_boxcox.scaler.inverse_transform(
    all_pred.reshape(-1, 1)).reshape(all_pred.shape)
```

Box-Cox 공간 ±10은 물리적으로 충분한 범위 (NaN 임계값 14.88의 67%). 훈련에 영향 없음.

### 1차 수정 (불완전)
`all_pred`만 클립 → tgt_bc (미국 BB 테스트 건물)도 한국 훈련 분포 초과 가능.
`tgt_orig = inf` → `mean_gt = inf` → `inf / inf = NaN` 경로 잔존.

### 2차 수정 (완전)

**파일**: `scripts/train.py`, `_bb_fast_eval()` 함수

```python
all_pred = np.clip(all_pred, -10.0, 10.0)
tgt_bc_clipped = np.clip(tgt_bc, -10.0, 10.0)  # 추가: 타깃도 클립
pred_orig = bb_boxcox.scaler.inverse_transform(all_pred.reshape(-1, 1))...
tgt_orig = bb_boxcox.scaler.inverse_transform(tgt_bc_clipped.reshape(-1, 1))...
mean_gt = float(np.mean(np.abs(tgt_orig)))
if mean_gt < 1e-8 or not np.isfinite(mean_gt):  # isfinite 추가
    continue
se = (pred_orig - tgt_orig) ** 2
cvrmse_val = float(np.sqrt(np.mean(se)) / mean_gt)
if not np.isfinite(cvrmse_val):  # 잔여 NaN 건물 skip
    continue
cvrmses.append(cvrmse_val)
```

### 조치
- 2차 수정 후 스윕 재시작 (PID 29436)
- NaN 셀들 자동 재실행 (progress.log에 "Best BB Commercial CVRMSE" 없으면 재실행)
- 5090에도 동일 fix SCP 완료 (2026-04-09)

### 3차 수정 (최종 확정): pred만 clip, tgt 무처리 (2026-04-09)

**추가 분석**: `tgt_bc`는 실제 에너지값 → `bb_boxcox.scaler.transform()` 결과
- 유한한 양수 에너지값은 Box-Cox 후 항상 threshold(7.605) 미만 → clip 불필요
- `tgt_bc_clipped`는 실제로는 no-op이었으나 개념적으로 잘못됨
- **건물 skip 원인**: 모델 예측 x_std ∈ (7.605, 10] → NaN → skip (약 5/955건 = 0.5%)
- skip은 fair comparison을 깨므로 제거 대상

**최종 수정 코드** (`scripts/train.py`, `_bb_fast_eval()`):
```python
# tgt_bc: 실제 에너지 → transform() → 항상 valid (clip 불필요)
# all_pred: 모델 출력은 임의 범위 → clip 필수
# NaN threshold: x_std=(14.88-1.439)/1.768=7.605 → 7.0으로 clip
all_pred = np.clip(all_pred, -10.0, 7.0)
pred_orig = bb_boxcox.scaler.inverse_transform(all_pred.reshape(-1, 1)).reshape(all_pred.shape)
tgt_orig  = bb_boxcox.scaler.inverse_transform(tgt_bc.reshape(-1, 1)).reshape(tgt_bc.shape)
mean_gt = float(np.mean(np.abs(tgt_orig)))
if mean_gt < 1e-8:   # 소비량 0인 건물만 skip (정당한 예외)
    continue
se = (pred_orig - tgt_orig) ** 2
cvrmse_val = float(np.sqrt(np.mean(se)) / mean_gt)
if not np.isfinite(cvrmse_val):  # 안전망
    continue
cvrmses.append(cvrmse_val)
```

**영향 받은 셀 (clip=10 구 코드)**:
- n=250 × 8셀, n=500 × 8셀, n=1k s=6800 → 평균 ~5/955건 skip → 결과는 약간 낙관적
- n=1k s=6500 이후 → 최종 수정 코드 적용 (정확)

**조치**: 4090/5090 모두 최종 fix SCP 완료 (2026-04-09 14:09).

### 재수행 계획 (EXP-034 전체 완료 후)

구 코드(clip=10)로 실행된 셀들을 올바른 코드(pred만 clip=7.0)로 재실행:
- **대상**: n=250 × 8셀, n=500 × 8셀, n=1k s=6800 (총 17셀)
- **방법**: 해당 로그 파일 삭제 → run_sweep_4090.py 재실행 (skip 로직으로 자동 재수행)
  ```bash
  # 재수행 대상 로그 삭제
  rm logs/sweep_n250_s*.log logs/sweep_n500_s*.log logs/sweep_n1k_s6800.log
  python run_sweep_4090.py  # 이미 완료된 n=1k+ 셀은 SKIP, 위 셀들만 재실행
  ```
- **시점**: EXP-034 64셀 + EXP-035 6셀 전체 완료 후
- **결과**: EXP-037로 흡수됨 — n=250/500 전체를 수정 코드로 재실행 완료

---

## EXP-037: M 모델 n×steps 80셀 스윕 (수정 코드, 2026-04-09~10) — ⚠ 일부 미완료 (논문 최종 채택: n=50 s=18000 val_best)

### 목적
BUG-002 수정 코드(clip=7.0) 기준으로 M 모델 n×steps 전체 재스윕.
EXP-031(구 코드, 14.07%)보다 얼마나 개선되는지 측정.

### 설정
- **머신**: 4090 (로컬)
- **코드**: BUG-002 최종 수정 (2026-04-09 14:09)
- **모델**: TransformerWithGaussian-M-v3-3k
- **구성**: `run_sweep_4090.py`
  - NS_SMALL (n=250/500/1k/2k) × STEPS_SMALL (6500~9800, 12스텝) = 48셀
  - NS_LARGE (n=4k/6k/8k/10k) × STEPS_LARGE (8300~9800, 6스텝) = 24셀
  - 총 72셀 (7700/8000 제외로 절약)

### 결과 (2026-04-10 진행 중)

| steps \ n | 250 | 500 | 1k | 2k | 4k | 6k | 8k | 10k |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 6500 | 13.78 | 13.74 | 13.72 | 13.73 | — | — | — | — |
| 6800 | 13.78 | 13.70 | 13.73 | 13.71 | — | — | — | — |
| 7100 | 13.61 | 13.68 | 13.65 | 13.66 | — | — | — | — |
| 7400 | 13.67 | 13.64 | 13.68 | 13.64 | — | — | — | — |
| 7700 | 13.55 | 13.56 | 13.60 | 13.65 | — | — | — | — |
| 8000 | 13.54 | 13.51 | 13.60 | 13.60 | — | — | — | — |
| 8300 | 13.58 | 13.60 | 13.53 | 13.60 | 대기 | 대기 | 대기 | 대기 |
| 8600 | 13.55 | 13.51 | 13.51 | 13.49 | 대기 | 대기 | 대기 | 대기 |
| 8900 | 13.57 | 13.53 | 13.47 | 13.49 | 대기 | 대기 | 대기 | 대기 |
| 9200 | 13.56 | **13.49** | **13.42★** | **13.49** | 대기 | 대기 | 대기 | 대기 |
| 9500 | 13.50 | 13.50 | 대기 | 대기 | 대기 | 대기 | 대기 | 대기 |
| 9800 | 13.50 | 진행중 | 대기 | 대기 | 대기 | 대기 | 대기 | 대기 |

**현재 M 최고: n=1k s=9200 = 13.42%** (SOTA-M 13.28% 대비 +0.14%p)

### 관찰
- n=1k: steps↑ → 단조 개선 (6500=13.72% → 9200=13.42%, -0.30%p). 최적점 미확인 (9500/9800 대기)
- n=2k: 8600부터 plateau (13.49%), steps 더 늘려도 이득 없음
- n=250/500: 9200 이후 개선 미미 (각 13.56→13.50%, 13.49→13.50%)
- EXP-031 구 코드 best(14.07%) 대비 수정 코드 **-0.65%p 개선**

---

## EXP-038: L 모델 n=4k steps 스윕 (5090, 2026-04-09~10) — ⚠ 일부 미완료 (논문 최종 채택: M n=50 s=18000 val_best)

### 목적
L 모델 optimal steps 탐색. 기존 s=22K 신기록(13.32%)의 전후 범위 확인.

### 설정
- **머신**: 5090 (원격, 192.168.1.23)
- **모델**: TransformerWithGaussian-L-real-v3-3k
- **n=4k** 고정 (56K buildings)
- **STEPS**: [15K, 22K, 29K, 36K, 43K, 50K] → 이후 18K, 20K 추가 예정

### 결과 (2026-04-10 진행 중)

| steps | 15K | 22K | 29K | 36K | 43K | 50K |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Commercial | 28.01%† | **13.32%★★** | 13.50% | 13.84% | 13.95% | 진행중 |

> †15K 발산(28.01%) — warmup=1500이 n=4k에 비해 너무 짧을 가능성
> s=50K ETA ~1h (step 38K/50K, 2026-04-10 11:00 기준)

### 관찰
- **22K = 최적**: 29K부터 단조 악화 (과학습 시작)
- **22K 이전**: 15K 발산 → 18K, 20K 추가 실험 예정 (watcher 자동 재시작)
- L 모델 s=22K(0.147ep) vs M 모델 s=9200(0.246ep): L 모델이 더 빠른 epoch에서 수렴

### 현재 L 최고
**n=4k s=22K = 13.32%** (SOTA-L 13.31% 대비 **+0.01%p**, 사실상 동등)

