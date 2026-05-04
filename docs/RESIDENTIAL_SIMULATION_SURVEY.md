# Residential Building Energy Simulation Survey

> Korean_BB 프로젝트 — 주거 부문 성능 개선을 위한 문헌 조사 및 공개 자원 정리
> 작성일: 2026-05-04

---

## 1. 배경: 현재 주거 부문 성능

Korean-700 모델(Commercial NRMSE 12.91%)의 BB 프로토콜 전체 평가 결과:

| 구분 | CVRMSE | 비고 |
|------|:------:|------|
| **Commercial (955건)** | **12.91%** | BB SOTA(13.27%) 초과 달성 |
| **Residential (953건)** | **77.14%** | Persistence(16.68%)보다 크게 열위 |
| LCL (713건) | 74.81% | UK London 개별 가구 |
| IDEAL (219건) | 85.53% | Scottish 개별 가구 |
| Borealis (15건) | 61.41% | Canadian 가구 |

### 성능 저하 원인 분석

1. **부하 스케일 갭**: Commercial 평균 ~85 kWh/h vs Residential ~0.2 kWh/h (415배 차이)
2. **분산 구조**: LCL 실측 = 27% periodic + **73% stochastic** → 학습 가능한 신호 부족
3. **패턴 간섭**: 모델이 상업용 패턴(주중/주말, 운영시간)을 강제 → 주거에서 오히려 악화
4. **Naive가 우월**: Context 7일 시간별 평균(76.1%)이 모델(83.7%)보다 낮음 (LCL 기준)

---

## 2. 개별 가구 에너지 시뮬레이션 방법론

### 2.1 Richardson et al. (2010) — Markov Chain Occupancy + Activity

- **핵심**: 10분 단위 Markov chain으로 재실 상태 전이 → 가전기기 활동 확률 → 전력 프로파일 생성
- **입력**: UK Time Use Survey (TUS) 데이터, 가전기기 보유 확률
- **출력**: 1분 단위 전력 프로파일 (조명 + 가전)
- **한계**: 난방/냉방 미포함, UK 생활패턴 특화
- **코드**: CREST Demand Model에 통합
- **참고**: Richardson, I., Thomson, M., Infield, D., Clifford, C. (2010). Domestic electricity use: A high-resolution energy demand model. *Energy and Buildings*, 42(10), 1878-1887.

### 2.2 CREST Demand Model (2015) — 통합 열-전기 모델

- **핵심**: Richardson 모델 확장. 재실→활동→가전+조명+온수+난방 통합
- **구조**: Occupancy → Activity → Appliance/Lighting/DHW/Heating 4개 서브모델
- **강점**: 열부하까지 포함, EnergyPlus 연동 가능
- **코드**: Excel VBA + MATLAB, Loughborough University 공개
- **참고**: McKenna, E., Thomson, M. (2016). High-resolution stochastic integrated thermal–electrical domestic demand model. *Applied Energy*, 165, 445-461.

### 2.3 StROBe (KU Leuven) — District-scale Stochastic

- **핵심**: 벨기에 TUS 데이터 기반, 지구 단위(district) 다가구 동시 시뮬레이션
- **특징**: 가구 간 다양성(diversity factor) 모델링, 피크 동시성 분석
- **언어**: Python, GitHub 공개 (open-IDEAS/StROBe)
- **한계**: 벨기에 생활패턴 기반, 한국 적용 시 TUS 교체 필요
- **참고**: Baetens, R., Saelens, D. (2016). Modelling uncertainty in district energy simulations by stochastic residential occupant behaviour. *JBPS*, 9(4), 431-447.

### 2.4 NREL ResStock (미국 대규모)

- **핵심**: 미국 주거 건물 550K+ EnergyPlus 시뮬레이션, 통계적 샘플링
- **방법**: ACS/RECS 주거 조사 → housing characteristics 확률 분포 → OpenStudio + E+ 시뮬
- **스케줄**: 확률적 가전 스케줄 생성 (occupancy-driven)
- **데이터**: OpenEI OEDI에 전량 공개 (시간별 에너지, 건물 특성)
- **BB 결과**: 550K 시뮬로도 **Residential에서 Persistence를 이기지 못함** (Transformer-M=92.6%, Persist=77.88%)
- **교훈**: 시뮬레이션 스케줄과 실제 개별가구 소비 패턴의 구조적 괴리
- **참고**: Wilson, E. et al. (2022). End-Use Load Profiles for the U.S. Building Stock. NREL/TP-5500-80889.

### 2.5 한국 주거 시뮬레이션 연구

| 연구 | 방법 | 결과 | IDF 공개 |
|------|------|------|:--------:|
| 아파트 EnergyPlus 모델 (2024 RSER) | 실측 기반 캘리브레이션, 대한민국 아파트 | CV(RMSE) 9.4% | ❌ |
| 한국 표준 주거 모델 (KEA) | 에너지효율등급 인증용 표준 모델 | 연간 에너지 검증 | ❌ |
| 한국 생활시간조사 (통계청) | 10분 단위 활동 일지, 5년 주기 | TUS 데이터만 | ❌ |

**한국 주거 IDF 공개 자원 없음** — 한국 아파트 열·전기 통합 IDF는 아직 공개된 사례가 없다.

---

## 3. 공개 데이터셋

### 3.1 실측 데이터

| 데이터셋 | 규모 | 해상도 | 지역 | 접근 |
|----------|------|--------|------|------|
| **LCL** (Low Carbon London) | 5,567가구 | 30분 | UK London | ✅ 완전 공개 (UK Power Networks) |
| **IDEAL** | 255가구 | 1초~1분 | UK Scotland | ✅ 공개 (Edinburgh DataShare, PMC) |
| **Pecan Street / Dataport** | 1,000+가구 | 1분 | US (Texas 중심) | ✅ 학술 무료 (dataport.pecanstreet.org) |
| **REDD** | 6가구 | 1초 | US | ✅ 공개 (MIT, redd.csail.mit.edu) |
| **UK-DALE** | 5가구 | 6초 | UK | ✅ 공개 (UKERC) |
| **Borealis** | 15가구 | 시간별 | Canada | ✅ BB에 포함 |
| **SMART** (UMass) | 5가구 | 1분 | US | ✅ BB에 포함 |
| **Open Power System Data** | 다국적 집계 | 시간별 | EU | ✅ 공개 (open-power-system-data.org) |
| **ResStock OEDI** | 550K+ 시뮬 | 시간별 | US 전역 | ✅ 완전 공개 (NREL OpenEI) |

### 3.2 BB에 이미 포함된 주거 데이터

BB 평가 프로토콜의 953건 주거 건물 구성:

| 데이터셋 | 건물 수 | 출처 | 특성 |
|----------|:-------:|------|------|
| LCL | 713 | UK London | 스마트미터, 30분→시간 리샘플 |
| IDEAL | 219 | Scotland | 전력+가스, 1분→시간 리샘플 |
| Borealis | 15 | Canada | 시간별 전력 |
| SMART | 5 | US Massachusetts | 아파트 5가구 |
| Sceaux | 1 | France | 단독 주택 |

---

## 4. 공개 IDF / 건물 모델 자원

### 4.1 PNNL IECC Residential Prototype Models

- **출처**: Pacific Northwest National Laboratory (energycodes.gov)
- **건물 유형**: SingleFamily Detached, Multifamily LowRise
- **기후대**: ASHRAE Climate Zone 1~8 (US 전역)
- **IECC 버전**: 2006, 2009, 2012, 2015, 2018, 2021
- **다운로드**: https://www.energycodes.gov/prototype-building-models
- **형식**: EnergyPlus IDF
- **한계**: 미국 건축 기준, 한국 아파트와 구조적 차이 큼

### 4.2 NREL ResStock / OpenStudio-HPXML

- **출처**: github.com/NREL/resstock, github.com/NREL/OpenStudio-HPXML
- **방법**: OpenStudio measures → EnergyPlus IDF 자동 변환
- **규모**: 550K+ 건물 특성 조합 가능
- **스케줄**: 확률적 가전/재실 스케줄 자동 생성
- **한계**: US housing stock 특화 (wood frame, HVAC 중심), 한국 RC 아파트 비적용

### 4.3 DOE Commercial Reference Buildings (참고)

- Korean_BB에서 이미 사용 중 (14 archetypes)
- 주거용: MidRise Apartment, HighRise Apartment만 존재
- **이미 v1~v3 시뮬에 포함** — 추가 활용 여지 없음

### 4.4 한국 주거 IDF 상황

| 출처 | 상태 | 비고 |
|------|------|------|
| 에너지효율등급 표준모델 (KEA) | 비공개 | 인증기관 내부 사용 |
| 제로에너지건축 인증 모델 | 비공개 | 심사용 |
| 학술 논문 IDF | 비공개 | 개별 연구자 보유, 미공개 |
| **공개 한국 주거 IDF** | **없음** | — |

---

## 5. 핵심 발견 및 시사점

### 5.1 개별 가구 시뮬레이션의 구조적 한계

BB ResStock 결과가 이를 증명한다:

| 모델 | ResStock 학습 | Residential CVRMSE |
|------|:------------:|:------------------:|
| Transformer-M | 550K buildings | 92.6% |
| Transformer-L | 550K buildings | 79.34% |
| Persistence | — | 77.88% |

**550K 시뮬로도 개별 가구 예측에서 Persistence를 이기지 못함.** 시뮬레이션 스케줄(확률적이라 해도)과 실제 가구의 비정상(non-stationary) 행동 패턴 사이의 구조적 괴리가 원인.

### 5.2 가구 집계(Aggregation) 효과

N가구 집계 시 stochastic 성분이 √N으로 감소:

| 집계 N | 예상 Baseline CVRMSE | 판정 |
|:------:|:--------------------:|:----:|
| 1 | ~83.5% | 학습 불가 |
| 5 | ~36.1% | 부족 |
| 10 | ~25.6% | 한계 |
| **20** | **~19.7%** | **달성 가능 영역** |
| 50 | ~14.3% | 상업용 수준 |
| 100 | ~10.2% | — |

→ **20가구 이상 집계** 시 주기적 신호가 지배적 → 시계열 모델 학습 유효.

### 5.3 한국 주거의 차별점

한국 아파트는 UK/US 개별 가구 대비 예측 용이한 특성:

- **가스 난방/취사** → 전력 프로파일에서 난방 스파이크 제거
- **중앙 급탕** → 전기 온수기 부하 없음
- **아파트 밀집** → 외피 열손실 적음, 안정적 프로파일
- **높은 가전 보급 균일성** → 가구 간 변동 감소

### 5.4 실현 가능한 전략

| 전략 | 데이터 | 예상 효과 | 실현성 |
|------|--------|----------|:------:|
| **A. 집계 레벨 학습** | LCL 20~50가구 묶음 | 20% 이하 가능 | ★★★ |
| **B. Context-adaptive** | BB 기존 데이터 | 모델 구조 변경 필요 | ★★☆ |
| **C. 한국 아파트 시뮬** | 자체 IDF 개발 필요 | 높지만 노력 큼 | ★☆☆ |
| **D. ResStock 활용** | 공개 550K | BB가 이미 실패 입증 | ✗ |

### 5.5 권장 방향

1. **논문 현재 범위**: Commercial-only 보고는 정당 (BB 원본도 동일 관행)
2. **리비전 대비**: "Residential은 개별 가구 수준에서 구조적으로 예측 불가" 논거 준비
3. **후속 연구**: 집계 레벨(20+ 가구) 예측 또는 한국 아파트 단지 단위 모델링

---

## 6. 참고문헌

1. Richardson, I. et al. (2010). Domestic electricity use: A high-resolution energy demand model. *Energy and Buildings*, 42(10), 1878-1887.
2. McKenna, E., Thomson, M. (2016). High-resolution stochastic integrated thermal–electrical domestic demand model. *Applied Energy*, 165, 445-461.
3. Baetens, R., Saelens, D. (2016). Modelling uncertainty in district energy simulations by stochastic residential occupant behaviour. *JBPS*, 9(4), 431-447.
4. Wilson, E. et al. (2022). End-Use Load Profiles for the U.S. Building Stock. NREL/TP-5500-80889.
5. Emami, P. et al. (2023). BuildingsBench: A Large-Scale Dataset and Benchmarks for Short-Term Load Forecasting. *NeurIPS 2023*.
6. PNNL Residential Prototype Building Models. https://www.energycodes.gov/prototype-building-models
7. NREL ResStock. https://github.com/NREL/resstock
8. Pecan Street Dataport. https://dataport.pecanstreet.org
9. Low Carbon London (LCL). UK Power Networks.
10. IDEAL Household Energy Dataset. University of Edinburgh.
