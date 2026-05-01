# Korean_BB v3 시뮬레이션 재검토 보고서

> 작성일: 2026-02-24
> 목적: 8.simulation 표준(NAMING_UNIFICATION.md) 변경에 따른 Korean_BB 전면 재검토
> 상태: **현황 분석 + 마이그레이션 전략 + 신규 시뮬 계획 확정**

---

## 1. 표준 변경 요약

2026-02-24 확정된 `8.simulation/NAMING_UNIFICATION.md` + `8.simulation/CLAUDE.md` 기준:

### 1.1 sim_id 형식 (강제)
```
{SRC}_{BLDG}_{HVAC}_{CITY}_{EMS}_{VAR}_{IDX}
```

| 필드 | 범위 | Korean_BB 적용 |
|------|------|----------------|
| SRC | KBB | KBB |
| BLDG | B01~B16 (DOE) | Korean 아키타입 → DOE 매핑 필요 |
| HVAC | HA~HE | 아키타입별 HVAC 코드 |
| CITY | C01~C10 | C01~C09 (5개 도시 사용) |
| EMS | M00~M15 | 항상 M00 (Korean_BB는 EMS 없음) |
| VAR | VA~VE | **Korean_BB 전용 확장**: Vintage 인코딩 |
| IDX | L001~L400 | LHS 샘플 인덱스 |

### 1.2 공통 시뮬 데이터 스키마 (출력 파일)

```
results/{sim_id}/
├── metadata.json
├── hourly_electricity.npy       # Tier A 15개 NPY
├── hourly_cooling.npy
├── hourly_heating.npy
├── hourly_gas.npy
├── hourly_fans.npy
├── hourly_equipment.npy
├── hourly_lights.npy
├── hourly_zone_temp.npy         ← NEW
├── hourly_indoor_humidity.npy   ← NEW
├── hourly_setpoint_cool.npy     ← NEW
├── hourly_setpoint_heat.npy     ← NEW
├── hourly_peak_demand.npy       ← NEW (Electricity [kWh/h] = kW)
├── hourly_outdoor_temp.npy
├── hourly_outdoor_humidity.npy  ← NEW
└── hourly_solar.npy             ← NEW
```

---

## 2. Korean_BB → 표준 매핑 정의

### 2.1 건물 타입 코드 (BLDG)

| Korean_BB 아키타입 | ASHRAE 템플릿 | DOE 코드 | 비고 |
|------------------|-------------|---------|------|
| office | OfficeMedium | **B02** | ASHRAE MediumOffice.idf 기반 |
| retail | RetailStandalone | **B05** | ASHRAE RetailStandalone.idf 기반 |
| school | SchoolPrimary | **B07** | ASHRAE PrimarySchool.idf 기반 |
| hotel | HotelLarge | **B15** | ASHRAE LargeHotel.idf 기반 |
| hospital | Hospital | **B12** | ASHRAE Hospital.idf 기반 |
| apartment_midrise | ApartmentMidRise | **B16** | ASHRAE MidRiseApartment.idf 기반 |
| apartment_highrise | (Korean 전용) | **B16*** | DOE 비해당 — B16 사용 + 별도 기록 |

> **B16 충돌 해결**: apartment_highrise와 apartment_midrise 모두 B16이지만 HVAC 코드가 다름.
> apartment_midrise = B16/HD, apartment_highrise = B16/HA → sim_id에서 구분 가능.
> 장기적으로 8.simulation 표준에 B17 (KoreanHighRiseApartment) 추가 검토 필요.

### 2.2 HVAC 코드

| Korean_BB 아키타입 | 실제 HVAC | 표준 코드 |
|------------------|---------|---------|
| office | VAV+Chiller (중앙냉방) | **HA** |
| retail | PSZ/Package | **HC** |
| school | VAV+Chiller | **HA** |
| hotel | FCU+Chiller | **HB** |
| hospital | VAV+AHU+Chiller | **HA** |
| apartment_midrise | VRF+개별가스보일러 | **HD** |
| apartment_highrise | 중앙냉방+가스보일러 | **HA** |

### 2.3 도시 코드

| Korean_BB 도시 | 표준 코드 |
|--------------|---------|
| seoul | **C01** |
| busan | **C02** |
| daegu | **C03** |
| gangneung | **C08** |
| jeju | **C09** |

### 2.4 VAR 필드 — Vintage 인코딩 (Korean_BB 전용)

Korean_BB는 VAR 필드를 Vintage 인코딩에 사용 (mcp_model의 variant_A/B와 역할 다름):

| VAR | Korean_BB Vintage | 외벽 U-value 범위 |
|-----|-----------------|----------------|
| VA | v1_pre1990 | 0.58~1.00 W/m²K |
| VB | v2_1991_2000 | 0.47~0.58 W/m²K |
| VC | v3_2001_2010 | 0.36~0.47 W/m²K |
| VD | v4_2011_2017 | 0.21~0.36 W/m²K |
| VE | v5_2018_plus | 0.15~0.22 W/m²K |

> **표준 확장**: 기존 VA~VD (4개) → VA~VE (5개)로 Korean_BB 전용 확장.
> 8.simulation/CLAUDE.md에 이 확장을 기록할 것.

### 2.5 IDX 범위

LHS 샘플 인덱스는 L001~L400 사용 (Vintage×City 조합당 최대 400개 LHS 샘플).

### 2.6 sim_id 예시

| 기존 building_id | 신규 sim_id |
|----------------|-----------|
| `office_v1_pre1990_busan_tmy_p0000` | `KBB_B02_HA_C02_M00_VA_L001` |
| `retail_v3_2001_2010_seoul_tmy_p0150` | `KBB_B05_HC_C01_M00_VC_L001` |
| `hospital_v5_2018_plus_gangneung_tmy_p0000` | `KBB_B12_HA_C08_M00_VE_L001` |
| `apartment_highrise_v2_1991_2000_jeju_tmy_p0099` | `KBB_B16_HA_C09_M00_VB_L100` |

---

## 3. 출력 변수 갭 분석

### 3.1 현재 상태

| 배치 | 수집 변수 수 | 누락 변수 |
|-----|:----------:|---------|
| s1~s3 (3,867건) | 8개 (7 에너지 + 외기온) | Gas, Pumps, WaterSystemsGas, 신규 6개 |
| s4~s5 (2,460건) | 11개 (10 에너지 + 외기온) | 신규 6개 |

### 3.2 Tier A 완성을 위해 필요한 추가 변수

| 변수명 (NPY) | EnergyPlus 출력 | 타입 | 비고 |
|------------|--------------|-----|------|
| hourly_zone_temp | `Zone Mean Air Temperature [C]` | Output:Variable | 면적가중 평균 필요 |
| hourly_indoor_humidity | `Zone Air Relative Humidity [%]` | Output:Variable | 면적가중 평균 필요 |
| hourly_setpoint_cool | `Zone Thermostat Cooling Setpoint Temperature [C]` | Output:Variable | 대표 zone만 |
| hourly_setpoint_heat | `Zone Thermostat Heating Setpoint Temperature [C]` | Output:Variable | 대표 zone만 |
| hourly_outdoor_humidity | `Site Outdoor Air Relative Humidity [%]` | Output:Variable | Environment 레벨 |
| hourly_solar | `Site Direct Solar Radiation Rate per Area [W/m2]` | Output:Variable | GHI 대용 |
| *(hourly_peak_demand)* | *(Electricity:Facility에서 유도)* | 계산값 | kWh/h = kW |

> **zone-level 변수 처리**: 복수 Zone 건물에서 면적가중 평균은 postprocess.py에서 처리.
> modifier.py는 `*` (모든 zone) 수집, postprocess.py에서 집계.

### 3.3 기존 시뮬레이션에서 추가 불가능한 이유

- `hourly_zone_temp`, `hourly_indoor_humidity`, `hourly_setpoint_cool/heat`: Zone-level 변수 → CSV에 없음 → **재시뮬레이션 필요**
- `hourly_outdoor_humidity`, `hourly_solar`: Site-level 변수 → 추가 가능은 하나 → **재시뮬레이션 필요**

---

## 4. 기존 시뮬레이션 마이그레이션 전략

### 4.1 원칙

1. **BB 평가 파이프라인 변경 없음** — 기존 Parquet 형식 유지 (evaluate_bb.py 그대로)
2. **원시 CSV 보존** — `simulations/results_v3_*/` 삭제 금지
3. **점진적 표준 준수** — 신규 시뮬(s6+)부터 새 표준, 기존은 어댑터로 호환

### 4.2 기존 데이터 처리 방법

```
기존 6,327건
  │
  ├─→ data/korean_bb/individual/*.parquet  ← 그대로 유지 (BB 평가용)
  │
  └─→ scripts/migrate_existing_to_npy.py 실행 시
         ├── results/{sim_id}/metadata.json    ← sim_id 형식 적용 + 소스 기록
         ├── hourly_electricity.npy             ← 추출 가능
         ├── hourly_cooling.npy                 ← 추출 가능
         ├── hourly_heating.npy                 ← 추출 가능
         ├── hourly_fans.npy                    ← 추출 가능
         ├── hourly_equipment.npy               ← 추출 가능
         ├── hourly_lights.npy                  ← 추출 가능
         ├── hourly_gas.npy                     ← s4+ 만 추출 가능 (s1~s3: NaN)
         ├── hourly_pumps.npy                   ← s4+ 일부 추출 가능
         ├── hourly_outdoor_temp.npy            ← 추출 가능
         └── (나머지 6개 변수)                   ← NaN 또는 재시뮬 필요
```

### 4.3 마이그레이션 우선순위

| 배치 | 상태 | 조치 |
|-----|------|------|
| s1~s3 (3,867건) | 부분 데이터 | NPY 마이그레이션 후 불완전 Tier A로 저장 |
| s4~s5 (2,460건) | 더 많은 변수 | NPY 마이그레이션 후 부분 Tier A로 저장 |
| s6+ (신규) | 새 표준 | **처음부터 완전 Tier A** |

### 4.4 마이그레이션 스크립트 필요 (신규)

`scripts/migrate_existing_to_npy.py`:
- 기존 `building_id` → `sim_id` 변환 매핑 테이블 생성
- CSV에서 NPY 추출
- 없는 변수는 NaN 배열로 채움
- `metadata.json` 생성 (부분 데이터 명시)

---

## 5. 신규 시뮬레이션 계획 (s6+)

### 5.1 기존 건물 수 현황 (s5 기준, 2026-02-24)

| 아키타입 | 현재 수 | 목표 수 | 필요 추가 |
|---------|:------:|:------:|:-------:|
| office | 2,220 | 10,000 | 7,780 |
| retail | 2,220 | 10,000 | 7,780 |
| school | 430 | 1,800 | 1,370 |
| hotel | 430 | 1,800 | 1,370 |
| hospital | 427 | 1,800 | 1,373 |
| apartment_midrise | 300 | 600 | 300 |
| apartment_highrise | 300 | 600 | 300 |
| **합계** | **6,327** | **26,600** | **20,273** |

### 5.2 s6+ 배치 계획 (1K 단위)

```
s6:  office 1K    (총 3,220)  → BB eval
s7:  office 1K    (총 4,220)  → BB eval
s8:  retail 1K    (총 3,220)  → BB eval
s9:  retail 1K    (총 4,220)  → BB eval
s10: office 1K    (총 5,220)  → BB eval ← 5K 기준점
s11: retail 1K    (총 5,220)  → BB eval
s12: school 500   (총 930)
s13: hotel 500    (총 930)
s14: hospital 500 (총 927)
...
```

### 5.3 LHS 겹침 방지 전략 (s5 중복 문제 해결)

#### 문제
s5에서 retail 1K 시뮬 중 540건이 기존 건물과 중복 → 460건만 추가.
원인: LHS 파라미터 공간을 배치마다 독립 샘플링 → 기존과 겹침.

#### 해결: 글로벌 LHS 파라미터 풀 관리

```python
# configs/lhs_pool/office.json 형식
{
  "archetype": "office",
  "total_samples": 10000,
  "used_indices": [0, 1, 2, ..., 2219],  # 기존 사용 인덱스
  "next_batch_start": 2220
}
```

**구현**:
1. 처음 1회 전체 목표 수(10,000)에 대한 LHS 샘플 생성 및 저장
2. 각 배치는 `next_batch_start`에서 N개 순차 할당
3. `used_indices`로 중복 방지
4. 파일: `configs/lhs_pool/{archetype}.npz`

### 5.4 신규 시뮬 출력 포맷

#### IDF 수정 (modifier.py 업데이트)
추가 Output:Variable:
- `Zone Mean Air Temperature` (*, Hourly)
- `Zone Air Relative Humidity` (*, Hourly)
- `Zone Thermostat Cooling Setpoint Temperature` (*, Hourly)
- `Zone Thermostat Heating Setpoint Temperature` (*, Hourly)
- `Site Outdoor Air Relative Humidity` (*, Hourly)
- `Site Direct Solar Radiation Rate per Area` (*, Hourly)

#### 결과 저장 구조
```
simulations/results_v3_s{N}/     ← 원시 EnergyPlus CSV (기존 유지)
    {old_building_id}/
        eplusout.csv
        status.json

results/                         ← 표준 NPY 형식 (신규)
    KBB_B02_HA_C01_M00_VA_L001/
        metadata.json
        hourly_electricity.npy
        hourly_zone_temp.npy
        ... (Tier A 15개)

data/korean_bb/                  ← BB 평가용 Parquet (기존 유지)
    individual/
        KBB_B02_HA_C01_M00_VA_L001.parquet
```

### 5.5 단계별 BB CVRMSE 목표

| 단계 | 누적 건물 | 목표 Commercial CVRMSE |
|-----|:--------:|:---------------------:|
| 현재 s5 | 6,327 | 16.82% (s4_scratch) |
| s6~s8 (+3K) | ~9,327 | ~16.0% |
| s10~s11 (+4K) | ~13,327 | ~15.0% |
| s15 (10K office+retail) | ~17,327 | ~14.5% |
| s26 (전체 26,600) | 26,600 | **<14.0%** |

---

## 6. 코드 변경 목록

| 파일 | 변경 내용 | 우선순위 |
|------|---------|---------|
| `src/idf/modifier.py` | set_output_meters()에 6개 변수 추가 | **즉시** |
| `scripts/generate_parametric_idfs.py` | sim_id 생성 + 새 metadata.json + LHS 풀 관리 | **즉시** |
| `scripts/postprocess.py` | NPY 내보내기 추가 (Parquet 병행) | **즉시** |
| `scripts/migrate_existing_to_npy.py` | 기존 6,327건 → NPY 마이그레이션 | 필요시 |
| `cloud/orchestrate_batch.py` | 새 sim_id 형식 지원 | s6 전 |
| `data/korean_bb/` | building_id → sim_id 리네이밍 | 마이그레이션 시 |

---

## 7. 핵심 제약 사항

1. **BB 평가 파이프라인 변경 없음**: evaluate_bb.py + train.py는 그대로. Parquet 형식 유지.
2. **기존 6,327건 재시뮬 불필요**: Electricity:Facility는 완벽. BB 평가에 충분.
3. **Tier A 완전 준수는 s6+부터**: 기존 데이터는 부분 Tier A (metadata에 `partial_tier_a: true` 표시).
4. **LHS 풀 전략**: 아키타입별 글로벌 풀로 중복 방지.
5. **apartment_highrise**: B16 사용하되 metadata에 `is_korean_highrise: true` 플래그.

---

## 8. 결론

| 구분 | 결정 |
|------|------|
| 기존 6,327건 | **보존** — Parquet 형식 유지, 마이그레이션 스크립트로 선택적 NPY 변환 |
| 신규 s6+ 시뮬 | **새 표준 완전 준수** — Tier A 15변수 + sim_id + NPY + metadata.json |
| BB 평가 | **변경 없음** — 기존 파이프라인 그대로 사용 |
| 크로스 프로젝트 공유 | **s6+부터** — 새 표준 따르면 mcp_model/ems_transformer 재활용 가능 |
| 즉시 구현 항목 | modifier.py (6변수) + generate_parametric_idfs.py (sim_id) + postprocess.py (NPY) |
