"""
파라메트릭 스케줄 기반 IDF 생성 (v3 — 패턴 갭 수정 + 대규모)

12D Latin Hypercube Sampling으로 다양한 운영 패턴을 체계적으로 커버.
v2.1→v3 핵심 변경:
  - 8D→12D LHS (야간장비, 주간깨짐, 계절변동, 공정부하)
  - Fix A: 야간/주간 비율 수정 (night_equipment_frac)
  - Fix B: 주간 자기상관 수정 (weekly_break_prob, seasonal_amplitude)
  - Fix C: Baseload 분포 확대 (process_load_frac, baseload_w 상향)
  - Fix D: 매우 안정적인 건물 표현 (flat profile)
  - 다중 도시 지원 (5개 도시)
  - 5개 vintage 전체 지원

12D 파라미터:
  1. op_start (0~12h)              - 운영 시작 시간
  2. op_duration (8~24h)           - 운영 지속 시간 (24=24시간 운영)
  3. baseload_pct (25~98%)         - 비운영시 부하 (peak 대비)
  4. weekend_factor (0~1.2)        - 주말/주중 비율
  5. ramp_hours (0.5~4h)           - 전환 경사도
  6. equip_always_on (30~95%)      - 장비 상시가동 비율 (점유 무관)
  7. daily_noise_std (5~35%)       - 일간 스케일 노이즈 표준편차
  8. scale_mult (0.3~3.0)          - 부하 밀도 스케일 배수
  9. night_equipment_frac (30~95%) - 야간 장비 잔류 비율 (Fix A)
  10. weekly_break_prob (0~25%)    - 주별 패턴 깨짐 확률 (Fix B)
  11. seasonal_amplitude (0~30%)   - 계절별 부하 진폭 (Fix B)
  12. process_load_frac (0~50%)    - 상시 공정부하 비율 (Fix C)

Usage:
    # Office 2000건 (400 sched x 5 vintage x 1 city)
    python scripts/generate_parametric_idfs.py --version v3 --archetype office --n-schedules 400 --cities seoul

    # 전체 도시, 전체 vintage
    python scripts/generate_parametric_idfs.py --version v3 --n-schedules 400 --cities seoul,busan,daegu,gangneung,jeju --vintages all

    # 테스트 (100건)
    python scripts/generate_parametric_idfs.py --version v3 --archetype office --n-schedules 50 --cities seoul --vintages v1_pre1990,v5_2018_plus --dry-run

    # v2 호환 모드 (기존 동작)
    python scripts/generate_parametric_idfs.py --version v2 --archetype office
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import time
import yaml
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats.qmc import LatinHypercube

from src.buildings.archetypes import ARCHETYPES, get_archetype
from src.buildings.envelope import get_envelope
from src.schedules.stochastic_generator import ScheduleOutput, StochasticParams
from src.idf.modifier import IDFModifier


# ============================================================
# 설정
# ============================================================

PROJECT_DIR = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_DIR / 'configs'

# 도시별 기후대 + weather 파일 매핑
CITIES = {
    'seoul':     {'climate_zone': 'central_2', 'weather': 'KOR_Seoul.epw'},
    'busan':     {'climate_zone': 'southern',  'weather': 'KOR_Busan.epw'},
    'daegu':     {'climate_zone': 'southern',  'weather': 'KOR_Daegu.epw'},
    'gangneung': {'climate_zone': 'central_1', 'weather': 'KOR_Gangneung.epw'},
    'jeju':      {'climate_zone': 'jeju',      'weather': 'KOR_Jeju.epw'},
}

WEATHER_YEAR = 'tmy'

# Vintage 전체 목록
ALL_VINTAGES = ['v1_pre1990', 'v2_1991_2000', 'v3_2001_2010', 'v4_2011_2017', 'v5_2018_plus']

# ============================================================
# 8.simulation 표준 코드 매핑 (NAMING_UNIFICATION.md 기반)
# ============================================================

# Korean_BB 아키타입 → DOE 건물 코드 (B01~B16)
ARCHETYPE_TO_BLDG = {
    'office':             'B02',  # OfficeMedium
    'retail':             'B05',  # StandaloneRetail
    'school':             'B07',  # PrimarySchool
    'hotel':              'B15',  # LargeHotel
    'hospital':           'B12',  # Hospital
    'apartment_midrise':  'B16',  # MidRiseApartment
    'apartment_highrise': 'B16',  # DOE 비해당, B16 사용 (metadata에 is_korean_highrise=true)
    'small_office':       'B01',  # OfficeSmall
    'large_office':       'B03',  # OfficeLarge
    'warehouse':          'B14',  # Warehouse
    'restaurant_full':    'B10',  # RestaurantSitDown
    'restaurant_quick':   'B09',  # RestaurantFastFood
    'strip_mall':         'B04',  # RetailStripmall
    'university':         'B07',  # PrimarySchool (school IDF 기반, 스케줄/LHS로 차별화)
}

# Korean_BB 아키타입 → HVAC 코드 (H_A~H_E → HA~HE)
ARCHETYPE_TO_HVAC = {
    'office':             'HA',   # VAV+Chiller
    'retail':             'HC',   # Package/PSZ
    'school':             'HA',   # VAV+Chiller
    'hotel':              'HB',   # FCU+Chiller
    'hospital':           'HA',   # VAV+AHU+Chiller
    'apartment_midrise':  'HD',   # VRF+개별보일러
    'apartment_highrise': 'HA',   # 중앙냉방+가스보일러 (Korea 특유)
    'small_office':       'HC',   # Package/PSZ (소형)
    'large_office':       'HA',   # VAV+Chiller
    'warehouse':          'HC',   # Package/PSZ (소형)
    'restaurant_full':    'HC',   # Package/PSZ
    'restaurant_quick':   'HC',   # Package/PSZ
    'strip_mall':         'HC',   # Package/PSZ (소형)
    'university':         'HA',   # VAV+Chiller
}

# 도시 → 도시 코드 (C01~C10)
CITY_TO_CODE = {
    'seoul':     'C01',
    'busan':     'C02',
    'daegu':     'C03',
    'incheon':   'C04',
    'gwangju':   'C05',
    'daejeon':   'C06',
    'ulsan':     'C07',
    'gangneung': 'C08',
    'jeju':      'C09',
    'cheongju':  'C10',
}

# Vintage → VAR 코드 (VA~VE, Korean_BB 전용 확장)
VINTAGE_TO_VAR = {
    'v1_pre1990':    'VA',
    'v2_1991_2000':  'VB',
    'v3_2001_2010':  'VC',
    'v4_2011_2017':  'VD',
    'v5_2018_plus':  'VE',
}


def build_sim_id(archetype: str, vintage: str, city: str, lhs_idx: int) -> str:
    """8.simulation 표준 sim_id 생성

    형식: KBB_{BLDG}_{HVAC}_{CITY}_M00_{VAR}_L{IDX:03d}
    예시: KBB_B02_HA_C01_M00_VA_L001
    """
    bldg = ARCHETYPE_TO_BLDG.get(archetype, 'B01')
    hvac = ARCHETYPE_TO_HVAC.get(archetype, 'HA')
    city_code = CITY_TO_CODE.get(city, 'C01')
    var = VINTAGE_TO_VAR.get(vintage, 'VA')
    idx = lhs_idx + 1  # 1-based
    return f'KBB_{bldg}_{hvac}_{city_code}_M00_{var}_L{idx:03d}'


# ============================================================
# LHS 글로벌 풀 관리 (중복 방지)
# ============================================================

LHS_POOL_DIR = PROJECT_DIR / 'configs' / 'lhs_pool'


def get_lhs_pool(archetype: str, target_n: int, version: str = 'v3',
                 seed: int = 42) -> np.ndarray:
    """아키타입별 글로벌 LHS 풀 로드 또는 생성

    첫 호출 시 target_n 크기의 전체 LHS 풀을 생성하여 저장.
    이후 호출은 저장된 풀을 로드.
    """
    LHS_POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool_path = LHS_POOL_DIR / f'{archetype}_{version}.npz'

    bounds = get_effective_bounds(archetype, version)
    param_names = list(bounds.keys())

    if pool_path.exists():
        data = np.load(pool_path)
        pool = data['pool']
        if len(pool) < target_n:
            print(f"  [LHS Pool] 확장: {len(pool)} → {target_n}")
            sampler = LatinHypercube(d=len(param_names), seed=seed + 1)
            extra = sampler.random(n=target_n - len(pool))
            pool = np.vstack([pool, extra])
            np.savez(pool_path, pool=pool, param_names=param_names)
        return pool, param_names

    print(f"  [LHS Pool] 신규 생성: {archetype} × {target_n}점")
    sampler = LatinHypercube(d=len(param_names), seed=seed)
    raw = sampler.random(n=target_n)
    # 범위 스케일링
    pool = np.zeros_like(raw)
    for j, name in enumerate(param_names):
        lo, hi = bounds[name]
        pool[:, j] = lo + raw[:, j] * (hi - lo)
    np.savez(pool_path, pool=pool, param_names=param_names)
    return pool, param_names


def get_lhs_pool_state(archetype: str) -> dict:
    """LHS 풀 사용 상태 로드"""
    LHS_POOL_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LHS_POOL_DIR / f'{archetype}_state.json'
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return {'used_indices': [], 'next_batch_start': 0}


def save_lhs_pool_state(archetype: str, state: dict):
    """LHS 풀 사용 상태 저장"""
    LHS_POOL_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LHS_POOL_DIR / f'{archetype}_state.json'
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def get_next_lhs_batch(archetype: str, n: int, target_total: int = 10000,
                       version: str = 'v3', seed: int = 42) -> list:
    """다음 LHS 배치 가져오기 (중복 없음 보장)

    Returns: list of param dicts (schedule_id 포함)
    """
    pool, param_names = get_lhs_pool(archetype, target_total, version, seed)
    state = get_lhs_pool_state(archetype)
    start = state['next_batch_start']

    if start + n > len(pool):
        print(f"  [WARN] LHS 풀 소진: {archetype} (start={start}, n={n}, pool={len(pool)})")
        n = max(0, len(pool) - start)

    schedules = []
    for i in range(n):
        idx = start + i
        params = {name: float(pool[idx, j]) for j, name in enumerate(param_names)}
        params['schedule_id'] = idx
        schedules.append(params)

    # 상태 업데이트
    state['used_indices'].extend(range(start, start + n))
    state['next_batch_start'] = start + n
    save_lhs_pool_state(archetype, state)

    return schedules


# ============================================================
# v2 파라미터 (하위 호환)
# ============================================================
PARAM_BOUNDS_V2 = {
    'op_start':        (0.0,  12.0),
    'op_duration':     (8.0,  24.0),
    'baseload_pct':    (0.10, 0.95),
    'weekend_factor':  (0.0,  1.2),
    'ramp_hours':      (0.5,  4.0),
    'equip_always_on': (0.20, 0.85),
    'daily_noise_std': (0.05, 0.25),
    'scale_mult':      (0.3,  3.0),
}

# ============================================================
# v3 파라미터 (12D — 패턴 갭 수정)
# ============================================================
PARAM_BOUNDS_V3 = {
    # --- 기존 8D (범위 조정) ---
    'op_start':           (0.0,  12.0),
    'op_duration':        (8.0,  24.0),
    'baseload_pct':       (0.25, 0.98),    # UP: 최소 baseload 상향
    'weekend_factor':     (0.0,  1.2),
    'ramp_hours':         (0.5,  4.0),
    'equip_always_on':    (0.30, 0.95),    # UP: 0.20→0.30, 0.85→0.95
    'daily_noise_std':    (0.05, 0.35),    # UP: 0.25→0.35
    'scale_mult':         (0.3,  3.0),

    # --- 신규 4D (패턴 갭 수정) ---
    'night_equipment_frac': (0.3, 0.95),   # 야간 장비 잔류 비율 → night/day 수정
    'weekly_break_prob':    (0.05, 0.40),  # 주별 패턴 깨짐 확률 → autocorr 수정 (UP: E+열관성 보상)
    'seasonal_amplitude':   (0.05, 0.40),  # 계절별 부하 진폭 → autocorr 수정 (UP: E+열관성 보상)
    'process_load_frac':    (0.0, 0.5),    # 상시 공정부하 비율 → baseload 수정
}

# ============================================================
# 아키타입별 LHS 파라미터 오버라이드 (v3 전용)
# ============================================================
ARCHETYPE_PARAM_OVERRIDES = {
    # university: 스케줄 오버라이드 제거 — school IDF의 PSZ-AC와 충돌
    # 차별화는 ARCHETYPE_LOADS (낮은 baseload/equipment)와 SCALE_MULT_CAP으로 달성
}


def get_effective_bounds(archetype: str, version: str = 'v3') -> dict:
    """글로벌 bounds에 아키타입별 오버라이드를 적용"""
    bounds = dict(PARAM_BOUNDS_V3 if version == 'v3' else PARAM_BOUNDS_V2)
    if version == 'v3' and archetype in ARCHETYPE_PARAM_OVERRIDES:
        bounds.update(ARCHETYPE_PARAM_OVERRIDES[archetype])
    return bounds


# 아키타입별 조명/장비 밀도 (W/m2)
# v3: baseload_w 대폭 상향 (Fix C)
ARCHETYPE_LOADS_V3 = {
    'office':             {'lighting': 12.0, 'equipment': 15.0, 'baseload_w': 1500},
    'retail':             {'lighting': 15.0, 'equipment':  8.0, 'baseload_w': 1000},
    'school':             {'lighting': 10.0, 'equipment':  8.0, 'baseload_w':  500},
    'hotel':              {'lighting':  8.0, 'equipment':  5.0, 'baseload_w': 1200},
    'hospital':           {'lighting': 12.0, 'equipment': 20.0, 'baseload_w': 3000},
    'apartment_midrise':  {'lighting':  6.0, 'equipment':  3.0, 'baseload_w':  200},
    'apartment_highrise': {'lighting':  6.0, 'equipment':  3.0, 'baseload_w':  300},
    'small_office':       {'lighting':  9.0, 'equipment':  7.0, 'baseload_w':  200},
    'large_office':       {'lighting': 12.0, 'equipment': 18.0, 'baseload_w': 5000},
    'warehouse':          {'lighting':  5.0, 'equipment':  3.0, 'baseload_w':  300},
    'restaurant_full':    {'lighting': 15.0, 'equipment': 40.0, 'baseload_w': 2000},
    'restaurant_quick':   {'lighting': 15.0, 'equipment': 50.0, 'baseload_w': 1500},
    'strip_mall':         {'lighting': 14.0, 'equipment':  8.0, 'baseload_w':  400},
    'university':         {'lighting': 12.0, 'equipment': 12.0, 'baseload_w':  800},
}

# v2 baseload_w (하위 호환)
ARCHETYPE_LOADS_V2 = {
    'office':             {'lighting': 12.0, 'equipment': 15.0, 'baseload_w': 500},
    'retail':             {'lighting': 15.0, 'equipment':  8.0, 'baseload_w': 300},
    'school':             {'lighting': 10.0, 'equipment':  8.0, 'baseload_w': 200},
    'hotel':              {'lighting':  8.0, 'equipment':  5.0, 'baseload_w': 400},
    'hospital':           {'lighting': 12.0, 'equipment': 20.0, 'baseload_w': 800},
    'apartment_midrise':  {'lighting':  6.0, 'equipment':  3.0, 'baseload_w': 100},
    'apartment_highrise': {'lighting':  6.0, 'equipment':  3.0, 'baseload_w': 150},
    'small_office':       {'lighting':  9.0, 'equipment':  7.0, 'baseload_w': 100},
    'large_office':       {'lighting': 12.0, 'equipment': 18.0, 'baseload_w': 2000},
    'warehouse':          {'lighting':  5.0, 'equipment':  3.0, 'baseload_w': 150},
    'restaurant_full':    {'lighting': 15.0, 'equipment': 40.0, 'baseload_w': 800},
    'restaurant_quick':   {'lighting': 15.0, 'equipment': 50.0, 'baseload_w': 600},
    'strip_mall':         {'lighting': 14.0, 'equipment':  8.0, 'baseload_w': 200},
    'university':         {'lighting': 12.0, 'equipment': 12.0, 'baseload_w': 300},
}

# 아키타입별 scale_mult 상한 — 복잡 건물(small zone)의 열 발산 방지
SCALE_MULT_CAP = {
    'office':             3.0,
    'retail':             3.0,
    'school':             3.0,
    'hotel':              1.8,
    'hospital':           1.5,
    'apartment_midrise':  2.5,
    'apartment_highrise': 2.0,
    'small_office':       3.0,
    'large_office':       2.0,
    'warehouse':          3.0,
    'restaurant_full':    1.2,
    'restaurant_quick':   1.2,
    'strip_mall':         2.5,
    'university':         1.5,
}


# ============================================================
# LHS 스케줄 생성
# ============================================================

def generate_lhs_schedules(n_schedules: int, seed: int = 42,
                           version: str = 'v3', lhs_offset: int = 0,
                           archetype: str = None) -> list:
    """LHS로 스케줄 파라미터 생성 (v2: 8D, v3: 12D)

    lhs_offset > 0 이면 총 (n_schedules + lhs_offset) 샘플을 생성하고
    앞 lhs_offset개를 버려서 building_id가 p{lhs_offset:04d}부터 시작하도록 함.
    이를 통해 기존 카탈로그의 p-번호와 충돌 없이 새 빌딩을 추가할 수 있음.
    """
    bounds = get_effective_bounds(archetype, version) if archetype else (PARAM_BOUNDS_V3 if version == 'v3' else PARAM_BOUNDS_V2)
    param_names = list(bounds.keys())

    total = n_schedules + lhs_offset
    sampler = LatinHypercube(d=len(param_names), seed=seed)
    samples = sampler.random(n=total)

    schedules = []
    for i, s in enumerate(samples[lhs_offset:]):
        params = {}
        for j, name in enumerate(param_names):
            lo, hi = bounds[name]
            params[name] = lo + s[j] * (hi - lo)
        params['schedule_id'] = i + lhs_offset   # p-번호를 offset부터 시작
        schedules.append(params)

    return schedules


def params_to_8760_v2(params: dict, year: int = 2023) -> tuple:
    """v2 파라미터 → 8760시간 occupancy/lighting/equipment 배열 (하위 호환)"""
    op_start = params['op_start']
    op_duration = params['op_duration']
    baseload_pct = params['baseload_pct']
    weekend_factor = params['weekend_factor']
    ramp_hours = params['ramp_hours']
    equip_always_on = params['equip_always_on']
    daily_noise_std = params['daily_noise_std']

    op_end = op_start + op_duration

    # 24시간 weekday profile 생성
    weekday_profile = np.full(24, baseload_pct)
    for h in range(24):
        in_op = False
        if op_end <= 24:
            in_op = op_start <= h < op_end
        else:
            in_op = h >= op_start or h < (op_end - 24)

        if in_op:
            if op_end <= 24:
                hours_since_start = h - op_start
                hours_until_end = op_end - h - 1
            else:
                if h >= op_start:
                    hours_since_start = h - op_start
                else:
                    hours_since_start = (24 - op_start) + h
                hours_until_end = op_duration - hours_since_start - 1

            ramp_up = min(1.0, hours_since_start / max(ramp_hours, 0.5))
            ramp_down = min(1.0, hours_until_end / max(ramp_hours, 0.5))
            ramp = min(ramp_up, ramp_down)
            weekday_profile[h] = baseload_pct + (1.0 - baseload_pct) * ramp

    weekend_profile = baseload_pct + (weekday_profile - baseload_pct) * weekend_factor

    rng = np.random.default_rng(params['schedule_id'])

    n_days = 365
    daily_scale = rng.normal(1.0, daily_noise_std, size=n_days)
    daily_scale = np.clip(daily_scale, 0.3, 1.8)

    n_holidays = rng.integers(5, 16)
    holiday_days = set(rng.choice(n_days, size=n_holidays, replace=False).tolist())

    from datetime import datetime, timedelta
    start_date = datetime(year, 1, 1)
    occupancy = np.zeros(8760)

    for hour_idx in range(8760):
        dt = start_date + timedelta(hours=hour_idx)
        h = dt.hour
        weekday = dt.weekday()
        doy = dt.timetuple().tm_yday - 1

        if weekday < 5:
            base = weekday_profile[h]
        else:
            base = weekend_profile[h]

        if doy in holiday_days:
            base = baseload_pct

        base = baseload_pct + (base - baseload_pct) * daily_scale[doy]
        occupancy[hour_idx] = base

    hourly_noise = rng.normal(0, 0.03, size=8760)
    occupancy = np.clip(occupancy + hourly_noise, 0.01, 1.0)

    n_spikes = rng.integers(3, 11)
    for _ in range(n_spikes):
        spike_start = rng.integers(0, 8760 - 6)
        spike_dur = rng.integers(2, 7)
        spike_mag = rng.uniform(1.1, 1.5)
        end = min(spike_start + spike_dur, 8760)
        occupancy[spike_start:end] = np.clip(
            occupancy[spike_start:end] * spike_mag, 0.01, 1.0)

    equipment_frac = equip_always_on + (1.0 - equip_always_on) * occupancy
    equipment_frac = np.clip(equipment_frac, 0.01, 1.0)

    lighting_always_on = 0.05
    lighting_frac = lighting_always_on + (1.0 - lighting_always_on) * occupancy
    lighting_frac = np.clip(lighting_frac, 0.01, 1.0)

    return occupancy, lighting_frac, equipment_frac


def params_to_8760_v3(params: dict, year: int = 2023) -> tuple:
    """v3 12D 파라미터 → 8760시간 occupancy/lighting/equipment 배열

    패턴 갭 수정:
    - Fix A: night_equipment_frac → 야간 장비 잔류 비율 직접 제어
    - Fix B: weekly_break_prob + seasonal_amplitude → 주간 자기상관 감소
    - Fix C: process_load_frac → 상시 공정부하 추가 (baseload 상향)
    - Fix D: 고 baseload_pct + 고 equip_always_on → 거의 flat 프로파일 (자동)

    Returns:
        (occupancy, lighting_frac, equipment_frac) — 각 (8760,) 0~1
    """
    op_start = params['op_start']
    op_duration = params['op_duration']
    baseload_pct = params['baseload_pct']
    weekend_factor = params['weekend_factor']
    ramp_hours = params['ramp_hours']
    equip_always_on = params['equip_always_on']
    daily_noise_std = params['daily_noise_std']
    night_equipment_frac = params['night_equipment_frac']
    weekly_break_prob = params['weekly_break_prob']
    seasonal_amplitude = params['seasonal_amplitude']
    process_load_frac = params['process_load_frac']

    op_end = op_start + op_duration

    # ---- 24시간 weekday profile 생성 ----
    weekday_profile = np.full(24, baseload_pct)
    for h in range(24):
        in_op = False
        if op_end <= 24:
            in_op = op_start <= h < op_end
        else:
            in_op = h >= op_start or h < (op_end - 24)

        if in_op:
            if op_end <= 24:
                hours_since_start = h - op_start
                hours_until_end = op_end - h - 1
            else:
                if h >= op_start:
                    hours_since_start = h - op_start
                else:
                    hours_since_start = (24 - op_start) + h
                hours_until_end = op_duration - hours_since_start - 1

            ramp_up = min(1.0, hours_since_start / max(ramp_hours, 0.5))
            ramp_down = min(1.0, hours_until_end / max(ramp_hours, 0.5))
            ramp = min(ramp_up, ramp_down)
            weekday_profile[h] = baseload_pct + (1.0 - baseload_pct) * ramp

    # Weekend profile
    weekend_profile = baseload_pct + (weekday_profile - baseload_pct) * weekend_factor

    # RNG (schedule_id 기반 재현성)
    rng = np.random.default_rng(params['schedule_id'])

    # ---- Fix B: 계절 변동 (seasonal_amplitude) ----
    # monthly_factor = 1 + seasonal_amplitude * sin(2π * month/12 + phase)
    seasonal_phase = rng.uniform(0, 2 * np.pi)  # 건물별 랜덤 위상

    # ---- 일간 확률적 변동 ----
    n_days = 365
    daily_scale = rng.normal(1.0, daily_noise_std, size=n_days)
    daily_scale = np.clip(daily_scale, 0.3, 1.8)

    # ---- Fix B: 휴일 확대 — 10 + weekly_break_prob * 40 (10~20일) ----
    n_holidays = 10 + int(weekly_break_prob * 40)
    holiday_days = set(rng.choice(n_days, size=min(n_holidays, n_days), replace=False).tolist())

    # ---- Fix B: 주별 패턴 깨짐 이벤트 ----
    # 52주 × weekly_break_prob = 연 0~13주에서 패턴 변경
    n_weeks = 52
    weekly_events = {}  # week_idx → event_type
    for w in range(n_weeks):
        if rng.random() < weekly_break_prob:
            # 이벤트 유형: (1) 주중→주말 운영 20%
            #              (2) 2~5일 셧다운 30%
            #              (3) 운영시간 ±2h 이동 30%
            #              (4) 부하 0.5~0.8x 감축 20%
            roll = rng.random()
            if roll < 0.15:
                weekly_events[w] = ('weekend_ops', {})
            elif roll < 0.45:
                shutdown_days = int(rng.integers(2, 7))  # 2~6일 (UP: 5→6)
                start_dow = int(rng.integers(0, 7))
                weekly_events[w] = ('shutdown', {'n_days': shutdown_days, 'start_dow': start_dow})
            elif roll < 0.70:
                shift_hours = rng.uniform(-3, 3)  # ±3h (UP: ±2→±3)
                weekly_events[w] = ('time_shift', {'shift': shift_hours})
            else:
                reduction = rng.uniform(0.3, 0.75)  # 더 강한 감축 (UP: 0.5~0.8→0.3~0.75)
                weekly_events[w] = ('load_reduction', {'factor': reduction})

    # ---- 8760시간 배열 생성 ----
    from datetime import datetime, timedelta
    start_date = datetime(year, 1, 1)
    occupancy = np.zeros(8760)

    for hour_idx in range(8760):
        dt = start_date + timedelta(hours=hour_idx)
        h = dt.hour
        weekday = dt.weekday()  # 0=Mon, 6=Sun
        doy = dt.timetuple().tm_yday - 1  # 0-indexed
        month = dt.month
        week_idx = min(doy // 7, 51)

        # 기본 점유 프로파일
        if weekday < 5:
            base = weekday_profile[h]
        else:
            base = weekend_profile[h]

        # 휴일 → baseload만
        if doy in holiday_days:
            base = baseload_pct

        # ---- Fix B: 주별 이벤트 적용 ----
        if week_idx in weekly_events:
            etype, eparam = weekly_events[week_idx]
            if etype == 'weekend_ops':
                # 주중도 주말 프로파일로 운영
                base = weekend_profile[h]
            elif etype == 'shutdown':
                # 해당 요일에 셧다운
                shutdown_start = eparam['start_dow']
                shutdown_end = (shutdown_start + eparam['n_days']) % 7
                in_shutdown = False
                if shutdown_end > shutdown_start:
                    in_shutdown = shutdown_start <= weekday < shutdown_end
                else:
                    in_shutdown = weekday >= shutdown_start or weekday < shutdown_end
                if in_shutdown:
                    base = baseload_pct
            elif etype == 'time_shift':
                # 운영시간 이동 — shifted_profile 사용
                shift = eparam['shift']
                shifted_h = int(round(h - shift)) % 24
                if weekday < 5:
                    base = weekday_profile[shifted_h]
                else:
                    base = weekend_profile[shifted_h]
            elif etype == 'load_reduction':
                # 부하 감축
                factor = eparam['factor']
                base = baseload_pct + (base - baseload_pct) * factor

        # 일간 스케일링 (deviation from baseload 부분만 스케일)
        base = baseload_pct + (base - baseload_pct) * daily_scale[doy]

        # ---- Fix B: 계절 변동 적용 ----
        seasonal_factor = 1.0 + seasonal_amplitude * np.sin(
            2 * np.pi * month / 12.0 + seasonal_phase)
        base = baseload_pct + (base - baseload_pct) * seasonal_factor

        occupancy[hour_idx] = base

    # 시간별 미세 노이즈
    hourly_noise = rng.normal(0, 0.03, size=8760)
    occupancy = np.clip(occupancy + hourly_noise, 0.01, 1.0)

    # 돌발 부하 스파이크: 연 3~10회, 2~6시간
    n_spikes = rng.integers(3, 11)
    for _ in range(n_spikes):
        spike_start = rng.integers(0, 8760 - 6)
        spike_dur = rng.integers(2, 7)
        spike_mag = rng.uniform(1.1, 1.5)
        end = min(spike_start + spike_dur, 8760)
        occupancy[spike_start:end] = np.clip(
            occupancy[spike_start:end] * spike_mag, 0.01, 1.0)

    # ---- Fix A: 장비 — 야간 장비 잔류 비율 직접 제어 ----
    # 비운영 시간: max(equip_always_on, night_equipment_frac) 수준 유지
    # 운영 시간: equip_always_on + (1 - equip_always_on) * occupancy
    equipment_frac = np.zeros(8760)
    for hour_idx in range(8760):
        occ = occupancy[hour_idx]
        # occupancy가 baseload_pct에 가까우면 비운영 시간으로 간주
        if occ < baseload_pct + 0.15:
            # 비운영: 야간 장비 잔류 적용
            equipment_frac[hour_idx] = max(equip_always_on, night_equipment_frac)
        else:
            # 운영: 기존 방식
            equipment_frac[hour_idx] = equip_always_on + (1.0 - equip_always_on) * occ
    equipment_frac = np.clip(equipment_frac, 0.01, 1.0)

    # ---- 조명: 5% 상시(비상/보안등) + 95% 점유 비례 ----
    lighting_always_on = 0.05
    lighting_frac = lighting_always_on + (1.0 - lighting_always_on) * occupancy
    lighting_frac = np.clip(lighting_frac, 0.01, 1.0)

    return occupancy, lighting_frac, equipment_frac


def params_to_8760(params: dict, year: int = 2023, version: str = 'v3') -> tuple:
    """버전별 디스패치"""
    if version == 'v3':
        return params_to_8760_v3(params, year)
    return params_to_8760_v2(params, year)


# ============================================================
# IDF 생성
# ============================================================

def load_config():
    with open(CONFIGS_DIR / 'korean_buildings.yaml', 'r', encoding='utf-8') as f:
        buildings_cfg = yaml.safe_load(f)
    with open(CONFIGS_DIR / 'idf_mapping.yaml', 'r', encoding='utf-8') as f:
        mapping_cfg = yaml.safe_load(f)
    return buildings_cfg, mapping_cfg


def resolve_source_idf(archetype_id: str, mapping: dict) -> Path:
    ep_dir = Path(mapping['energyplus_dir'])
    example_dir = ep_dir / 'ExampleFiles'
    arch_map = mapping['archetype_mapping'].get(archetype_id, {})
    source_name = arch_map.get('source_idf', '')
    fallback_name = arch_map.get('fallback_idf', '')

    source_path = example_dir / source_name
    if source_path.exists():
        return source_path
    if fallback_name:
        fallback_path = example_dir / fallback_name
        if fallback_path.exists():
            return fallback_path
    raise FileNotFoundError(f"IDF not found: {source_path}")


def generate_single_parametric_idf(
    archetype_id: str,
    vintage: str,
    city: str,
    sched_params: dict,
    mapping: dict,
    output_dir: Path,
    dry_run: bool = False,
    version: str = 'v3',
) -> dict:
    """파라메트릭 스케줄로 단일 IDF 생성"""
    arch = get_archetype(archetype_id)
    city_info = CITIES.get(city, CITIES['seoul'])
    climate_zone = city_info['climate_zone']
    envelope = arch.get_envelope(vintage, climate_zone)

    # 버전별 부하 테이블
    loads_table = ARCHETYPE_LOADS_V3 if version == 'v3' else ARCHETYPE_LOADS_V2
    base_loads = loads_table[archetype_id]
    sid = sched_params['schedule_id']

    # scale_mult 적용 — 부하 밀도 다양성
    raw_scale = sched_params.get('scale_mult', 1.0)
    cap = SCALE_MULT_CAP.get(archetype_id, 3.0)
    scale = min(raw_scale, cap)

    # Fix C: process_load_frac 적용 — 상시 공정부하
    process_load_frac = sched_params.get('process_load_frac', 0.0) if version == 'v3' else 0.0

    loads = {
        'lighting': base_loads['lighting'] * scale,
        'equipment': base_loads['equipment'] * scale,
        'baseload_w': base_loads['baseload_w'] * scale,
    }

    building_id = f"{archetype_id}_{vintage}_{city}_{WEATHER_YEAR}_p{sid:04d}"

    # 8.simulation 표준 sim_id 생성
    sim_id = build_sim_id(archetype_id, vintage, city, sid)

    meta = {
        'building_id': building_id,    # 기존 형식 (BB 파이프라인 호환)
        'sim_id': sim_id,              # 8.simulation 표준 형식 (신규)
        'source_project': 'Korean_BB',
        'archetype': archetype_id,
        'archetype_kr': arch.name_kr,
        'vintage': vintage,
        'climate_zone': climate_zone,
        'city': city,
        'weather': CITIES.get(city, CITIES['seoul'])['weather'],  # EPW 파일명
        'weather_year': WEATHER_YEAR,
        'hvac': ARCHETYPE_TO_HVAC.get(archetype_id, 'HA'),
        'ems': 'M00',                  # Korean_BB는 EMS 없음 = Baseline
        'ems_variant': 'VA',
        'output_tier': 'A',
        'is_korean_highrise': archetype_id == 'apartment_highrise',
        'schedule_class': f'parametric_{sid:04d}',
        'variant_id': 0,
        'envelope': envelope.to_dict(),
        'category': arch.category,
        'parametric_params': {k: float(v) for k, v in sched_params.items()},
        'version': version,
        'lhs_index': sid,              # 글로벌 LHS 풀 인덱스
    }

    if dry_run:
        return meta

    # 8760 스케줄 생성
    occupancy, lighting_frac, equipment_frac = params_to_8760(sched_params, version=version)

    # Fix C: 공정부하를 baseload에 추가
    effective_baseload_w = (
        loads['baseload_w'] * sched_params['baseload_pct']
        + process_load_frac * loads['equipment'] * 10.0  # equipment W/m2 × 가정 면적 스케일
    )

    # ScheduleOutput 구성
    schedule = ScheduleOutput(
        building_id=building_id,
        class_id=f'parametric_{sid:04d}',
        variant_id=0,
        occupancy=occupancy.astype(np.float32),
        lighting=(lighting_frac * loads['lighting']).astype(np.float32),
        equipment=(equipment_frac * loads['equipment']).astype(np.float32),
        baseload=np.full(8760, effective_baseload_w, dtype=np.float32),
        params=StochasticParams(
            baseload_scale=sched_params['baseload_pct'],
            night_occupancy=sched_params['baseload_pct'],
        ),
    )

    # 소스 IDF
    source_idf = resolve_source_idf(archetype_id, mapping)

    # IDF 수정
    mod = IDFModifier(str(source_idf))
    mod.fix_run_period()
    mod.fix_simulation_control()
    mod.set_timestep(4)
    mod.set_warmup_days(25)
    mod.modify_envelope(envelope)
    mod.inject_occupancy_schedule(schedule)
    mod.set_lighting_density(loads['lighting'])
    mod.set_equipment_density(loads['equipment'])
    mod.add_baseload(effective_baseload_w)
    mod.modify_cooling_setpoint(occupied=26.0, setback=30.0)
    mod.modify_heating_setpoint(occupied=22.0, setback=15.0)
    mod.remove_transformer()
    mod.set_output_meters()

    # 수렴 안정성이 낮은 아키타입은 convergence tolerance 완화
    if archetype_id in ARCHETYPE_PARAM_OVERRIDES:
        mod.relax_convergence(temp_tol=0.4, loads_tol=0.04)

    # 저장
    idf_dir = output_dir / building_id
    idf_path = idf_dir / f'{building_id}.idf'
    mod.save(str(idf_path))

    meta['idf_path'] = str(idf_path)
    meta['modifications'] = mod.modification_log

    meta_path = idf_dir / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='파라메트릭 스케줄 기반 IDF 생성 (v2/v3)')
    parser.add_argument('--version', type=str, default='v3',
                        choices=['v2', 'v3'],
                        help='버전 (v2: 8D 기존, v3: 12D 패턴수정)')
    parser.add_argument('--archetype', type=str, default='',
                        help='단일 아키타입 (예: office). 미지정시 전체 7개')
    parser.add_argument('--n-schedules', type=int, default=150,
                        help='LHS 스케줄 수 (기본 150)')
    parser.add_argument('--seed', type=int, default=42,
                        help='LHS 시드')
    parser.add_argument('--lhs-offset', type=int, default=0,
                        help='LHS 인덱스 시작 오프셋 (기본 0). 기존 p-번호와 충돌 방지에 사용. '
                             '예: --lhs-offset 150 → p0150부터 시작 (0~149 skip)')
    parser.add_argument('--dry-run', action='store_true',
                        help='파일 생성 없이 매트릭스만 확인')
    parser.add_argument('--output-dir', type=str, default='',
                        help='출력 디렉토리')
    parser.add_argument('--cities', type=str, default='seoul',
                        help='쉼표 구분 도시 목록 (예: seoul,busan,daegu)')
    parser.add_argument('--vintages', type=str, default='',
                        help='쉼표 구분 vintage (예: v1_pre1990,v5_2018_plus). "all"=전체5개')
    parser.add_argument('--use-pool', action='store_true',
                        help='글로벌 LHS 풀 사용 (중복 방지). '
                             'configs/lhs_pool/에 상태 저장. s6+부터 권장.')
    parser.add_argument('--pool-target', type=int, default=10000,
                        help='LHS 풀 전체 크기 (기본 10000). use-pool 사용시만 의미있음.')
    args = parser.parse_args()

    version = args.version
    _, mapping_cfg = load_config()

    # 출력 디렉토리
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = 'idfs_v3' if version == 'v3' else 'idfs_v2'
        output_dir = PROJECT_DIR / 'simulations' / suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    # 대상 아키타입
    if args.archetype:
        archetypes = [args.archetype]
    else:
        archetypes = list(ARCHETYPES.keys())

    # 도시 목록
    cities = [c.strip() for c in args.cities.split(',')]
    for c in cities:
        if c not in CITIES:
            print(f"[WARN] Unknown city '{c}' — using seoul config as fallback")

    # Vintage 목록
    if args.vintages == 'all' or (version == 'v3' and not args.vintages):
        vintages = ALL_VINTAGES
    elif args.vintages:
        vintages = [v.strip() for v in args.vintages.split(',')]
    else:
        # v2 기본: 2개만
        vintages = ['v1_pre1990', 'v5_2018_plus']

    # LHS 스케줄 생성
    lhs_offset = args.lhs_offset
    if args.use_pool:
        # 글로벌 LHS 풀에서 아키타입별 순차 할당 (중복 방지)
        # 아키타입별로 별도 풀 사용
        per_arch_schedules = {}
        for arch_id in archetypes:
            per_arch_schedules[arch_id] = get_next_lhs_batch(
                arch_id, args.n_schedules,
                target_total=args.pool_target,
                version=version, seed=args.seed,
            )
            state = get_lhs_pool_state(arch_id)
            print(f"  [Pool] {arch_id}: idx {state['next_batch_start'] - args.n_schedules}"
                  f"~{state['next_batch_start'] - 1} ({args.n_schedules}개 할당)")
        schedules = None  # per_arch_schedules 사용
    else:
        # 아키타입별 오버라이드가 있으면 per-archetype 스케줄 생성
        has_overrides = any(a in ARCHETYPE_PARAM_OVERRIDES for a in archetypes)
        if has_overrides and version == 'v3':
            per_arch_schedules = {}
            for arch_id in archetypes:
                per_arch_schedules[arch_id] = generate_lhs_schedules(
                    args.n_schedules, seed=args.seed,
                    version=version, lhs_offset=lhs_offset,
                    archetype=arch_id,
                )
            schedules = None
        else:
            schedules = generate_lhs_schedules(args.n_schedules, seed=args.seed,
                                               version=version, lhs_offset=lhs_offset)
            per_arch_schedules = None
        if lhs_offset > 0:
            print(f"  LHS offset: {lhs_offset} (p{lhs_offset:04d}부터 시작)")

    # 매트릭스
    sample_schedules = schedules or per_arch_schedules.get(archetypes[0], [])
    total = len(archetypes) * len(vintages) * len(cities) * len(sample_schedules)
    param_bounds = PARAM_BOUNDS_V3 if version == 'v3' else PARAM_BOUNDS_V2
    param_names = list(param_bounds.keys())

    print("=" * 70)
    print(f"Korean_BB {version.upper()} — Parametric IDF Generation")
    print(f"  Archetypes:  {len(archetypes)} — {archetypes}")
    print(f"  Vintages:    {len(vintages)} — {vintages}")
    print(f"  Cities:      {len(cities)} — {cities}")
    print(f"  Schedules:   {args.n_schedules} ({len(param_names)}D LHS)")
    print(f"  Total IDFs:  {total:,}")
    print(f"  Output:      {output_dir}")
    print(f"  LHS mode:    {'pool (중복방지)' if args.use_pool else f'offset={lhs_offset}'}")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 70)

    # 스케줄 파라미터 범위 요약
    print(f"\n{len(param_names)}D Schedule Parameter Space:")
    for name in param_names:
        vals = [s[name] for s in sample_schedules]
        print(f"  {name:>22s}: [{min(vals):.3f}, {max(vals):.3f}] "
              f"(mean={np.mean(vals):.3f})")

    if args.dry_run:
        print(f"\nPer combo (archetype × vintage × city): {len(sample_schedules)} IDFs each")
        for a in archetypes:
            n = len(vintages) * len(cities) * len(sample_schedules)
            print(f"  {a:<22}: {n:>6,}")

        # 스케줄 파라미터를 JSON으로 저장 (참조용)
        params_path = output_dir / f'lhs_schedule_params_{version}.json'
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, 'w') as f:
            json.dump(sample_schedules, f, indent=2)
        print(f"\nSchedule params saved: {params_path}")
        return

    # 생성
    t0 = time.time()
    success = 0
    errors = []

    for arch_id in archetypes:
        arch_t0 = time.time()
        arch_ok = 0
        arch_fail = 0
        # 아키타입별 스케줄 선택
        arch_schedules = per_arch_schedules[arch_id] if per_arch_schedules else schedules

        for city in cities:
            for vintage in vintages:
                for sched in arch_schedules:
                    try:
                        generate_single_parametric_idf(
                            archetype_id=arch_id,
                            vintage=vintage,
                            city=city,
                            sched_params=sched,
                            mapping=mapping_cfg,
                            output_dir=output_dir,
                            version=version,
                        )
                        success += 1
                        arch_ok += 1
                    except Exception as e:
                        errors.append(
                            f"{arch_id}_{vintage}_{city}_p{sched['schedule_id']:04d}: {e}")
                        arch_fail += 1

        elapsed = time.time() - arch_t0
        print(f"  {arch_id:<22}: {arch_ok:>6,} OK, {arch_fail} FAIL ({elapsed:.1f}s)")

    total_elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Complete: {success:,} IDFs ({total_elapsed:.1f}s)")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")

    # 스케줄 파라미터 저장
    params_path = output_dir / f'lhs_schedule_params_{version}.json'
    save_scheds = schedules if schedules else {k: v for k, v in per_arch_schedules.items()}
    with open(params_path, 'w') as f:
        json.dump(save_scheds, f, indent=2)
    print(f"\nSchedule params: {params_path}")


if __name__ == '__main__':
    main()
