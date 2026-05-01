"""
점유 스케줄 클래스 정의 (22종)

각 클래스는 시간대별 기본 점유율 패턴을 정의.
StochasticGenerator에서 이를 기반으로 변동을 적용하여 고유 스케줄 생성.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BaseloadConfig:
    """24시간 상시 가동 baseload 구성 (W)"""
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total_w(self) -> float:
        return sum(self.components.values())


@dataclass
class OccupancyClass:
    """점유 스케줄 클래스"""
    class_id: str
    name: str
    name_kr: str
    category: str  # residential, office, school, retail, hospital, hotel

    # 시간별 점유율 (0~23시, 0.0~1.0)
    weekday_pattern: np.ndarray = field(default_factory=lambda: np.zeros(24))
    weekend_pattern: np.ndarray = field(default_factory=lambda: np.zeros(24))

    # 부하 밀도 (W/m²)
    lighting_density: float = 10.0
    equipment_density: float = 10.0

    # Baseload (24h 상시)
    baseload: BaseloadConfig = field(default_factory=BaseloadConfig)

    # 특수 속성
    lighting_coupled: bool = True   # 조명 = 점유율 연동
    friday_early: bool = False      # 금요일 조기 퇴근
    saturday_half: bool = False     # 토요일 오전만
    has_break: bool = False         # 방학/휴무 기간
    summer_break: Optional[Dict] = None
    winter_break: Optional[Dict] = None
    seasonal_variation: Optional[Dict] = None  # 계절별 점유율 배수
    heating_setpoint_offset: float = 0.0       # 기본 대비 난방 설정온도 오프셋

    def get_pattern(self, is_weekend: bool) -> np.ndarray:
        return self.weekend_pattern if is_weekend else self.weekday_pattern

    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'name': self.name,
            'category': self.category,
            'weekday_peak': float(np.max(self.weekday_pattern)),
            'weekend_peak': float(np.max(self.weekend_pattern)),
            'baseload_w': self.baseload.total_w,
        }


def _interpolate_pattern(sparse: Dict[int, float]) -> np.ndarray:
    """희소 시간-값 딕셔너리 → 24시간 보간 배열

    Args:
        sparse: {hour: occupancy_fraction} (0~23시)
                정의되지 않은 시간은 선형 보간

    Returns:
        np.ndarray of shape (24,)
    """
    hours = sorted(sparse.keys())
    values = [sparse[h] for h in hours]

    # 24시간 전체 보간
    result = np.zeros(24)
    for i in range(24):
        if i in sparse:
            result[i] = sparse[i]
        else:
            # 양쪽 가장 가까운 정의된 시간 찾기
            left_h = max((h for h in hours if h <= i), default=hours[-1])
            right_h = min((h for h in hours if h >= i), default=hours[0])

            if left_h == right_h:
                result[i] = sparse[left_h]
            else:
                # 선형 보간
                t = (i - left_h) / (right_h - left_h)
                result[i] = sparse[left_h] * (1 - t) + sparse[right_h] * t

    return np.clip(result, 0.0, 1.0)


# ============================================================
# 22개 스케줄 클래스 등록
# ============================================================

OCCUPANCY_CLASSES: Dict[str, OccupancyClass] = {}


def _reg(oc: OccupancyClass):
    OCCUPANCY_CLASSES[oc.class_id] = oc


# --- 주거 (4) ---

_reg(OccupancyClass(
    class_id='residential_dual_income',
    name='Dual-Income Household',
    name_kr='맞벌이 가구',
    category='residential',
    weekday_pattern=_interpolate_pattern({
        0: 1.0, 6: 0.9, 7: 0.5, 8: 0.1, 9: 0.05,
        12: 0.05, 17: 0.1, 18: 0.3, 19: 0.7, 20: 0.9, 22: 1.0, 23: 1.0,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 1.0, 8: 0.9, 10: 0.7, 12: 0.5, 14: 0.4,
        17: 0.6, 19: 0.8, 21: 1.0, 23: 1.0,
    }),
    lighting_density=8.0,
    equipment_density=3.0,
    lighting_coupled=True,
    baseload=BaseloadConfig({'refrigerator': 50, 'standby': 30, 'common_lighting': 15, 'elevator': 20}),
))

_reg(OccupancyClass(
    class_id='residential_single_income',
    name='Single-Income Household',
    name_kr='외벌이/주부 가구',
    category='residential',
    weekday_pattern=_interpolate_pattern({
        0: 1.0, 7: 0.8, 8: 0.5, 9: 0.5, 11: 0.4,
        13: 0.5, 15: 0.6, 17: 0.7, 19: 0.9, 21: 1.0, 23: 1.0,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 1.0, 9: 0.9, 11: 0.6, 14: 0.5,
        17: 0.7, 20: 0.9, 23: 1.0,
    }),
    lighting_density=8.0,
    equipment_density=3.5,
    lighting_coupled=True,
    baseload=BaseloadConfig({'refrigerator': 50, 'standby': 35, 'common_lighting': 15, 'elevator': 20}),
))

_reg(OccupancyClass(
    class_id='residential_single',
    name='Single-Person Household',
    name_kr='1인 가구',
    category='residential',
    weekday_pattern=_interpolate_pattern({
        0: 1.0, 7: 0.8, 8: 0.3, 9: 0.05, 12: 0.05,
        18: 0.1, 19: 0.3, 20: 0.7, 21: 0.9, 23: 1.0,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 1.0, 10: 0.8, 12: 0.6, 14: 0.4,
        18: 0.5, 20: 0.8, 23: 1.0,
    }),
    lighting_density=8.0,
    equipment_density=2.5,
    lighting_coupled=True,
    baseload=BaseloadConfig({'refrigerator': 50, 'standby': 25, 'common_lighting': 10, 'elevator': 15}),
))

_reg(OccupancyClass(
    class_id='residential_elderly',
    name='Elderly Household',
    name_kr='고령 가구',
    category='residential',
    weekday_pattern=_interpolate_pattern({
        0: 1.0, 5: 0.9, 6: 1.0, 9: 0.6, 11: 0.7,
        12: 0.9, 14: 0.7, 16: 0.8, 18: 1.0, 20: 1.0, 21: 1.0,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 1.0, 6: 1.0, 9: 0.7, 12: 0.9, 15: 0.7, 18: 1.0, 21: 1.0,
    }),
    lighting_density=8.0,
    equipment_density=2.0,
    lighting_coupled=True,
    heating_setpoint_offset=2.0,
    baseload=BaseloadConfig({'refrigerator': 50, 'standby': 20, 'common_lighting': 15, 'elevator': 20}),
))

# --- 사무실 (5) ---

_reg(OccupancyClass(
    class_id='office_government',
    name='Government Office',
    name_kr='정부/공공기관',
    category='office',
    weekday_pattern=_interpolate_pattern({
        0: 0.05, 7: 0.05, 8: 0.2, 9: 0.85, 12: 0.5,
        13: 0.85, 17: 0.6, 18: 0.15, 19: 0.05, 23: 0.05,
    }),
    weekend_pattern=_interpolate_pattern({0: 0.02, 12: 0.02, 23: 0.02}),
    lighting_density=12.0,
    equipment_density=15.0,
    friday_early=True,
    baseload=BaseloadConfig({
        'server': 300, 'standby': 150, 'security': 100,
        'common_lighting': 250, 'elevator': 200, 'ventilation': 100,
    }),
))

_reg(OccupancyClass(
    class_id='office_corporate',
    name='Corporate Office',
    name_kr='일반 대기업',
    category='office',
    weekday_pattern=_interpolate_pattern({
        0: 0.05, 7: 0.1, 8: 0.4, 9: 0.8, 12: 0.5,
        13: 0.8, 18: 0.5, 19: 0.3, 20: 0.2, 21: 0.1, 22: 0.05, 23: 0.05,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.02, 10: 0.1, 15: 0.08, 20: 0.02, 23: 0.02,
    }),
    lighting_density=14.0,
    equipment_density=20.0,
    baseload=BaseloadConfig({
        'server': 500, 'standby': 200, 'security': 100,
        'common_lighting': 300, 'elevator': 250, 'ventilation': 150,
    }),
))

_reg(OccupancyClass(
    class_id='office_it_startup',
    name='IT/Startup Office',
    name_kr='IT/스타트업',
    category='office',
    weekday_pattern=_interpolate_pattern({
        0: 0.05, 9: 0.1, 10: 0.5, 11: 0.7, 12: 0.5,
        13: 0.7, 18: 0.6, 19: 0.5, 20: 0.4, 21: 0.3, 22: 0.15, 23: 0.05,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.03, 13: 0.15, 18: 0.1, 23: 0.03,
    }),
    lighting_density=10.0,
    equipment_density=25.0,
    baseload=BaseloadConfig({
        'server': 800, 'standby': 300, 'security': 50,
        'common_lighting': 200, 'elevator': 150, 'ventilation': 120,
    }),
))

_reg(OccupancyClass(
    class_id='office_callcenter',
    name='Call Center',
    name_kr='콜센터/교대',
    category='office',
    weekday_pattern=_interpolate_pattern({
        0: 0.1, 6: 0.15, 8: 0.9, 12: 0.7, 13: 0.9,
        17: 0.5, 18: 0.6, 20: 0.5, 22: 0.2, 23: 0.1,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.08, 9: 0.5, 17: 0.3, 22: 0.08,
    }),
    lighting_density=14.0,
    equipment_density=18.0,
    baseload=BaseloadConfig({
        'server': 600, 'standby': 250, 'security': 80,
        'common_lighting': 300, 'elevator': 200, 'ventilation': 200,
    }),
))

_reg(OccupancyClass(
    class_id='office_small',
    name='Small Office',
    name_kr='소규모 사무실',
    category='office',
    weekday_pattern=_interpolate_pattern({
        0: 0.02, 8: 0.1, 9: 0.6, 12: 0.3, 13: 0.5,
        17: 0.3, 18: 0.1, 19: 0.02, 23: 0.02,
    }),
    weekend_pattern=_interpolate_pattern({0: 0.0, 12: 0.0, 23: 0.0}),
    lighting_density=10.0,
    equipment_density=12.0,
    baseload=BaseloadConfig({
        'server': 100, 'standby': 50, 'security': 30,
        'common_lighting': 80, 'elevator': 100, 'ventilation': 50,
    }),
))

# --- 학교 (3) ---

_reg(OccupancyClass(
    class_id='school_elementary',
    name='Elementary School',
    name_kr='초등학교',
    category='school',
    weekday_pattern=_interpolate_pattern({
        0: 0.02, 7: 0.1, 8: 0.7, 9: 0.9, 12: 0.7,
        13: 0.85, 14: 0.5, 15: 0.2, 16: 0.05, 17: 0.02, 23: 0.02,
    }),
    weekend_pattern=_interpolate_pattern({0: 0.02, 12: 0.02, 23: 0.02}),
    lighting_density=10.0,
    equipment_density=8.0,
    has_break=True,
    summer_break={'start_month': 7, 'start_day': 20, 'end_month': 8, 'end_day': 31},
    winter_break={'start_month': 12, 'start_day': 25, 'end_month': 2, 'end_day': 28},
    baseload=BaseloadConfig({'security': 50, 'common_lighting': 100, 'ventilation': 80}),
))

_reg(OccupancyClass(
    class_id='school_secondary',
    name='Secondary School',
    name_kr='중고등학교',
    category='school',
    weekday_pattern=_interpolate_pattern({
        0: 0.02, 7: 0.3, 8: 0.9, 12: 0.7, 13: 0.9,
        15: 0.7, 16: 0.5, 17: 0.4, 18: 0.3, 21: 0.15, 22: 0.05, 23: 0.02,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.02, 9: 0.2, 17: 0.1, 22: 0.02,
    }),
    lighting_density=12.0,
    equipment_density=10.0,
    has_break=True,
    summer_break={'start_month': 7, 'start_day': 20, 'end_month': 8, 'end_day': 25},
    winter_break={'start_month': 12, 'start_day': 25, 'end_month': 2, 'end_day': 20},
    baseload=BaseloadConfig({'security': 80, 'common_lighting': 150, 'ventilation': 100}),
))

_reg(OccupancyClass(
    class_id='school_university',
    name='University',
    name_kr='대학교',
    category='school',
    weekday_pattern=_interpolate_pattern({
        0: 0.05, 8: 0.2, 9: 0.5, 10: 0.7, 12: 0.5,
        13: 0.7, 15: 0.6, 17: 0.4, 18: 0.3, 20: 0.2, 22: 0.1, 23: 0.05,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.03, 10: 0.15, 14: 0.2, 18: 0.15, 22: 0.05,
    }),
    lighting_density=12.0,
    equipment_density=15.0,
    has_break=True,
    summer_break={'start_month': 6, 'start_day': 20, 'end_month': 8, 'end_day': 31},
    winter_break={'start_month': 12, 'start_day': 20, 'end_month': 2, 'end_day': 28},
    baseload=BaseloadConfig({'server': 500, 'security': 100, 'common_lighting': 200, 'ventilation': 150}),
))

# --- 판매 (4) ---

_reg(OccupancyClass(
    class_id='retail_mall',
    name='Shopping Mall',
    name_kr='대형 쇼핑몰',
    category='retail',
    weekday_pattern=_interpolate_pattern({
        0: 0.05, 9: 0.1, 10: 0.4, 12: 0.7, 14: 0.8,
        17: 0.9, 19: 0.85, 21: 0.5, 22: 0.2, 23: 0.05,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.05, 10: 0.5, 12: 0.85, 14: 0.95,
        17: 0.9, 20: 0.6, 22: 0.15, 23: 0.05,
    }),
    lighting_density=18.0,
    equipment_density=10.0,
    baseload=BaseloadConfig({
        'security': 200, 'common_lighting': 500, 'elevator_escalator': 400,
        'ventilation': 300, 'refrigeration': 800,
    }),
))

_reg(OccupancyClass(
    class_id='retail_department',
    name='Department Store',
    name_kr='백화점',
    category='retail',
    weekday_pattern=_interpolate_pattern({
        0: 0.05, 10: 0.3, 11: 0.6, 12: 0.7, 14: 0.75,
        17: 0.8, 19: 0.6, 20: 0.3, 21: 0.1, 23: 0.05,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.05, 10: 0.5, 12: 0.8, 14: 0.9,
        17: 0.85, 20: 0.3, 23: 0.05,
    }),
    lighting_density=20.0,
    equipment_density=8.0,
    baseload=BaseloadConfig({
        'security': 150, 'common_lighting': 400, 'elevator_escalator': 500, 'ventilation': 250,
    }),
))

_reg(OccupancyClass(
    class_id='retail_convenience',
    name='Convenience Store 24h',
    name_kr='편의점/24h',
    category='retail',
    weekday_pattern=_interpolate_pattern({
        0: 0.3, 3: 0.15, 6: 0.2, 8: 0.5, 12: 0.7,
        18: 0.6, 21: 0.5, 23: 0.35,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.35, 3: 0.15, 10: 0.5, 14: 0.6,
        18: 0.65, 22: 0.4, 23: 0.35,
    }),
    lighting_density=16.0,
    equipment_density=12.0,
    baseload=BaseloadConfig({
        'refrigeration': 1200, 'signage': 100, 'security': 30, 'ventilation': 80,
    }),
))

_reg(OccupancyClass(
    class_id='retail_neighborhood',
    name='Neighborhood Shop',
    name_kr='근린상가',
    category='retail',
    weekday_pattern=_interpolate_pattern({
        0: 0.02, 9: 0.1, 10: 0.5, 12: 0.6, 14: 0.5,
        17: 0.4, 19: 0.2, 20: 0.05, 23: 0.02,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.02, 10: 0.3, 13: 0.4, 17: 0.3, 19: 0.1, 23: 0.02,
    }),
    lighting_density=12.0,
    equipment_density=8.0,
    baseload=BaseloadConfig({'security': 30, 'common_lighting': 50, 'ventilation': 40}),
))

# --- 병원 (3) ---

_reg(OccupancyClass(
    class_id='hospital_general',
    name='General Hospital',
    name_kr='종합병원',
    category='hospital',
    weekday_pattern=_interpolate_pattern({
        0: 0.3, 6: 0.4, 7: 0.6, 8: 0.85, 12: 0.7,
        13: 0.85, 17: 0.6, 18: 0.45, 20: 0.35, 22: 0.3,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.25, 8: 0.4, 12: 0.35, 18: 0.3, 22: 0.25,
    }),
    lighting_density=16.0,
    equipment_density=25.0,
    baseload=BaseloadConfig({
        'medical_equipment': 2000, 'server': 500, 'security': 200,
        'common_lighting': 500, 'elevator': 400, 'ventilation': 500,
    }),
))

_reg(OccupancyClass(
    class_id='hospital_clinic',
    name='Clinic',
    name_kr='의원/클리닉',
    category='hospital',
    weekday_pattern=_interpolate_pattern({
        0: 0.02, 8: 0.1, 9: 0.8, 12: 0.3, 13: 0.3,
        14: 0.8, 17: 0.5, 18: 0.1, 19: 0.02, 23: 0.02,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.02, 9: 0.5, 12: 0.1, 23: 0.02,
    }),
    lighting_density=14.0,
    equipment_density=15.0,
    saturday_half=True,
    baseload=BaseloadConfig({
        'medical_equipment': 300, 'security': 30, 'common_lighting': 80, 'ventilation': 100,
    }),
))

_reg(OccupancyClass(
    class_id='hospital_nursing',
    name='Nursing Home',
    name_kr='요양원',
    category='hospital',
    weekday_pattern=_interpolate_pattern({
        0: 0.6, 6: 0.7, 8: 0.8, 12: 0.85, 14: 0.7,
        17: 0.8, 20: 0.7, 22: 0.6,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.6, 8: 0.75, 14: 0.65, 20: 0.65, 23: 0.6,
    }),
    lighting_density=10.0,
    equipment_density=8.0,
    baseload=BaseloadConfig({
        'medical_equipment': 200, 'security': 50, 'common_lighting': 200,
        'elevator': 150, 'ventilation': 150,
    }),
))

# --- 호텔 (3) ---

_reg(OccupancyClass(
    class_id='hotel_business',
    name='Business Hotel',
    name_kr='비즈니스호텔',
    category='hotel',
    weekday_pattern=_interpolate_pattern({
        0: 0.7, 6: 0.6, 8: 0.3, 10: 0.2, 12: 0.15,
        15: 0.25, 18: 0.5, 20: 0.65, 22: 0.7,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.4, 8: 0.3, 12: 0.15, 18: 0.3, 22: 0.4,
    }),
    lighting_density=12.0,
    equipment_density=8.0,
    baseload=BaseloadConfig({
        'lobby_lighting': 300, 'elevator': 200, 'security': 100,
        'ventilation': 200, 'laundry': 400,
    }),
))

_reg(OccupancyClass(
    class_id='hotel_tourist',
    name='Tourist Hotel',
    name_kr='관광호텔',
    category='hotel',
    weekday_pattern=_interpolate_pattern({
        0: 0.5, 8: 0.4, 10: 0.2, 12: 0.15, 15: 0.2,
        18: 0.45, 20: 0.6, 22: 0.55,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.75, 8: 0.6, 10: 0.3, 14: 0.25,
        18: 0.6, 22: 0.7,
    }),
    lighting_density=14.0,
    equipment_density=10.0,
    baseload=BaseloadConfig({
        'lobby_lighting': 500, 'elevator': 300, 'security': 120,
        'ventilation': 250, 'laundry': 600, 'pool_spa': 800,
    }),
))

_reg(OccupancyClass(
    class_id='hotel_resort',
    name='Resort',
    name_kr='리조트',
    category='hotel',
    weekday_pattern=_interpolate_pattern({
        0: 0.3, 8: 0.25, 10: 0.15, 15: 0.2, 18: 0.3, 22: 0.3,
    }),
    weekend_pattern=_interpolate_pattern({
        0: 0.6, 8: 0.5, 10: 0.3, 15: 0.35, 18: 0.55, 22: 0.6,
    }),
    lighting_density=12.0,
    equipment_density=8.0,
    seasonal_variation={'summer_peak': 1.8, 'winter_peak': 1.5, 'shoulder': 0.6},
    baseload=BaseloadConfig({
        'lobby_lighting': 400, 'elevator': 200, 'security': 80,
        'ventilation': 200, 'pool_spa': 1200,
    }),
))


# ============================================================
# 조회 함수
# ============================================================

def get_occupancy_class(class_id: str) -> OccupancyClass:
    if class_id not in OCCUPANCY_CLASSES:
        raise ValueError(f"Unknown class: {class_id}. "
                         f"Available: {list(OCCUPANCY_CLASSES.keys())}")
    return OCCUPANCY_CLASSES[class_id]


def list_occupancy_classes() -> list:
    return list(OCCUPANCY_CLASSES.keys())


def list_by_category(category: str) -> list:
    return [oc for oc in OCCUPANCY_CLASSES.values() if oc.category == category]


if __name__ == '__main__':
    print("=== 점유 스케줄 클래스 (22종) ===\n")
    print(f"{'ID':<30} {'이름':<14} {'카테고리':<12} {'주중피크':>8} {'주말피크':>8} {'Baseload(W)':>11}")
    print("-" * 90)
    for cid, oc in OCCUPANCY_CLASSES.items():
        print(f"{cid:<30} {oc.name_kr:<14} {oc.category:<12} "
              f"{np.max(oc.weekday_pattern):8.2f} {np.max(oc.weekend_pattern):8.2f} "
              f"{oc.baseload.total_w:11.0f}")
    print(f"\n총 클래스: {len(OCCUPANCY_CLASSES)}")
