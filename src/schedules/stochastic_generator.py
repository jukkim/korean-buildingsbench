"""
확률적 스케줄 생성기

OccupancyClass의 기본 패턴에 확률적 변동을 적용하여
건물별 고유한 8760시간 점유/조명/장비 스케줄을 생성.
320개 고정 스케줄 → 1,550+ 고유 스케줄로 다양성 확대.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta

from .occupancy_classes import OccupancyClass, get_occupancy_class


@dataclass
class StochasticParams:
    """확률적 변동 파라미터"""
    arrival_shift_h: float = 0.0     # 출근 시간 이동 (시간)
    departure_shift_h: float = 0.0   # 퇴근 시간 이동
    lunch_shift_h: float = 0.0       # 점심 시간 이동
    peak_occupancy_scale: float = 1.0 # 피크 점유율 배수
    weekend_scale: float = 1.0       # 주말 점유율 배수
    night_occupancy: float = 0.05    # 야간 최소 점유율
    noise_sigma: float = 0.02        # 시간별 랜덤 노이즈 σ

    # 계절 변동
    summer_factor: float = 1.0
    winter_factor: float = 1.0

    # Baseload 변동
    baseload_scale: float = 1.0


@dataclass
class ScheduleOutput:
    """생성된 8760시간 스케줄"""
    building_id: str
    class_id: str
    variant_id: int

    occupancy: np.ndarray   # (8760,) 점유율 0~1
    lighting: np.ndarray    # (8760,) W/m²
    equipment: np.ndarray   # (8760,) W/m²
    baseload: np.ndarray    # (8760,) W (총 상시부하)

    params: StochasticParams = field(default_factory=StochasticParams)

    def total_internal_gains(self, floor_area: float) -> np.ndarray:
        """총 내부 발열 (W) = (조명 + 장비) × 면적 + baseload"""
        return (self.lighting + self.equipment) * floor_area + self.baseload


class ScheduleGenerator:
    """확률적 스케줄 생성기

    Usage:
        gen = ScheduleGenerator(seed=42, year=2023)
        schedule = gen.generate('office_corporate', variant_id=0)
    """

    def __init__(self, seed: int = 42, year: int = 2023):
        self.rng = np.random.default_rng(seed)
        self.year = year
        self._build_calendar()

    def _build_calendar(self):
        """연간 달력 생성 (공휴일, 요일)"""
        start = date(self.year, 1, 1)
        self.n_days = 366 if self._is_leap(self.year) else 365
        self.n_hours = self.n_days * 24

        self.day_of_week = np.zeros(self.n_days, dtype=int)  # 0=Mon, 6=Sun
        self.month = np.zeros(self.n_days, dtype=int)
        self.day = np.zeros(self.n_days, dtype=int)
        self.is_holiday = np.zeros(self.n_days, dtype=bool)
        self.is_weekend = np.zeros(self.n_days, dtype=bool)

        for i in range(self.n_days):
            d = start + timedelta(days=i)
            self.day_of_week[i] = d.weekday()
            self.month[i] = d.month
            self.day[i] = d.day
            self.is_weekend[i] = d.weekday() >= 5

        # 고정 공휴일
        fixed_holidays = [
            (1, 1), (3, 1), (5, 5), (6, 6),
            (8, 15), (10, 3), (10, 9), (12, 25),
        ]
        for m, d in fixed_holidays:
            mask = (self.month == m) & (self.day == d)
            self.is_holiday[mask] = True

        # 음력 공휴일 (근사 - 연도별로 다름)
        lunar_holidays_2023 = [
            (1, 21), (1, 22), (1, 23),  # 설날
            (5, 27),                     # 부처님오신날
            (9, 28), (9, 29), (9, 30),  # 추석
        ]
        if self.year == 2023:
            for m, d in lunar_holidays_2023:
                mask = (self.month == m) & (self.day == d)
                self.is_holiday[mask] = True

        # 계절 구분 (월 기준)
        self.season = np.zeros(self.n_days, dtype=int)  # 0=겨울, 1=봄, 2=여름, 3=가을
        for i in range(self.n_days):
            m = self.month[i]
            if m in (12, 1, 2):
                self.season[i] = 0  # 겨울
            elif m in (3, 4, 5):
                self.season[i] = 1  # 봄
            elif m in (6, 7, 8):
                self.season[i] = 2  # 여름
            else:
                self.season[i] = 3  # 가을

    @staticmethod
    def _is_leap(year: int) -> bool:
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def _sample_params(self) -> StochasticParams:
        """확률적 파라미터 샘플링"""
        rng = self.rng
        return StochasticParams(
            arrival_shift_h=rng.normal(0, 0.5),
            departure_shift_h=rng.normal(0, 0.75),
            lunch_shift_h=rng.normal(0, 0.25),
            peak_occupancy_scale=np.clip(rng.normal(1.0, 0.1), 0.7, 1.3),
            weekend_scale=np.clip(rng.uniform(0.5, 1.5), 0.3, 2.0),
            night_occupancy=np.clip(rng.uniform(0.02, 0.10), 0.0, 0.15),
            noise_sigma=np.clip(rng.uniform(0.01, 0.05), 0.0, 0.1),
            summer_factor=np.clip(rng.normal(1.0, 0.1), 0.8, 1.2),
            winter_factor=np.clip(rng.normal(0.95, 0.1), 0.7, 1.1),
            baseload_scale=np.clip(rng.normal(1.0, 0.15), 0.7, 1.5),
        )

    def _shift_pattern(self, pattern: np.ndarray, shift_h: float) -> np.ndarray:
        """패턴을 시간 단위로 이동 (보간)"""
        if abs(shift_h) < 0.01:
            return pattern.copy()

        hours = np.arange(24)
        shifted = np.interp(
            hours - shift_h,
            hours,
            pattern,
            period=24,
        )
        return np.clip(shifted, 0.0, 1.0)

    def _apply_break(self, daily_occ: np.ndarray, oc: OccupancyClass) -> np.ndarray:
        """방학/휴무 기간 적용"""
        if not oc.has_break:
            return daily_occ

        result = daily_occ.copy()
        break_factor = 0.1  # 방학 중 10% 운영

        for brk in [oc.summer_break, oc.winter_break]:
            if brk is None:
                continue
            sm, sd = brk['start_month'], brk['start_day']
            em, ed = brk['end_month'], brk['end_day']

            for i in range(self.n_days):
                m, d = self.month[i], self.day[i]
                in_break = False
                if sm <= em:  # 같은 해 (여름방학)
                    in_break = (m > sm or (m == sm and d >= sd)) and \
                               (m < em or (m == em and d <= ed))
                else:  # 연말~연초 (겨울방학)
                    in_break = (m > sm or (m == sm and d >= sd)) or \
                               (m < em or (m == em and d <= ed))

                if in_break:
                    result[i] *= break_factor

        return result

    def generate(
        self,
        class_id: str,
        variant_id: int = 0,
        building_id: str = '',
        params: Optional[StochasticParams] = None,
    ) -> ScheduleOutput:
        """8760시간 스케줄 생성

        Args:
            class_id: 점유 클래스 ID
            variant_id: 변동 번호
            building_id: 건물 식별자
            params: 직접 지정 시 사용, None이면 랜덤 샘플링

        Returns:
            ScheduleOutput
        """
        oc = get_occupancy_class(class_id)

        if params is None:
            params = self._sample_params()

        # --- 1. 기본 패턴에 시간 이동 적용 ---
        weekday_base = self._shift_pattern(oc.weekday_pattern, params.arrival_shift_h)
        weekend_base = self._shift_pattern(oc.weekend_pattern, params.arrival_shift_h * 0.5)

        # 피크 점유율 스케일
        weekday_base *= params.peak_occupancy_scale
        weekend_base *= params.peak_occupancy_scale * params.weekend_scale

        # 금요일 패턴 (조기 퇴근)
        friday_base = weekday_base.copy()
        if oc.friday_early:
            friday_base[16:] *= 0.3  # 16시 이후 급감

        # 토요일 오전만
        saturday_base = weekend_base.copy()
        if oc.saturday_half:
            saturday_base[13:] *= 0.1  # 13시 이후 거의 없음

        # --- 2. 일별 점유율 (n_days,) ---
        daily_peak_occ = np.zeros(self.n_days)
        for i in range(self.n_days):
            dow = self.day_of_week[i]
            if self.is_holiday[i]:
                daily_peak_occ[i] = np.mean(weekend_base) * 0.5
            elif dow == 4:  # 금요일
                daily_peak_occ[i] = np.mean(friday_base)
            elif dow == 5:  # 토요일
                daily_peak_occ[i] = np.mean(saturday_base)
            elif dow == 6:  # 일요일
                daily_peak_occ[i] = np.mean(weekend_base)
            else:
                daily_peak_occ[i] = np.mean(weekday_base)

        # 방학 적용
        daily_peak_occ = self._apply_break(daily_peak_occ, oc)

        # --- 3. 8760시간 점유율 ---
        occupancy = np.zeros(self.n_hours)

        for i in range(self.n_days):
            h_start = i * 24
            h_end = h_start + 24
            dow = self.day_of_week[i]

            if self.is_holiday[i]:
                hourly = weekend_base * 0.5
            elif dow == 4:
                hourly = friday_base
            elif dow == 5:
                hourly = saturday_base
            elif dow == 6:
                hourly = weekend_base
            else:
                hourly = weekday_base

            # 계절 보정
            season = self.season[i]
            if season == 2:  # 여름
                hourly = hourly * params.summer_factor
            elif season == 0:  # 겨울
                hourly = hourly * params.winter_factor

            # 랜덤 노이즈
            noise = self.rng.normal(0, params.noise_sigma, 24)
            hourly = hourly + noise

            # 야간 최소값
            hourly = np.maximum(hourly, params.night_occupancy)

            occupancy[h_start:h_end] = np.clip(hourly, 0.0, 1.0)

        # 리조트 등 강한 계절 변동
        if oc.seasonal_variation:
            sv = oc.seasonal_variation
            for i in range(self.n_days):
                h_start = i * 24
                h_end = h_start + 24
                m = self.month[i]
                if m in (7, 8):
                    occupancy[h_start:h_end] *= sv.get('summer_peak', 1.0)
                elif m in (12, 1, 2):
                    occupancy[h_start:h_end] *= sv.get('winter_peak', 1.0)
                elif m in (4, 5, 10, 11):
                    occupancy[h_start:h_end] *= sv.get('shoulder', 1.0)
            occupancy = np.clip(occupancy, 0.0, 1.0)

        # --- 4. 조명 스케줄 (W/m²) ---
        if oc.lighting_coupled:
            lighting = occupancy * oc.lighting_density
        else:
            lighting = np.full(self.n_hours, oc.lighting_density * 0.5)

        # --- 5. 장비 스케줄 (W/m²) ---
        # 장비 = baseload_fraction + occupancy_fraction
        equip_baseload_frac = 0.3  # 장비의 30%는 상시 가동
        equipment = (
            oc.equipment_density * equip_baseload_frac +
            oc.equipment_density * (1 - equip_baseload_frac) * occupancy
        )

        # --- 6. Baseload (W) ---
        baseload_total = oc.baseload.total_w * params.baseload_scale
        baseload = np.full(self.n_hours, baseload_total)

        return ScheduleOutput(
            building_id=building_id,
            class_id=class_id,
            variant_id=variant_id,
            occupancy=occupancy,
            lighting=lighting,
            equipment=equipment,
            baseload=baseload,
            params=params,
        )

    def generate_batch(
        self,
        class_id: str,
        n_variants: int,
        building_prefix: str = '',
    ) -> List[ScheduleOutput]:
        """동일 클래스에서 n_variants개의 고유 스케줄 생성"""
        results = []
        for v in range(n_variants):
            bid = f"{building_prefix}_{class_id}_v{v:04d}" if building_prefix else f"{class_id}_v{v:04d}"
            schedule = self.generate(class_id, variant_id=v, building_id=bid)
            results.append(schedule)
        return results


if __name__ == '__main__':
    gen = ScheduleGenerator(seed=42, year=2023)

    print("=== 스케줄 생성 테스트 ===\n")

    # 사무실 3개 변동 비교
    for v in range(3):
        s = gen.generate('office_corporate', variant_id=v, building_id=f'test_office_{v}')
        mean_occ = np.mean(s.occupancy)
        peak_occ = np.max(s.occupancy)
        mean_base = np.mean(s.baseload)
        print(f"office_corporate v{v}: mean_occ={mean_occ:.3f}, peak={peak_occ:.3f}, "
              f"baseload={mean_base:.0f}W, arrival_shift={s.params.arrival_shift_h:+.2f}h")

    print()

    # 각 카테고리별 1개씩
    test_classes = [
        'residential_dual_income', 'office_corporate', 'school_secondary',
        'retail_mall', 'hospital_general', 'hotel_business',
    ]
    for cid in test_classes:
        s = gen.generate(cid, building_id=f'test_{cid}')
        daily_profile = s.occupancy[:168].reshape(7, 24)  # 첫 주
        print(f"{cid:<30}: weekly_mean={np.mean(daily_profile):.3f}, "
              f"weekday_mean={np.mean(daily_profile[:5]):.3f}, "
              f"weekend_mean={np.mean(daily_profile[5:]):.3f}")

    print(f"\n총 시간: {gen.n_hours}h ({gen.n_days}일)")
    print(f"공휴일: {np.sum(gen.is_holiday)}일")
    print(f"주말: {np.sum(gen.is_weekend)}일")
