"""
IDF 수정 엔진

ASHRAE ExampleFile (.idf) 을 기반으로:
1. RunPeriod → 전체 연간 (1/1 ~ 12/31)
2. 외피 U-value → 한국 vintage별 기준
3. 스케줄 → 확률적 생성 스케줄 주입 + People/Lights/Equipment 연결
4. 기밀성 (Infiltration) → AirChanges/Hour 상수 방식 (2026-03-29: Flow/ExteriorArea → ACH 교체)
5. 조명/장비 밀도 → 한국 기준
6. Output 변수 → BB 호환 (Electricity:Facility hourly)
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..buildings.envelope import EnvelopeSpec
from ..schedules.stochastic_generator import ScheduleOutput


class IDFModifier:
    """IDF 파일 수정기

    Usage:
        mod = IDFModifier(base_idf_path)
        mod.fix_run_period()
        mod.fix_simulation_control()
        mod.set_timestep(4)
        mod.modify_envelope(envelope_spec)
        mod.inject_occupancy_schedule(schedule)
        mod.set_lighting_density(12.0)
        mod.set_equipment_density(15.0)
        mod.add_baseload(1500)
        mod.set_output_meters()
        mod.save(output_path)
    """

    def __init__(self, idf_path: str):
        with open(idf_path, 'r', encoding='utf-8', errors='replace') as f:
            self.content = f.read()
        self.source_path = Path(idf_path)
        self._modifications = []

    # ============================================================
    # 1. RunPeriod
    # ============================================================

    def fix_run_period(self, start_month=1, start_day=1, end_month=12, end_day=31):
        """RunPeriod를 전체 연간으로 변경"""
        self.content = re.sub(
            r'RunPeriod,.*?;',
            '',
            self.content,
            flags=re.DOTALL,
        )

        run_period = f"""
RunPeriod,
    Annual,                  !- Name
    {start_month},           !- Begin Month
    {start_day},             !- Begin Day of Month
    ,                        !- Begin Year
    {end_month},             !- End Month
    {end_day},               !- End Day of Month
    ,                        !- End Year
    Sunday,                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays and Special Days
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes;                     !- Use Weather File Snow Indicators
"""
        self.content += run_period
        self._modifications.append('fix_run_period')
        return self

    def fix_simulation_control(self):
        """SimulationControl 교체: sizing + weather run 모두 활성화"""
        self.content = re.sub(
            r'SimulationControl,.*?;',
            """SimulationControl,
    Yes,                     !- Do Zone Sizing Calculation
    Yes,                     !- Do System Sizing Calculation
    Yes,                     !- Do Plant Sizing Calculation
    Yes,                     !- Run Simulation for Sizing Periods
    Yes,                     !- Run Simulation for Weather File Run Periods
    No,                      !- Do HVAC Sizing Simulation for Sizing Periods
    1;                       !- Maximum Number of HVAC Sizing Simulation Passes""",
            self.content,
            flags=re.DOTALL,
        )
        self._modifications.append('fix_simulation_control')
        return self

    # ============================================================
    # 2. 외피 U-value 수정
    # ============================================================

    def modify_envelope(self, spec: EnvelopeSpec):
        """외피 열성능 수정

        1. WindowMaterial:SimpleGlazingSystem → U-value, SHGC 직접 수정
        2. Material:NoMass → 벽/지붕 단열재 R-value 스케일링
        3. ZoneInfiltration → Flow/ExteriorArea 스케일링
        """
        self._modify_window_u_and_shgc(spec.window_u, spec.shgc)
        self._modify_opaque_insulation(spec)
        self._modify_infiltration(spec.infiltration_ach)

        self._modifications.append(f'envelope_{spec.vintage}_{spec.climate_zone}')
        return self

    def _modify_window_u_and_shgc(self, target_u: float, target_shgc: float):
        """WindowMaterial:SimpleGlazingSystem의 U-Factor와 SHGC 수정

        ASHRAE IDF 형식:
          WindowMaterial:SimpleGlazingSystem,
              Glazing Layer,           !- Name
              2.0441736,               !- U-Factor {W/m2-K}
              0.38,                    !- Solar Heat Gain Coefficient
        """
        # U-Factor: Name 다음 줄의 숫자값 교체
        self.content = re.sub(
            r'(WindowMaterial:SimpleGlazingSystem,\s*\n'
            r'\s*[^,]+,\s*!-\s*Name\s*\n'  # Name 줄 (주석 포함)
            r'\s*)'
            r'[\d.]+(\s*,\s*!-\s*U-Factor)',
            lambda m: f'{m.group(1)}{target_u}{m.group(2)}',
            self.content,
        )

        # SHGC: U-Factor 다음 줄
        self.content = re.sub(
            r'(WindowMaterial:SimpleGlazingSystem,\s*\n'
            r'\s*[^,]+,\s*!-\s*Name\s*\n'
            r'\s*[\d.]+\s*,\s*!-\s*U-Factor[^\n]*\n'  # U-Factor 줄
            r'\s*)'
            r'[\d.]+(\s*,?\s*!-\s*Solar Heat Gain Coefficient)',
            lambda m: f'{m.group(1)}{target_shgc}{m.group(2)}',
            self.content,
        )

    def _modify_opaque_insulation(self, spec: EnvelopeSpec):
        """Material:NoMass 단열재 R-value 스케일링

        ASHRAE IDF의 벽/지붕 단열재는 Material:NoMass 객체.
        목표 U-value에 맞게 R-value를 조정.

        ASHRAE 기준 U-value (Denver STD2019):
          벽: ~0.365 W/m2K (R=2.818)
          지붕: ~0.189 W/m2K (R=5.307)
        """
        # 벽 단열재: 이름에 'Wall' + 'Insulation' 포함
        self._scale_nomass_r_value(
            name_pattern=r'[Ww]all.*?[Ii]nsulation|[Ii]nsulation.*?[Ww]all',
            target_u=spec.wall_u,
            base_u=0.365,  # ASHRAE Denver STD2019 벽 기준
        )

        # 지붕 단열재: 이름에 'Roof' + 'Insulation' 포함
        self._scale_nomass_r_value(
            name_pattern=r'[Rr]oof.*?[Ii]nsulation|[Ii]nsulation.*?[Rr]oof',
            target_u=spec.roof_u,
            base_u=0.189,  # ASHRAE Denver STD2019 지붕 기준
        )

    def _scale_nomass_r_value(self, name_pattern: str, target_u: float, base_u: float):
        """Material:NoMass의 Thermal Resistance 스케일링

        R_new = R_original * (base_U / target_U)
        target_U가 높으면(오래된 건물) R이 작아지고, 낮으면(신축) R이 커짐
        """
        scale = base_u / max(target_u, 0.01)

        def replace_r_value(match):
            original_r = float(match.group(2))
            new_r = original_r * scale
            new_r = max(0.1, min(new_r, 50.0))  # 합리적 범위
            return f'{match.group(1)}{new_r:.4f}{match.group(3)}'

        # Material:NoMass 객체에서 이름 패턴 매칭 후 R-value 수정
        # 형식: Material:NoMass,\n  Name,\n  Roughness,\n  R-value, ...
        self.content = re.sub(
            r'(Material:NoMass,\s*\n'
            r'\s*' + name_pattern + r'[^;]*?'  # 이름 매칭
            r'!-\s*Thermal Resistance[^\n]*\n'  # Roughness 등 건너뛰기... 아님
            r')',
            lambda m: m.group(0),  # placeholder
            self.content,
            flags=re.DOTALL,
        )

        # 더 간단한 접근: Material:NoMass 블록을 파싱하여 수정
        blocks = list(re.finditer(
            r'(Material:NoMass,\s*\n\s*)([^,]+)(,.*?)([\d.]+)(\s*[,;]\s*!-\s*Thermal Resistance)',
            self.content,
            flags=re.DOTALL,
        ))
        for block in reversed(blocks):  # 역순으로 수정 (인덱스 보존)
            name = block.group(2).strip()
            if re.search(name_pattern, name):
                original_r = float(block.group(4))
                new_r = original_r * scale
                new_r = max(0.1, min(new_r, 50.0))
                start, end = block.start(4), block.end(4)
                self.content = self.content[:start] + f'{new_r:.4f}' + self.content[end:]

    def _modify_infiltration(self, target_ach: float):
        """ZoneInfiltration 수정 — AirChanges/Hour 상수 방식으로 교체

        기존 Flow/ExteriorArea 방식은 바람 속도에 의존하여
        실제 ACH가 목표값의 20-30%에 불과 (서울 평균 풍속 기준).
        단순 상수 ACH 방식으로 교체하여 정확한 기밀성 재현.

        예: 목표 0.5 ACH → 연중 모든 시간 0.5 ACH 적용
        """
        field_pat = re.compile(
            r'^\s*([^,;!\n]*?)\s*[,;]\s*!-.*$', re.MULTILINE
        )

        def rebuild_block(match):
            block = match.group(0)
            fields = [m.group(1).strip() for m in field_pat.finditer(block)]
            # fields[0] = Name, [1] = Zone, [2] = Schedule
            if len(fields) < 3:
                return block
            name = fields[0]
            zone = fields[1]
            sched = fields[2]
            return (
                f'  ZoneInfiltration:DesignFlowRate,\n'
                f'    {name},               !- Name\n'
                f'    {zone},               !- Zone or ZoneList or Space or SpaceList Name\n'
                f'    {sched},              !- Schedule Name\n'
                f'    AirChanges/Hour,      !- Design Flow Rate Calculation Method\n'
                f'    ,                     !- Design Flow Rate {{m3/s}}\n'
                f'    ,                     !- Flow per Zone Floor Area {{m3/s-m2}}\n'
                f'    ,                     !- Flow per Exterior Surface Area {{m3/s-m2}}\n'
                f'    {target_ach:.4f},     !- Air Changes per Hour {{1/hr}}\n'
                f'    1,                    !- Constant Term Coefficient\n'
                f'    0,                    !- Temperature Term Coefficient\n'
                f'    0,                    !- Velocity Term Coefficient\n'
                f'    0;                    !- Velocity Squared Term Coefficient\n'
            )

        self.content = re.sub(
            r'ZoneInfiltration:DesignFlowRate,.*?;',
            rebuild_block,
            self.content,
            flags=re.DOTALL,
        )
        self._modifications.append(f'infiltration_{target_ach}ACH_constant')

    # ============================================================
    # 3. 스케줄 주입 + People/Lights/Equipment 연결
    # ============================================================

    def inject_occupancy_schedule(self, schedule: ScheduleOutput, zone_name: str = ''):
        """점유/조명/장비 스케줄을 IDF에 주입하고 기존 객체에 연결

        1. Schedule:Compact 정의 추가
        2. People 객체의 스케줄 참조 → Korean_Occupancy_N
        3. Lights 객체의 스케줄 참조 → Korean_Lighting_N
        4. ElectricEquipment 객체의 스케줄 참조 → Korean_Equipment_N
        """
        occ_name = f'Korean_Occupancy_{schedule.variant_id}'
        light_name = f'Korean_Lighting_{schedule.variant_id}'
        equip_name = f'Korean_Equipment_{schedule.variant_id}'

        # Schedule:Compact 정의 추가
        occ_sched = self._array_to_schedule_compact(
            occ_name, schedule.occupancy, 'Fraction',
        )
        light_sched = self._array_to_schedule_compact(
            light_name,
            schedule.lighting / max(np.max(schedule.lighting), 1e-6),
            'Fraction',
        )
        equip_sched = self._array_to_schedule_compact(
            equip_name,
            schedule.equipment / max(np.max(schedule.equipment), 1e-6),
            'Fraction',
        )

        self.content += occ_sched + light_sched + equip_sched

        # People 객체의 스케줄 연결
        self._replace_schedule_in_objects(
            object_type='People',
            field_comment='Number of People Schedule Name',
            new_schedule=occ_name,
        )

        # Lights 객체의 스케줄 연결
        self._replace_schedule_in_objects(
            object_type='Lights',
            field_comment='Schedule Name',
            new_schedule=light_name,
        )

        # ElectricEquipment 객체의 스케줄 연결 (Korean_Baseload 제외)
        self._replace_schedule_in_objects(
            object_type='ElectricEquipment',
            field_comment='Schedule Name',
            new_schedule=equip_name,
            exclude_name='Korean_Baseload',
        )

        self._modifications.append(f'schedule_{schedule.class_id}_v{schedule.variant_id}')
        return self

    def _replace_schedule_in_objects(
        self,
        object_type: str,
        field_comment: str,
        new_schedule: str,
        exclude_name: str = '',
    ):
        """IDF 객체의 스케줄 필드를 교체

        People/Lights/ElectricEquipment의 Schedule Name 필드를 new_schedule로 교체.
        줄 단위로 파싱하여 안전하게 교체.
        """
        lines = self.content.split('\n')
        result = []
        in_object = False
        current_obj_name = ''
        field_idx = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 객체 시작 감지
            if stripped.startswith(f'{object_type},'):
                in_object = True
                field_idx = 0
                current_obj_name = ''
                result.append(line)
                continue

            if in_object:
                field_idx += 1

                # 첫 번째 필드 = Name
                if field_idx == 1:
                    current_obj_name = stripped.split(',')[0].split('!')[0].strip()

                # Schedule Name 필드 교체
                if f'!- {field_comment}' in line:
                    if not (exclude_name and exclude_name in current_obj_name):
                        # 기존 스케줄 이름을 new_schedule로 교체
                        parts = line.split('!-')
                        value_part = parts[0]
                        comment_part = '!-' + parts[1] if len(parts) > 1 else ''
                        # 값 부분에서 기존 스케줄 이름 교체
                        # 쉼표 앞의 값을 추출하여 교체
                        value_stripped = value_part.rstrip().rstrip(',')
                        indent = len(value_part) - len(value_part.lstrip())
                        line = f'{" " * indent}{new_schedule},      {comment_part}'

                # 세미콜론 = 객체 끝
                if ';' in stripped:
                    in_object = False

            result.append(line)

        self.content = '\n'.join(result)

    def _array_to_schedule_compact(
        self,
        name: str,
        hourly_values: np.ndarray,
        schedule_type: str = 'Fraction',
    ) -> str:
        """8760시간 배열 → Schedule:Compact IDF 문자열

        주중(Mon-Fri)/주말(Sat-Sun) 프로파일 분리.
        8760h 배열에서 요일별로 분류하여 각각의 평균 프로파일 생성.
        """
        lines = [
            f'\nSchedule:Compact,',
            f'    {name},                    !- Name',
            f'    {schedule_type},           !- Schedule Type Limits Name',
        ]

        n_hours = len(hourly_values)
        n_days = n_hours // 24

        # 요일 계산 (2023년 1월 1일 = 일요일 → day_of_week=6)
        # 0=Mon, 1=Tue, ..., 5=Sat, 6=Sun
        import datetime
        start_date = datetime.date(2023, 1, 1)
        start_dow = start_date.weekday()  # 0=Mon, 6=Sun

        day_idx = 0
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if n_days == 366:
            month_days[1] = 29

        for month_i, mdays in enumerate(month_days):
            month = month_i + 1
            end_day = mdays
            month_start = day_idx
            month_end = min(day_idx + mdays, n_days)

            lines.append(f'    Through: {month:02d}/{end_day:02d},')

            month_hours = hourly_values[month_start * 24:month_end * 24]
            if len(month_hours) == 0:
                break

            # 주중/주말 분리
            weekday_days = []
            weekend_days = []
            for d in range(month_start, month_end):
                dow = (start_dow + d) % 7
                day_hours = hourly_values[d * 24:(d + 1) * 24]
                if len(day_hours) == 24:
                    if dow < 5:  # Mon-Fri
                        weekday_days.append(day_hours)
                    else:  # Sat-Sun
                        weekend_days.append(day_hours)

            is_last_month = (month == 12)

            # Weekdays 프로파일
            if weekday_days:
                wd_profile = np.mean(weekday_days, axis=0)
            else:
                wd_profile = np.mean(month_hours[:mdays * 24].reshape(-1, 24), axis=0) if len(month_hours) >= 24 else month_hours[:24]

            lines.append(f'    For: Weekdays SummerDesignDay,')
            for h in range(24):
                val = float(wd_profile[h]) if h < len(wd_profile) else 0.0
                val = max(0.0, min(1.0, val))
                lines.append(f'    Until: {h + 1:02d}:00, {val:.4f},')

            # Weekend 프로파일
            if weekend_days:
                we_profile = np.mean(weekend_days, axis=0)
            else:
                we_profile = wd_profile * 0.3  # 주말 데이터 없으면 30%

            sep_final = ',' if not is_last_month else ''
            lines.append(f'    For: Weekends WinterDesignDay Holiday AllOtherDays,')
            for h in range(24):
                val = float(we_profile[h]) if h < len(we_profile) else 0.0
                val = max(0.0, min(1.0, val))
                if is_last_month and h == 23:
                    sep = ';'
                else:
                    sep = ','
                lines.append(f'    Until: {h + 1:02d}:00, {val:.4f}{sep}')

            day_idx = month_end

        return '\n'.join(lines) + '\n'

    # ============================================================
    # 4. 내부 부하 설정
    # ============================================================

    def set_lighting_density(self, density_w_m2: float):
        """조명 밀도 일괄 수정 (Watts/Area)

        ASHRAE IDF 형식:
          Lights,
              Name, Zone, Schedule,
              Watts/Area,              !- Design Level Calculation Method
              ,                        !- Lighting Level {W}
              6.888902667,             !- Watts per Zone Floor Area {W/m2}
        """
        self.content = re.sub(
            r'(Lights,.*?Watts/Area.*?\n'  # Method 줄
            r'\s*,\s*!-\s*Lighting Level[^\n]*\n'  # Lighting Level (빈 값) 건너뛰기
            r'\s*)'
            r'[\d.]+(\s*,\s*!-\s*Watts per Zone Floor Area)',
            lambda m: f'{m.group(1)}{density_w_m2}{m.group(2)}',
            self.content,
            flags=re.DOTALL,
        )
        self._modifications.append(f'lighting_{density_w_m2}')
        return self

    def set_equipment_density(self, density_w_m2: float):
        """장비 밀도 수정

        ASHRAE IDF는 두 가지 방식 사용:
        1. EquipmentLevel: 직접 W → 비율 스케일링
        2. Watts/Area: W/m2 → 직접 값 교체

        base_density ≈ 10.76 W/m2 (ASHRAE OfficeMedium 기준, EquipmentLevel용)
        """
        base_density = 10.76  # ASHRAE STD2019 OfficeMedium 기본값

        # --- 방식 1: EquipmentLevel (Design Level {W}) — 비율 스케일링 ---
        scale = density_w_m2 / base_density

        def replace_equip_level(match):
            original_w = float(match.group(1))
            new_w = original_w * scale
            return f'{new_w:.2f}{match.group(2)}'

        self.content = re.sub(
            r'([\d.]+(?:[eE][+-]?\d+)?)(\s*,\s*!-\s*Design Level \{W\})',
            replace_equip_level,
            self.content,
        )

        # --- 방식 2: Watts/Area (Watts per Zone Floor Area {W/m2}) — 직접 교체 ---
        # OfficeSmall 등 소형 건물에서 사용
        self.content = re.sub(
            r'(ElectricEquipment,.*?Watts/Area.*?\n'
            r'\s*,\s*!-\s*Design Level[^\n]*\n'
            r'\s*)'
            r'[\d.]+(\s*,\s*!-\s*Watts per Zone Floor Area)',
            lambda m: f'{m.group(1)}{density_w_m2}{m.group(2)}',
            self.content,
            flags=re.DOTALL,
        )

        self._modifications.append(f'equipment_{density_w_m2}')
        return self

    def add_baseload(self, baseload_w: float, zone_name: str = ''):
        """상시 가동 baseload 추가 (ElectricEquipment, Always On)"""
        if not zone_name:
            zones = self.get_zones()
            if not zones:
                return self
            zone_name = zones[0]

        schedule_name = self._find_always_on_schedule()

        baseload_obj = f"""
ElectricEquipment,
    Korean_Baseload,         !- Name
    {zone_name},             !- Zone or ZoneList or Space or SpaceList Name
    {schedule_name},         !- Schedule Name
    EquipmentLevel,          !- Design Level Calculation Method
    {baseload_w:.1f},        !- Design Level {{W}}
    ,                        !- Watts per Zone Floor Area {{W/m2}}
    ,                        !- Watts per Person {{W/person}}
    0.0,                     !- Fraction Latent
    0.3,                     !- Fraction Radiant
    0.0;                     !- Fraction Lost
"""
        self.content += baseload_obj
        self._modifications.append(f'baseload_{baseload_w:.0f}W')
        return self

    def _find_always_on_schedule(self) -> str:
        """IDF에서 always-on 스케줄 이름을 자동 감지"""
        candidates = ['ALWAYS_ON', 'Always On Discrete', 'Always On', 'ALWAYS ON']
        for name in candidates:
            if re.search(rf'Schedule:Compact,\s*\n\s*{re.escape(name)}\s*,', self.content, re.IGNORECASE):
                return name
        always_on = """
Schedule:Compact,
    Korean_AlwaysOn,         !- Name
    Fraction,                !- Schedule Type Limits Name
    Through: 12/31,          !- Field 1
    For: AllDays,            !- Field 2
    Until: 24:00, 1.0;       !- Field 3
"""
        self.content += always_on
        return 'Korean_AlwaysOn'

    # ============================================================
    # 5. 온도 설정 수정
    # ============================================================

    def modify_cooling_setpoint(self, occupied: float = 26.0, setback: float = 30.0):
        """냉방 설정온도 수정

        냉방: occupied(낮음=냉방작동) / setback(높음=냉방비작동)
        ASHRAE 기본: occupied~24, setback~26.7
        한국: occupied=26, setback=30
        """
        self._modify_setpoint_schedules(
            pattern_check=self._is_cooling_schedule,
            occupied_temp=occupied, setback_temp=setback,
            is_cooling=True,
        )
        self._modifications.append(f'cooling_{occupied}C')
        return self

    def modify_heating_setpoint(self, occupied: float = 22.0, setback: float = 15.0):
        """난방 설정온도 수정

        난방: occupied(높음=난방작동) / setback(낮음=난방비작동)
        """
        self._modify_setpoint_schedules(
            pattern_check=self._is_heating_schedule,
            occupied_temp=occupied, setback_temp=setback,
            is_cooling=False,
        )
        self._modifications.append(f'heating_{occupied}C')
        return self

    @staticmethod
    def _is_cooling_schedule(name: str) -> bool:
        n = name.lower()
        return ('cooling' in n and 'setpoint' in n) or 'clgsetp' in n or 'clgsp' in n

    @staticmethod
    def _is_heating_schedule(name: str) -> bool:
        n = name.lower()
        return ('heating' in n and 'setpoint' in n) or 'htgsetp' in n or 'htgsp' in n

    def _modify_setpoint_schedules(self, pattern_check, occupied_temp, setback_temp,
                                    is_cooling: bool = False) -> int:
        """Schedule:Compact 내 온도값 수정

        난방: 높은 원래값 → occupied (난방가동), 낮은 원래값 → setback (비가동)
        냉방: 낮은 원래값 → occupied (냉방가동), 높은 원래값 → setback (비가동)
        """
        lines = self.content.split('\n')
        result = []
        in_schedule = False
        count = 0

        # 원래 스케줄의 온도값들을 먼저 수집하여 중간값(threshold) 계산
        original_temps = []
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped.startswith('schedule:compact,'):
                if i + 1 < len(lines):
                    next_name = lines[i + 1].split('!')[0].strip().rstrip(',')
                    if pattern_check(next_name):
                        # 이 스케줄의 모든 온도값 수집
                        for j in range(i + 2, min(i + 300, len(lines))):
                            s = lines[j].strip().lower()
                            if s.startswith('until:'):
                                m = re.search(r'until:\s*\d{1,2}:\d{2}\s*,\s*([\d.]+)', s)
                                if m:
                                    original_temps.append(float(m.group(1)))
                            if ';' in lines[j]:
                                break

        if original_temps:
            threshold = (min(original_temps) + max(original_temps)) / 2.0
        else:
            threshold = (occupied_temp + setback_temp) / 2.0

        for i, line in enumerate(lines):
            stripped = line.strip().lower()

            if stripped.startswith('schedule:compact,'):
                if i + 1 < len(lines):
                    next_name = lines[i + 1].split('!')[0].strip().rstrip(',')
                    in_schedule = pattern_check(next_name)
                else:
                    in_schedule = False

            if in_schedule and ';' in line:
                if stripped.startswith('until:'):
                    line = self._replace_temp_value(
                        line, occupied_temp, setback_temp, threshold, is_cooling)
                    count += 1
                result.append(line)
                in_schedule = False
                continue

            if in_schedule and stripped.startswith('until:'):
                line = self._replace_temp_value(
                    line, occupied_temp, setback_temp, threshold, is_cooling)
                count += 1

            result.append(line)

        self.content = '\n'.join(result)
        return count

    @staticmethod
    def _replace_temp_value(line: str, occupied: float, setback: float,
                            threshold: float, is_cooling: bool = False) -> str:
        """Until: HH:MM, VALUE 에서 VALUE 교체

        난방: old > threshold → occupied (난방가동), old < threshold → setback
        냉방: old < threshold → occupied (냉방가동), old > threshold → setback
        """
        match = re.match(r'(\s*Until:\s*\d{1,2}:\d{2}\s*,\s*)([\d.]+)(.*)', line, re.IGNORECASE)
        if match:
            old_val = float(match.group(2))
            if is_cooling:
                # 냉방: 낮은값=재실(냉방가동), 높은값=비재실(setback)
                new_val = occupied if old_val <= threshold else setback
            else:
                # 난방: 높은값=재실(난방가동), 낮은값=비재실(setback)
                new_val = occupied if old_val >= threshold else setback
            return f'{match.group(1)}{new_val}{match.group(3)}'
        return line

    # ============================================================
    # 6. Output 설정 (BB 호환)
    # ============================================================

    def remove_transformer(self):
        """ElectricLoadCenter:Transformer 제거 (45kVA 과부하 방지)"""
        self.content = re.sub(
            r'ElectricLoadCenter:Transformer,.*?;',
            '',
            self.content,
            flags=re.DOTALL,
        )
        self._modifications.append('remove_transformer')
        return self

    def set_output_meters(self):
        """BB 호환 출력 미터 설정 + 8.simulation 공통 스키마 Tier A 변수

        Energy meters (9개):
          Electricity:Facility, Fans, Cooling, Heating (전기), Lights, Equipment,
          Heating:NaturalGas, Pumps, WaterSystems:NaturalGas

        Tier A Output:Variable (7개):
          Site Outdoor Air Drybulb Temperature  (외기온)
          Site Outdoor Air Relative Humidity    (외기습도) ← NEW
          Site Direct Solar Radiation Rate per Area (태양복사) ← NEW
          Zone Mean Air Temperature             (실내온도, * = 모든 zone) ← NEW
          Zone Air Relative Humidity            (실내습도, *) ← NEW
          Zone Thermostat Cooling Setpoint Temperature (*) ← NEW
          Zone Thermostat Heating Setpoint Temperature (*) ← NEW
        """
        # 기존 Output:Meter 제거 (Output:Meter + Output:Meter:MeterFileOnly)
        self.content = re.sub(
            r'^\s*Output:Meter(?::MeterFileOnly)?,\s*\n(?:.*\n)*?.*?;\s*$',
            '',
            self.content,
            flags=re.MULTILINE,
        )
        # 기존 Output:Variable 제거
        self.content = re.sub(
            r'^\s*Output:Variable,\s*\n(?:.*\n)*?.*?;\s*$',
            '',
            self.content,
            flags=re.MULTILINE,
        )

        meters = [
            'Electricity:Facility',
            'Fans:Electricity',
            'Cooling:Electricity',
            'Heating:Electricity',
            'InteriorLights:Electricity',
            'InteriorEquipment:Electricity',
            'Heating:NaturalGas',        # 가스 난방
            'Pumps:Electricity',         # 펌프
            'WaterSystems:NaturalGas',   # DHW 가스 (호텔/병원/아파트)
        ]

        meter_strs = []
        for m in meters:
            meter_strs.append(f"""
Output:Meter,
    {m},                     !- Key Name
    Hourly;                  !- Reporting Frequency
""")

        # Site-level 기상 변수 (Environment 키)
        site_variables = [
            'Site Outdoor Air Drybulb Temperature',
            'Site Outdoor Air Relative Humidity',
            'Site Direct Solar Radiation Rate per Area',
        ]
        for var in site_variables:
            meter_strs.append(f"""
Output:Variable,
    *,                       !- Key Value
    {var},                   !- Variable Name
    Hourly;                  !- Reporting Frequency
""")

        # Zone-level 변수 (모든 zone 수집 → postprocess에서 면적가중 평균)
        zone_variables = [
            'Zone Mean Air Temperature',
            'Zone Air Relative Humidity',
            'Zone Thermostat Cooling Setpoint Temperature',
            'Zone Thermostat Heating Setpoint Temperature',
        ]
        for var in zone_variables:
            meter_strs.append(f"""
Output:Variable,
    *,                       !- Key Value
    {var},                   !- Variable Name
    Hourly;                  !- Reporting Frequency
""")

        self.content += '\n'.join(meter_strs)
        self._modifications.append('output_meters_tier_a')
        return self

    def set_timestep(self, timesteps_per_hour: int = 4):
        """Timestep 설정 (기본 15분)"""
        self.content = re.sub(
            r'Timestep,\s*\d+\s*;',
            f'Timestep, {timesteps_per_hour};',
            self.content,
        )
        self._modifications.append(f'timestep_{timesteps_per_hour}')
        return self

    def set_warmup_days(self, max_days: int = 25):
        """Maximum Number of Warmup Days 설정"""
        self.content = re.sub(
            r'(\d+),(\s*!-\s*Maximum Number of Warmup Days)',
            f'{max_days},\\2',
            self.content,
        )
        self._modifications.append(f'warmup_days_{max_days}')
        return self

    def relax_convergence(self, temp_tol: float = 0.4, loads_tol: float = 0.04):
        """Building convergence tolerance 완화 (warmup 수렴 실패 방지)"""
        self.content = re.sub(
            r'([\d.]+),(\s*!-\s*Loads Convergence Tolerance Value)',
            f'{loads_tol:.4f},\\2',
            self.content,
        )
        self.content = re.sub(
            r'([\d.]+),(\s*!-\s*Temperature Convergence Tolerance Value)',
            f'{temp_tol:.4f},\\2',
            self.content,
        )
        # SolarDistribution을 FullExterior로 변경 (warmup 안정성 향상)
        self.content = re.sub(
            r'FullInteriorAndExterior,(\s*!-\s*Solar Distribution)',
            r'FullExterior,\1',
            self.content,
        )
        self._modifications.append(f'relax_convergence_t{temp_tol}_l{loads_tol}')
        return self

    # ============================================================
    # 7. 저장
    # ============================================================

    def save(self, output_path: str):
        """수정된 IDF 저장"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        header = f"! Korean_BB Modified IDF\n"
        header += f"! Source: {self.source_path.name}\n"
        header += f"! Modifications: {', '.join(self._modifications)}\n\n"

        with open(output, 'w', encoding='utf-8') as f:
            f.write(header + self.content)

        return output

    def get_zones(self) -> List[str]:
        """IDF에서 Zone 이름 목록 추출"""
        zones = []
        matches = re.findall(
            r'Zone,\s*\n?\s*([^,!]+)',
            self.content,
        )
        for m in matches:
            name = m.strip()
            if name and name.lower() not in ('', 'zone'):
                zones.append(name)
        return zones

    @property
    def modification_log(self) -> List[str]:
        return self._modifications.copy()
