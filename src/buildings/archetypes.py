"""
한국 건물 아키타입 정의 (13종)

각 아키타입은 건물 유형, 층수, 연면적, HVAC, WWR 등을 정의.
Vintage(연식)와 결합하여 시뮬레이션 케이스를 생성.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from .envelope import get_envelope, EnvelopeSpec, VINTAGES, CLIMATE_ZONES


@dataclass
class HVACConfig:
    """HVAC 시스템 설정"""
    heating_type: str       # ondol_gas_boiler, district_heating, chiller_boiler, vrf
    cooling_type: str       # wall_split, system_ac, vrf, chiller_ahu, chiller_ahu_fcu
    heating_cop: float = 0.9   # 보일러 효율 또는 COP
    cooling_cop: float = 3.5
    has_ventilation: bool = False
    ventilation_type: str = 'natural'  # natural, exhaust, hrv


@dataclass
class BuildingArchetype:
    """건물 아키타입 정의"""
    archetype_id: str
    name: str
    name_kr: str
    category: str  # residential, office, school, retail, hospital, hotel

    # 기하
    floors_options: List[int] = field(default_factory=list)
    gfa_range: Tuple[float, float] = (0, 0)  # m²
    wwr: Dict[str, float] = field(default_factory=dict)  # south, north, east, west
    aspect_ratio: float = 1.5  # 장변/단변

    # HVAC (vintage별로 다를 수 있지만 기본값)
    hvac: HVACConfig = field(default_factory=lambda: HVACConfig('ondol_gas_boiler', 'wall_split'))

    # 스케줄
    occupancy_classes: List[str] = field(default_factory=list)
    schedule_variants_per_class: int = 50

    # 내부 부하 (W/m²)
    lighting_density: float = 10.0
    equipment_density: float = 10.0
    people_density: float = 0.05  # people/m²

    # 시뮬레이션 비중
    share: float = 0.0

    def get_envelope(self, vintage: str, climate_zone: str) -> EnvelopeSpec:
        """이 아키타입의 외피 사양 조회"""
        cat = 'residential' if self.category == 'residential' else 'non_residential'
        return get_envelope(vintage, climate_zone, cat)

    def total_schedule_variants(self) -> int:
        return len(self.occupancy_classes) * self.schedule_variants_per_class

    def to_dict(self) -> dict:
        return {
            'archetype_id': self.archetype_id,
            'name': self.name,
            'name_kr': self.name_kr,
            'category': self.category,
            'floors_options': self.floors_options,
            'gfa_range': list(self.gfa_range),
            'wwr': self.wwr,
            'hvac_heating': self.hvac.heating_type,
            'hvac_cooling': self.hvac.cooling_type,
            'occupancy_classes': self.occupancy_classes,
            'schedule_variants_per_class': self.schedule_variants_per_class,
            'share': self.share,
        }


# ============================================================
# 13개 아키타입 정의
# ============================================================

ARCHETYPES: Dict[str, BuildingArchetype] = {}


def _register(a: BuildingArchetype):
    ARCHETYPES[a.archetype_id] = a


_register(BuildingArchetype(
    archetype_id='apartment_highrise',
    name='Apartment High-rise',
    name_kr='아파트 고층',
    category='residential',
    floors_options=[15, 20, 25],
    gfa_range=(10000, 25000),
    wwr={'south': 0.45, 'north': 0.15, 'east': 0.20, 'west': 0.20},
    hvac=HVACConfig(
        heating_type='ondol_gas_boiler',
        cooling_type='wall_split',
        heating_cop=0.92,
        cooling_cop=3.5,
    ),
    occupancy_classes=[
        'residential_dual_income',
        'residential_single_income',
        'residential_single',
        'residential_elderly',
    ],
    schedule_variants_per_class=100,
    lighting_density=8.0,
    equipment_density=3.0,
    people_density=0.04,
    share=0.40,
))

_register(BuildingArchetype(
    archetype_id='apartment_midrise',
    name='Apartment Mid-rise',
    name_kr='아파트 중층',
    category='residential',
    floors_options=[5],
    gfa_range=(2000, 5000),
    wwr={'south': 0.40, 'north': 0.12, 'east': 0.18, 'west': 0.18},
    hvac=HVACConfig(
        heating_type='ondol_gas_boiler',
        cooling_type='wall_split',
        heating_cop=0.90,
        cooling_cop=3.2,
    ),
    occupancy_classes=[
        'residential_dual_income',
        'residential_single_income',
        'residential_single',
        'residential_elderly',
    ],
    schedule_variants_per_class=100,
    lighting_density=8.0,
    equipment_density=3.0,
    people_density=0.04,
    share=0.15,
))

_register(BuildingArchetype(
    archetype_id='office',
    name='Office',
    name_kr='업무시설',
    category='office',
    floors_options=[10, 15, 20],
    gfa_range=(8000, 30000),
    wwr={'south': 0.50, 'north': 0.50, 'east': 0.40, 'west': 0.40},
    hvac=HVACConfig(
        heating_type='vrf',
        cooling_type='vrf',
        heating_cop=3.8,
        cooling_cop=4.0,
    ),
    occupancy_classes=[
        'office_government',
        'office_corporate',
        'office_it_startup',
        'office_callcenter',
        'office_small',
    ],
    schedule_variants_per_class=100,
    lighting_density=12.0,
    equipment_density=18.0,
    people_density=0.10,
    share=0.15,
))

_register(BuildingArchetype(
    archetype_id='school',
    name='School',
    name_kr='학교',
    category='school',
    floors_options=[4],
    gfa_range=(5000, 10000),
    wwr={'south': 0.35, 'north': 0.15, 'east': 0.20, 'west': 0.20},
    hvac=HVACConfig(
        heating_type='gas_boiler',
        cooling_type='vrf',
        heating_cop=0.88,
        cooling_cop=3.5,
    ),
    occupancy_classes=[
        'school_elementary',
        'school_secondary',
        'school_university',
    ],
    schedule_variants_per_class=50,
    lighting_density=10.0,
    equipment_density=8.0,
    people_density=0.25,
    share=0.08,
))

_register(BuildingArchetype(
    archetype_id='university',
    name='University',
    name_kr='대학교',
    category='school',
    floors_options=[4, 5, 6],
    gfa_range=(5000, 15000),
    wwr={'south': 0.40, 'north': 0.20, 'east': 0.25, 'west': 0.25},
    hvac=HVACConfig(
        heating_type='gas_boiler',
        cooling_type='vrf',
        heating_cop=0.88,
        cooling_cop=3.5,
    ),
    occupancy_classes=['school_university'],
    schedule_variants_per_class=50,
    lighting_density=12.0,
    equipment_density=12.0,
    people_density=0.20,
    share=0.05,
))

_register(BuildingArchetype(
    archetype_id='retail',
    name='Retail',
    name_kr='판매시설',
    category='retail',
    floors_options=[3, 5],
    gfa_range=(15000, 40000),
    wwr={'south': 0.25, 'north': 0.25, 'east': 0.25, 'west': 0.25},
    hvac=HVACConfig(
        heating_type='chiller_boiler',
        cooling_type='chiller_ahu',
        heating_cop=0.90,
        cooling_cop=4.5,
    ),
    occupancy_classes=[
        'retail_mall',
        'retail_department',
        'retail_convenience',
        'retail_neighborhood',
    ],
    schedule_variants_per_class=50,
    lighting_density=16.0,
    equipment_density=10.0,
    people_density=0.15,
    share=0.08,
))

_register(BuildingArchetype(
    archetype_id='hospital',
    name='Hospital',
    name_kr='병원',
    category='hospital',
    floors_options=[8, 12, 15],
    gfa_range=(15000, 50000),
    wwr={'south': 0.35, 'north': 0.25, 'east': 0.25, 'west': 0.25},
    hvac=HVACConfig(
        heating_type='chiller_boiler',
        cooling_type='chiller_ahu_fcu',
        heating_cop=0.90,
        cooling_cop=4.5,
    ),
    occupancy_classes=[
        'hospital_general',
        'hospital_clinic',
        'hospital_nursing',
    ],
    schedule_variants_per_class=50,
    lighting_density=14.0,
    equipment_density=20.0,
    people_density=0.08,
    share=0.07,
))

_register(BuildingArchetype(
    archetype_id='hotel',
    name='Hotel',
    name_kr='호텔',
    category='hotel',
    floors_options=[15],
    gfa_range=(8000, 20000),
    wwr={'south': 0.30, 'north': 0.30, 'east': 0.25, 'west': 0.25},
    hvac=HVACConfig(
        heating_type='vrf',
        cooling_type='vrf',
        heating_cop=3.8,
        cooling_cop=4.0,
    ),
    occupancy_classes=[
        'hotel_business',
        'hotel_tourist',
        'hotel_resort',
    ],
    schedule_variants_per_class=50,
    lighting_density=12.0,
    equipment_density=8.0,
    people_density=0.03,
    share=0.07,
))

_register(BuildingArchetype(
    archetype_id='small_office',
    name='Small Office',
    name_kr='소규모 업무시설',
    category='office',
    floors_options=[1, 2],
    gfa_range=(200, 1500),
    wwr={'south': 0.25, 'north': 0.25, 'east': 0.25, 'west': 0.25},
    hvac=HVACConfig(
        heating_type='gas_boiler',
        cooling_type='split_dx',
        heating_cop=0.88,
        cooling_cop=3.2,
    ),
    occupancy_classes=[
        'office_small',
        'office_corporate',
    ],
    schedule_variants_per_class=50,
    lighting_density=9.0,
    equipment_density=7.0,
    people_density=0.06,
    share=0.05,
))

_register(BuildingArchetype(
    archetype_id='large_office',
    name='Large Office',
    name_kr='대규모 업무시설',
    category='office',
    floors_options=[12, 20, 30],
    gfa_range=(40000, 100000),
    wwr={'south': 0.60, 'north': 0.60, 'east': 0.40, 'west': 0.40},
    hvac=HVACConfig(
        heating_type='chiller_boiler',
        cooling_type='chiller_ahu',
        heating_cop=0.92,
        cooling_cop=5.0,
    ),
    occupancy_classes=[
        'office_corporate',
        'office_government',
    ],
    schedule_variants_per_class=50,
    lighting_density=12.0,
    equipment_density=18.0,
    people_density=0.10,
    share=0.05,
))

_register(BuildingArchetype(
    archetype_id='warehouse',
    name='Warehouse',
    name_kr='창고시설',
    category='warehouse',
    floors_options=[1],
    gfa_range=(2000, 10000),
    wwr={'south': 0.05, 'north': 0.05, 'east': 0.05, 'west': 0.05},
    hvac=HVACConfig(
        heating_type='gas_boiler',
        cooling_type='split_dx',
        heating_cop=0.85,
        cooling_cop=3.0,
    ),
    occupancy_classes=[
        'office_small',
    ],
    schedule_variants_per_class=50,
    lighting_density=5.0,
    equipment_density=3.0,
    people_density=0.02,
    share=0.03,
))

_register(BuildingArchetype(
    archetype_id='restaurant_full',
    name='Full Service Restaurant',
    name_kr='일반음식점',
    category='food',
    floors_options=[1, 2],
    gfa_range=(300, 1000),
    wwr={'south': 0.30, 'north': 0.10, 'east': 0.20, 'west': 0.20},
    hvac=HVACConfig(
        heating_type='gas_boiler',
        cooling_type='split_dx',
        heating_cop=0.88,
        cooling_cop=3.5,
        has_ventilation=True,
        ventilation_type='exhaust',
    ),
    occupancy_classes=[
        'retail_neighborhood',
    ],
    schedule_variants_per_class=50,
    lighting_density=15.0,
    equipment_density=40.0,
    people_density=0.20,
    share=0.02,
))

_register(BuildingArchetype(
    archetype_id='restaurant_quick',
    name='Quick Service Restaurant',
    name_kr='패스트푸드점',
    category='food',
    floors_options=[1],
    gfa_range=(150, 500),
    wwr={'south': 0.35, 'north': 0.10, 'east': 0.25, 'west': 0.25},
    hvac=HVACConfig(
        heating_type='gas_boiler',
        cooling_type='split_dx',
        heating_cop=0.88,
        cooling_cop=3.5,
        has_ventilation=True,
        ventilation_type='exhaust',
    ),
    occupancy_classes=[
        'retail_convenience',
    ],
    schedule_variants_per_class=50,
    lighting_density=15.0,
    equipment_density=50.0,
    people_density=0.25,
    share=0.02,
))

_register(BuildingArchetype(
    archetype_id='strip_mall',
    name='Strip Mall',
    name_kr='근린상가',
    category='retail',
    floors_options=[1, 2],
    gfa_range=(500, 3000),
    wwr={'south': 0.30, 'north': 0.10, 'east': 0.20, 'west': 0.20},
    hvac=HVACConfig(
        heating_type='gas_boiler',
        cooling_type='split_dx',
        heating_cop=0.88,
        cooling_cop=3.2,
    ),
    occupancy_classes=[
        'retail_neighborhood',
        'retail_convenience',
    ],
    schedule_variants_per_class=50,
    lighting_density=14.0,
    equipment_density=8.0,
    people_density=0.12,
    share=0.03,
))


# ============================================================
# 조회 함수
# ============================================================

def get_archetype(archetype_id: str) -> BuildingArchetype:
    if archetype_id not in ARCHETYPES:
        raise ValueError(f"Unknown archetype: {archetype_id}. "
                         f"Available: {list(ARCHETYPES.keys())}")
    return ARCHETYPES[archetype_id]


def list_archetypes() -> List[str]:
    return list(ARCHETYPES.keys())


def compute_simulation_matrix() -> dict:
    """전체 시뮬레이션 매트릭스 계산"""
    cities_per_zone = {
        'central_1': ['chuncheon', 'wonju'],
        'central_2': ['seoul', 'incheon', 'daejeon', 'sejong'],
        'southern': ['busan', 'daegu', 'gwangju'],
        'jeju': ['jeju'],
    }
    weather_years = ['tmy', 'amy2023']

    total = 0
    breakdown = {}

    for arch_id, arch in ARCHETYPES.items():
        arch_total = 0
        for vintage in VINTAGES:
            for zone, cities in cities_per_zone.items():
                for city in cities:
                    for wy in weather_years:
                        n_variants = arch.total_schedule_variants()
                        arch_total += n_variants
        breakdown[arch_id] = arch_total
        total += arch_total

    return {
        'total_simulations': total,
        'per_archetype': breakdown,
        'n_archetypes': len(ARCHETYPES),
        'n_vintages': len(VINTAGES),
        'n_cities': sum(len(c) for c in cities_per_zone.values()),
        'n_weather_years': len(weather_years),
    }


if __name__ == '__main__':
    print("=== 건물 아키타입 요약 ===\n")
    print(f"{'ID':<22} {'이름':<12} {'비중':>5} {'스케줄수':>8} {'층수'}")
    print("-" * 65)
    for aid, a in ARCHETYPES.items():
        print(f"{aid:<22} {a.name_kr:<12} {a.share:5.0%} {a.total_schedule_variants():>8} {a.floors_options}")

    print(f"\n총 스케줄 아키타입: {sum(a.total_schedule_variants() for a in ARCHETYPES.values())}")

    print("\n=== 시뮬레이션 매트릭스 ===")
    matrix = compute_simulation_matrix()
    print(f"총 시뮬레이션 수: {matrix['total_simulations']:,}")
    for arch_id, n in matrix['per_archetype'].items():
        print(f"  {arch_id:<22}: {n:>10,}")
