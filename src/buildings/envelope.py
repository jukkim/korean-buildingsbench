"""
건물 연식(Vintage)별 외피 성능 정의

한국 에너지절약설계기준 기반 U-value 및 기밀성.
5개 vintage × 4개 기후대 = 20개 조합.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EnvelopeSpec:
    """건물 외피 사양"""
    vintage: str
    climate_zone: str
    building_category: str  # 'residential' or 'non_residential'

    # U-values (W/m²K)
    wall_u: float = 0.0
    roof_u: float = 0.0
    floor_u: float = 0.0
    window_u: float = 0.0

    # 기밀성
    infiltration_ach: float = 0.5  # Air Changes per Hour

    # HVAC 효율 보정
    hvac_efficiency_factor: float = 1.0

    # 기타
    has_hrv: bool = False  # 열회수환기장치
    shgc: float = 0.4      # Solar Heat Gain Coefficient

    def to_dict(self) -> dict:
        return {
            'vintage': self.vintage,
            'climate_zone': self.climate_zone,
            'building_category': self.building_category,
            'wall_u': self.wall_u,
            'roof_u': self.roof_u,
            'floor_u': self.floor_u,
            'window_u': self.window_u,
            'infiltration_ach': self.infiltration_ach,
            'hvac_efficiency_factor': self.hvac_efficiency_factor,
            'has_hrv': self.has_hrv,
            'shgc': self.shgc,
        }


# ============================================================
# U-value 테이블: [vintage][climate_zone][category]
# 에너지절약설계기준 연도별 기준값
# ============================================================

# 주거용 U-values (W/m²K)
_RESIDENTIAL_U = {
    # vintage: {climate_zone: (wall, roof, floor, window)}
    'v1_pre1990': {
        'central_1': (0.80, 0.60, 0.80, 4.5),
        'central_2': (0.70, 0.55, 0.70, 4.0),
        'southern':  (0.60, 0.50, 0.60, 3.5),
        'jeju':      (0.55, 0.45, 0.55, 3.5),
    },
    'v2_1991_2000': {
        'central_1': (0.58, 0.40, 0.58, 3.5),
        'central_2': (0.47, 0.35, 0.47, 3.0),
        'southern':  (0.42, 0.32, 0.42, 2.7),
        'jeju':      (0.40, 0.30, 0.40, 2.7),
    },
    'v3_2001_2010': {
        'central_1': (0.47, 0.30, 0.47, 2.7),
        'central_2': (0.36, 0.25, 0.36, 2.3),
        'southern':  (0.32, 0.22, 0.32, 2.1),
        'jeju':      (0.30, 0.22, 0.30, 2.1),
    },
    'v4_2011_2017': {
        'central_1': (0.27, 0.18, 0.27, 1.5),
        'central_2': (0.21, 0.15, 0.21, 1.2),
        'southern':  (0.26, 0.18, 0.26, 1.5),
        'jeju':      (0.33, 0.25, 0.33, 1.8),
    },
    'v5_2018_plus': {
        'central_1': (0.150, 0.150, 0.150, 1.0),
        'central_2': (0.170, 0.150, 0.170, 1.0),
        'southern':  (0.220, 0.180, 0.220, 1.2),
        'jeju':      (0.290, 0.250, 0.290, 1.6),
    },
}

# 비주거용 U-values (W/m²K)
_NON_RESIDENTIAL_U = {
    'v1_pre1990': {
        'central_1': (1.00, 0.70, 1.00, 5.0),
        'central_2': (0.80, 0.60, 0.80, 4.5),
        'southern':  (0.70, 0.55, 0.70, 4.0),
        'jeju':      (0.65, 0.50, 0.65, 4.0),
    },
    'v2_1991_2000': {
        'central_1': (0.65, 0.45, 0.65, 3.8),
        'central_2': (0.55, 0.40, 0.55, 3.3),
        'southern':  (0.50, 0.35, 0.50, 3.0),
        'jeju':      (0.45, 0.35, 0.45, 3.0),
    },
    'v3_2001_2010': {
        'central_1': (0.50, 0.35, 0.50, 3.0),
        'central_2': (0.42, 0.30, 0.42, 2.5),
        'southern':  (0.38, 0.28, 0.38, 2.2),
        'jeju':      (0.35, 0.28, 0.35, 2.2),
    },
    'v4_2011_2017': {
        'central_1': (0.32, 0.22, 0.32, 1.8),
        'central_2': (0.28, 0.20, 0.28, 1.5),
        'southern':  (0.35, 0.25, 0.35, 1.8),
        'jeju':      (0.42, 0.35, 0.42, 2.2),
    },
    'v5_2018_plus': {
        'central_1': (0.170, 0.150, 0.170, 1.2),
        'central_2': (0.240, 0.180, 0.240, 1.5),
        'southern':  (0.320, 0.250, 0.320, 1.8),
        'jeju':      (0.410, 0.350, 0.410, 2.2),
    },
}

# 기밀성 (ACH)
_INFILTRATION = {
    'v1_pre1990':   1.0,
    'v2_1991_2000': 0.7,
    'v3_2001_2010': 0.5,
    'v4_2011_2017': 0.3,
    'v5_2018_plus': 0.2,
}

# HVAC 효율 보정계수
_HVAC_EFF = {
    'v1_pre1990':   0.70,
    'v2_1991_2000': 0.80,
    'v3_2001_2010': 0.90,
    'v4_2011_2017': 0.95,
    'v5_2018_plus': 1.00,
}

# HRV 설치 여부
_HAS_HRV = {
    'v1_pre1990':   False,
    'v2_1991_2000': False,
    'v3_2001_2010': False,
    'v4_2011_2017': True,   # 2013년 100세대 이상 의무화
    'v5_2018_plus': True,
}

# SHGC (Solar Heat Gain Coefficient)
_SHGC = {
    'v1_pre1990':   0.70,  # 단판/복층, 높은 일사 투과
    'v2_1991_2000': 0.55,
    'v3_2001_2010': 0.45,  # Low-E
    'v4_2011_2017': 0.35,  # 삼중 Low-E
    'v5_2018_plus': 0.30,  # 삼중 Low-E + 아르곤
}


def get_envelope(
    vintage: str,
    climate_zone: str,
    building_category: str = 'residential',
) -> EnvelopeSpec:
    """외피 사양 조회

    Args:
        vintage: v1_pre1990 ~ v5_2018_plus
        climate_zone: central_1, central_2, southern, jeju
        building_category: 'residential' or 'non_residential'

    Returns:
        EnvelopeSpec
    """
    if building_category == 'residential':
        u_table = _RESIDENTIAL_U
    else:
        u_table = _NON_RESIDENTIAL_U

    if vintage not in u_table:
        raise ValueError(f"Unknown vintage: {vintage}")
    if climate_zone not in u_table[vintage]:
        raise ValueError(f"Unknown climate zone: {climate_zone}")

    wall, roof, floor, window = u_table[vintage][climate_zone]

    return EnvelopeSpec(
        vintage=vintage,
        climate_zone=climate_zone,
        building_category=building_category,
        wall_u=wall,
        roof_u=roof,
        floor_u=floor,
        window_u=window,
        infiltration_ach=_INFILTRATION[vintage],
        hvac_efficiency_factor=_HVAC_EFF[vintage],
        has_hrv=_HAS_HRV[vintage],
        shgc=_SHGC[vintage],
    )


# 편의: 모든 vintage 목록
VINTAGES = list(_INFILTRATION.keys())
CLIMATE_ZONES = ['central_1', 'central_2', 'southern', 'jeju']


if __name__ == '__main__':
    print("=== 외피 사양 테이블 ===\n")
    for cat in ['residential', 'non_residential']:
        print(f"[{cat}]")
        print(f"{'Vintage':<16} {'Zone':<12} {'Wall':>6} {'Roof':>6} {'Floor':>6} {'Window':>7} {'ACH':>5} {'HRV':>5}")
        print("-" * 70)
        for v in VINTAGES:
            for z in CLIMATE_ZONES:
                spec = get_envelope(v, z, cat)
                print(f"{v:<16} {z:<12} {spec.wall_u:6.3f} {spec.roof_u:6.3f} "
                      f"{spec.floor_u:6.3f} {spec.window_u:7.2f} {spec.infiltration_ach:5.2f} "
                      f"{'Y' if spec.has_hrv else 'N':>5}")
        print()
