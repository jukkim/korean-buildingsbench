"""시뮬 결과 CSV → Tier A NPY 일괄 추출 (postprocess 없이 독립 실행).

Usage:
    python scripts/extract_npy.py --all          # 모든 results_v3_* 처리
    python scripts/extract_npy.py --step s14_mid  # 특정 step만
    python scripts/extract_npy.py --workers 4     # 병렬 처리
    python scripts/extract_npy.py --fix-water-gas # 기존 npy에 water_gas NaN 추가
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ─── 채널 매핑 (postprocess.py와 동일) ──────────────────────
METER_TO_NPY = {
    'Electricity:Facility':         'hourly_electricity',
    'Cooling:Electricity':          'hourly_cooling',
    'Heating:Electricity':          'hourly_heating',
    'Heating:NaturalGas':           'hourly_gas',
    'Fans:Electricity':             'hourly_fans',
    'InteriorEquipment:Electricity': 'hourly_equipment',
    'InteriorLights:Electricity':   'hourly_lights',
    'Pumps:Electricity':            'hourly_pumps',
    'WaterSystems:NaturalGas':      'hourly_water_gas',
}

VARIABLE_TO_NPY = {
    'Site Outdoor Air Drybulb Temperature': 'hourly_outdoor_temp',
    'Site Outdoor Air Relative Humidity':   'hourly_outdoor_humidity',
    'Site Direct Solar Radiation':          'hourly_solar',
    'Zone Mean Air Temperature':            'hourly_zone_temp',
    'Zone Air Relative Humidity':           'hourly_indoor_humidity',
    'Zone Thermostat Cooling Setpoint':     'hourly_setpoint_cool',
    'Zone Thermostat Heating Setpoint':     'hourly_setpoint_heat',
}


def to_hourly(arr: np.ndarray) -> np.ndarray:
    if len(arr) > 8760:
        n = len(arr) // 8760
        arr = arr[:8760 * n].reshape(8760, n).sum(axis=1)
    arr = arr[:8760]
    if len(arr) < 8760:
        pad = np.zeros(8760)
        pad[:len(arr)] = arr
        arr = pad
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_and_save(result_dir: Path, npy_root: Path) -> str:
    """단일 건물 CSV → NPY 추출 + 저장. 반환: 'ok' | 'skip' | 'fail:reason'

    우선순위:
    1) eplusout.csv (미터 + 존 변수 모두 포함) — 신규 IDF (s20+)
    2) eplusmtr.csv (미터만, 존 변수 = NaN) — 구 IDF (s1-s5) fallback
    """
    csv_full = result_dir / 'eplusout.csv'
    csv_mtr  = result_dir / 'eplusmtr.csv'

    has_full = csv_full.exists()
    has_mtr  = csv_mtr.exists()

    if not has_full and not has_mtr:
        return 'skip'

    building_id = result_dir.name
    npy_dir = npy_root / building_id

    # 이미 추출된 경우 스킵
    if (npy_dir / 'hourly_electricity.npy').exists():
        return 'skip'

    channels = {}
    meter_only = False  # eplusmtr.csv fallback 사용 시 True

    if has_full:
        try:
            df = pd.read_csv(csv_full, index_col=0)
        except Exception as e:
            return f'fail:csv_read:{e}'
        if len(df) < 8760:
            return f'fail:rows:{len(df)}'

        # 에너지 미터 (J → kWh)
        for keyword, npy_name in METER_TO_NPY.items():
            cols = [c for c in df.columns if keyword in c]
            if cols:
                vals = df[cols[0]].values.astype(np.float64)
                channels[npy_name] = to_hourly(vals) / 3_600_000.0
            else:
                channels[npy_name] = np.full(8760, np.nan, dtype=np.float32)

        # 변수 (온도, 습도 등)
        for keyword, npy_name in VARIABLE_TO_NPY.items():
            cols = [c for c in df.columns if keyword in c]
            if cols:
                arr = df[cols].values.astype(np.float64)
                if arr.ndim == 2 and arr.shape[1] > 1:
                    arr = arr.mean(axis=1)
                else:
                    arr = arr.flatten()
                channels[npy_name] = to_hourly(arr)
            else:
                channels[npy_name] = np.full(8760, np.nan, dtype=np.float32)

    else:
        # eplusmtr.csv fallback — 미터만, 존 변수 NaN
        meter_only = True
        try:
            df = pd.read_csv(csv_mtr, index_col=0)
        except Exception as e:
            return f'fail:mtr_read:{e}'
        if len(df) < 8760:
            return f'fail:rows:{len(df)}'

        for keyword, npy_name in METER_TO_NPY.items():
            cols = [c for c in df.columns if keyword in c]
            if cols:
                vals = df[cols[0]].values.astype(np.float64)
                channels[npy_name] = to_hourly(vals) / 3_600_000.0
            else:
                channels[npy_name] = np.full(8760, np.nan, dtype=np.float32)

        # 존 변수는 모두 NaN
        for npy_name in VARIABLE_TO_NPY.values():
            channels[npy_name] = np.full(8760, np.nan, dtype=np.float32)

    # peak_demand = electricity copy
    if 'hourly_electricity' in channels:
        channels['hourly_peak_demand'] = channels['hourly_electricity'].copy()

    # 저장
    npy_dir.mkdir(parents=True, exist_ok=True)

    # metadata 복사
    meta_src = result_dir / 'metadata.json'
    if meta_src.exists():
        try:
            with open(meta_src, encoding='utf-8') as f:
                meta = json.load(f)
            meta['output_tier'] = 'A'
            meta['source_project'] = 'Korean_BB'
            # 존 변수가 NaN이면 partial_tier_a=True
            has_nan_channel = any(np.all(np.isnan(v)) for v in channels.values())
            meta['partial_tier_a'] = meter_only or has_nan_channel
            with open(npy_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    for npy_name, arr in channels.items():
        np.save(npy_dir / f'{npy_name}.npy', arr)

    return 'ok'


def fix_water_gas(npy_root: Path, workers: int):
    """기존 npy_tier_a 건물에 hourly_water_gas.npy (NaN) 추가 — 채널 통일."""
    dirs = sorted([
        d for d in npy_root.iterdir()
        if d.is_dir()
        and (d / 'hourly_electricity.npy').exists()
        and not (d / 'hourly_water_gas.npy').exists()
    ])
    if not dirs:
        print('hourly_water_gas.npy가 누락된 건물 없음.')
        return

    nan_arr = np.full(8760, np.nan, dtype=np.float32)
    t0 = time.time()
    for i, d in enumerate(dirs):
        np.save(d / 'hourly_water_gas.npy', nan_arr)
        if (i + 1) % 5000 == 0:
            print(f'  [{i+1:,}/{len(dirs):,}] ({time.time()-t0:.0f}s)', flush=True)

    print(f'완료: {len(dirs):,}건에 hourly_water_gas.npy (NaN) 추가 ({time.time()-t0:.0f}s)')


def process_step(step_dir: Path, npy_root: Path, workers: int):
    """한 step의 모든 건물 처리 (eplusout.csv 또는 eplusmtr.csv 보유 폴더)."""
    building_dirs = sorted([
        d for d in step_dir.iterdir()
        if d.is_dir() and ((d / 'eplusout.csv').exists() or (d / 'eplusmtr.csv').exists())
    ])

    if not building_dirs:
        print(f'  {step_dir.name}: 0 buildings (skip)')
        return 0, 0, 0

    ok = skip = fail = 0
    t0 = time.time()

    if workers <= 1:
        for i, bdir in enumerate(building_dirs):
            result = extract_and_save(bdir, npy_root)
            if result == 'ok': ok += 1
            elif result == 'skip': skip += 1
            else: fail += 1
            if (i + 1) % 500 == 0:
                print(f'  [{i+1:>6,}/{len(building_dirs):,}] ok={ok} skip={skip} fail={fail} ({time.time()-t0:.0f}s)', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(extract_and_save, bd, npy_root): bd for bd in building_dirs}
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                if result == 'ok': ok += 1
                elif result == 'skip': skip += 1
                else: fail += 1
                if i % 500 == 0:
                    print(f'  [{i:>6,}/{len(building_dirs):,}] ok={ok} skip={skip} fail={fail} ({time.time()-t0:.0f}s)', flush=True)

    elapsed = time.time() - t0
    print(f'  {step_dir.name}: ok={ok} skip={skip} fail={fail} ({elapsed:.0f}s)', flush=True)
    return ok, skip, fail


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSV → Tier A NPY 일괄 추출')
    parser.add_argument('--all', action='store_true', help='모든 results_v3_* 처리')
    parser.add_argument('--step', type=str, help='특정 step만 (e.g. s14_mid)')
    parser.add_argument('--npy-root', type=str, default=None,
                        help='NPY 출력 루트 (기본: simulations/npy_tier_a/)')
    parser.add_argument('--workers', type=int, default=1, help='병렬 처리 수')
    parser.add_argument('--fix-water-gas', action='store_true',
                        help='기존 npy_tier_a 건물에 hourly_water_gas.npy (NaN) 추가')
    args = parser.parse_args()

    npy_root = Path(args.npy_root) if args.npy_root else PROJECT_DIR / 'simulations' / 'npy_tier_a'
    npy_root.mkdir(parents=True, exist_ok=True)

    if args.fix_water_gas:
        fix_water_gas(npy_root, args.workers)
        sys.exit(0)

    sim_root = PROJECT_DIR / 'simulations'

    if args.step:
        step_dirs = [sim_root / f'results_v3_{args.step}']
    elif args.all:
        step_dirs = sorted(sim_root.glob('results_v3_s*'))
    else:
        parser.error('--all, --step, 또는 --fix-water-gas 필수')

    print(f'NPY 추출: {len(step_dirs)} steps → {npy_root}')
    print(f'Workers: {args.workers}')
    print()

    total_ok = total_skip = total_fail = 0
    t_start = time.time()

    for sd in step_dirs:
        if not sd.is_dir():
            print(f'  {sd.name}: NOT FOUND')
            continue
        o, s, f = process_step(sd, npy_root, args.workers)
        total_ok += o
        total_skip += s
        total_fail += f

    elapsed = time.time() - t_start
    print(f'\n완료: ok={total_ok:,} skip={total_skip:,} fail={total_fail:,} ({elapsed:.0f}s)')
    print(f'출력: {npy_root}')
