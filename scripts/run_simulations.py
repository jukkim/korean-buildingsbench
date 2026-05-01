"""
EnergyPlus 배치 시뮬레이션 실행

simulations/idfs/ 아래의 IDF 파일을 EnergyPlus로 실행.
멀티프로세싱으로 병렬 실행 지원.

Usage:
    # 전체 실행 (CPU 코어수 자동)
    python scripts/run_simulations.py

    # 워커 수 지정
    python scripts/run_simulations.py --workers 8

    # 특정 아키타입만
    python scripts/run_simulations.py --filter office

    # 실패한 것만 재실행
    python scripts/run_simulations.py --retry-failed
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
IDF_DIR = PROJECT_DIR / 'simulations' / 'idfs'
RESULT_DIR = PROJECT_DIR / 'simulations' / 'results'

# EnergyPlus 실행 파일
EP_EXE = Path(os.environ.get('ENERGYPLUS_EXE', 'C:/EnergyPlusV24-1-0/energyplus.exe'))


def find_idf_jobs(filter_str: str = '', retry_failed: bool = False):
    """실행할 IDF 작업 목록 생성"""
    jobs = []

    for meta_file in sorted(IDF_DIR.rglob('metadata.json')):
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        building_id = meta['building_id']
        idf_path = meta.get('idf_path', '')

        if not idf_path or not Path(idf_path).exists():
            continue

        if filter_str and filter_str not in building_id:
            continue

        # 결과 디렉토리
        result_subdir = RESULT_DIR / building_id
        result_csv = result_subdir / 'eplusout.csv'
        status_file = result_subdir / 'status.json'

        # 이미 완료된 건 스킵
        if result_csv.exists() and not retry_failed:
            continue

        # retry-failed: 실패한 것만
        if retry_failed and status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            if status.get('success', False):
                continue

        # weather file 경로 해석 (idf_mapping.yaml 참조)
        city = meta.get('city', 'seoul')
        CITY_TO_WEATHER = {
            'chuncheon': 'KOR_Gangneung.epw',
            'wonju': 'KOR_Cheongju.epw',
            'seoul': 'KOR_Seoul.epw',
            'incheon': 'KOR_Incheon.epw',
            'daejeon': 'KOR_Daejeon.epw',
            'sejong': 'KOR_Daejeon.epw',
            'busan': 'KOR_Busan.epw',
            'daegu': 'KOR_Daegu.epw',
            'gwangju': 'KOR_Gwangju.epw',
            'gangneung': 'KOR_Gangneung.epw',
            'jeju': 'KOR_Jeju.epw',
            'ulsan': 'KOR_Ulsan.epw',
        }
        weather_name = CITY_TO_WEATHER.get(city, f'KOR_{city.capitalize()}.epw')
        weather_file = PROJECT_DIR / 'weather' / weather_name

        jobs.append({
            'building_id': building_id,
            'idf_path': idf_path,
            'weather_file': str(weather_file),
            'output_dir': str(result_subdir),
            'meta': meta,
        })

    return jobs


def run_single_simulation(job: dict) -> dict:
    """단일 EnergyPlus 시뮬레이션 실행"""
    building_id = job['building_id']
    idf_path = job['idf_path']
    weather_file = job['weather_file']
    output_dir = job['output_dir']

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    try:
        cmd = [
            str(EP_EXE),
            '-w', weather_file,
            '-d', output_dir,
            '-r',   # readvars
            '-a',   # annual
            idf_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30분 타임아웃 (대형 건물 모델용)
        )

        elapsed = time.time() - t0

        # 결과 CSV 확인 (returncode보다 CSV 존재+행수가 더 신뢰할 수 있음)
        # Windows에서 병렬 실행시 readvars.audit 파일 잠금으로 rc=1 가능
        csv_path = Path(output_dir) / 'eplusout.csv'
        has_csv = csv_path.exists()
        csv_rows = 0
        if has_csv:
            with open(csv_path, 'r') as f:
                csv_rows = sum(1 for _ in f) - 1  # header 제외

        success = has_csv and csv_rows >= 8760

        status = {
            'building_id': building_id,
            'success': success,
            'returncode': result.returncode,
            'elapsed_seconds': round(elapsed, 1),
            'has_csv': has_csv,
            'csv_rows': csv_rows,
        }

        if not success:
            # 에러 메시지 저장
            err_path = Path(output_dir) / 'eplusout.err'
            if err_path.exists():
                with open(err_path, 'r', errors='replace') as f:
                    err_lines = f.readlines()
                severe = [l.strip() for l in err_lines if '** Severe  **' in l]
                fatal = [l.strip() for l in err_lines if '**  Fatal  **' in l]
                status['severe_errors'] = severe[:5]
                status['fatal_errors'] = fatal[:3]

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = {
            'building_id': building_id,
            'success': False,
            'error': 'timeout (600s)',
            'elapsed_seconds': round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        status = {
            'building_id': building_id,
            'success': False,
            'error': str(e),
            'elapsed_seconds': round(elapsed, 1),
        }

    # 상태 저장
    status_path = Path(output_dir) / 'status.json'
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)

    return status


def main():
    parser = argparse.ArgumentParser(description='EnergyPlus 배치 실행')
    parser.add_argument('--workers', type=int, default=None,
                        help='병렬 워커 수 (기본: CPU-2)')
    parser.add_argument('--filter', type=str, default='',
                        help='building_id 필터 (예: office)')
    parser.add_argument('--retry-failed', action='store_true',
                        help='실패한 시뮬만 재실행')
    parser.add_argument('--idf-dir', type=str, default='',
                        help='IDF 디렉토리 (기본: simulations/idfs)')
    parser.add_argument('--result-dir', type=str, default='',
                        help='결과 디렉토리 (기본: simulations/results)')
    args = parser.parse_args()

    if not EP_EXE.exists():
        print(f"[ERROR] EnergyPlus not found: {EP_EXE}")
        print("  ENERGYPLUS_EXE 환경변수 또는 기본 경로 확인")
        return

    # 사용자 지정 디렉토리
    global IDF_DIR, RESULT_DIR
    if args.idf_dir:
        IDF_DIR = Path(args.idf_dir)
    if args.result_dir:
        RESULT_DIR = Path(args.result_dir)

    # 작업 목록
    jobs = find_idf_jobs(filter_str=args.filter, retry_failed=args.retry_failed)

    if not jobs:
        print("실행할 IDF가 없습니다.")
        return

    n_workers = args.workers or max(1, os.cpu_count() - 2)

    print("=" * 70)
    print(f"EnergyPlus Batch Simulation")
    print(f"  Jobs: {len(jobs):,}")
    print(f"  Workers: {n_workers}")
    print(f"  EnergyPlus: {EP_EXE}")
    print(f"  Output: {RESULT_DIR}")
    print("=" * 70)

    t0 = time.time()
    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_simulation, job): job for job in jobs}

        for i, future in enumerate(as_completed(futures)):
            status = future.result()
            if status['success']:
                success += 1
            else:
                failed += 1
                err = status.get('error', status.get('fatal_errors', ['unknown']))
                print(f"  [FAIL] {status['building_id']}: {err}")

            if (i + 1) % 10 == 0 or (i + 1) == len(jobs):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i + 1:>6,}/{len(jobs):,}] "
                      f"OK:{success} FAIL:{failed} "
                      f"{rate:.1f}/s ETA:{eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"완료: {success:,} OK, {failed:,} FAIL ({elapsed:.0f}s)")


if __name__ == '__main__':
    main()
