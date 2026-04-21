"""
창 없이 BAT 파일 실행 (CMD 창 닫아도 학습 유지됨)
사용법: python launch_hidden.py run_exp034_cells79.bat logs/exp034_cells79_runner.log
"""
import sys, subprocess, os

BASE = r'C:\Users\User\Desktop\myjob\8.simulation\Korean_BB'

bat_file = sys.argv[1]
log_file = sys.argv[2] if len(sys.argv) > 2 else 'logs/hidden_runner.log'

bat_path = os.path.join(BASE, bat_file)
log_path = os.path.join(BASE, log_file)

os.makedirs(os.path.dirname(log_path), exist_ok=True)

with open(log_path, 'wb') as log_out:
    p = subprocess.Popen(
        ['cmd', '/c', bat_path],
        stdout=log_out,
        stderr=log_out,
        cwd=BASE,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )

print(f'Launched: {bat_file}')
print(f'PID: {p.pid}')
print(f'Log: {log_path}')
print('CMD 창을 닫아도 학습이 계속됩니다.')
