@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
REM Korean_BB Forecast API 영구 가동 (port 8040, TwGauss-M v1.0 CVRMSE 12.93%%)
REM Cloudflare Tunnel: 없음 — ems_transformer Gateway :8030 가 통합 진입점
REM Gateway adapter: serving/adapters/kbb_client.py (KOREAN_BB_URL env)

cd /d "%~dp0"

REM 이미 8040이 사용 중이면 종료 (중복 방지)
netstat -ano | findstr "LISTENING" | findstr ":8040 " >nul 2>&1
if %errorlevel%==0 (
  echo [%date% %time%] Port 8040 already in use - skip start >> kbb_api.log
  exit /b 0
)

echo [%date% %time%] Starting Korean_BB API on port 8040 >> kbb_api.log
C:\Python313\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8040 --log-level warning >> kbb_api.log 2>&1
