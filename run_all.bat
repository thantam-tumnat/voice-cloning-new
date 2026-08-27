@echo off
title Launcher - ThonburianTTS + SeedVC Tone Studio (:8012)
chcp 65001 >nul

echo ======================================================================
echo   Starting ThonburianTTS + SeedVC Tone Studio Pipeline
echo ======================================================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%voice-cloning-with-tones"

set SEEDVC_REPO=C:\Users\opendream002\Desktop\seed-vc
set SEEDVC_PYTHON=C:\Users\opendream002\Desktop\seed-vc\seedvc-venv\Scripts\python.exe
if not exist "%SEEDVC_PYTHON%" (
    set SEEDVC_PYTHON=python
)

set FLOWTTS_SRC=C:\Users\opendream002\Desktop\thonburian\thonburian-tts
set SERVICE_PORT=8012
set SEEDVC_URL=http://127.0.0.1:8022

echo [1/3] Starting SeedVC Worker Service (:8022) in background window...
start "SeedVC Worker (:8022)" cmd /k "title SeedVC Worker (:8022) && "%SEEDVC_PYTHON%" tools\seedvc_server.py --seedvc-repo "%SEEDVC_REPO%" --port 8022"

echo [2/3] Waiting for SeedVC Worker (:8022) to initialize and be ready...
python tools\wait_for_seedvc.py "http://127.0.0.1:8022/health"

echo [3/3] Launching Tone Studio (:8012)...
echo.
echo Opening browser at http://localhost:8012 ...
start http://localhost:8012/

python -m uvicorn app.main:app --host 0.0.0.0 --port 8012 --reload
pause
