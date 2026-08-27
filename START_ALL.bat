@echo off
title START ALL - ThonburianTTS + SeedVC Tone Studio
chcp 65001 >nul

echo ======================================================================
echo   START ALL  -  SeedVC Worker (:8022) + Tone Studio (:8012)
echo ======================================================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%voice-cloning-with-tones"

REM --- Paths / env -----------------------------------------------------
set SEEDVC_REPO=C:\Users\opendream002\Desktop\seed-vc
set SEEDVC_PYTHON=C:\Users\opendream002\Desktop\seed-vc\seedvc-venv\Scripts\python.exe
if not exist "%SEEDVC_PYTHON%" set SEEDVC_PYTHON=python

set FLOWTTS_SRC=C:\Users\opendream002\Desktop\thonburian\thonburian-tts
set SERVICE_PORT=8012
set SEEDVC_URL=http://127.0.0.1:8022

REM --- 1) SeedVC worker: reuse if already healthy, else start ----------
echo [1/3] Checking SeedVC Worker (:8022) ...
curl -s -o nul -m 3 http://127.0.0.1:8022/health
if %errorlevel%==0 (
    echo       already running - reusing it.
) else (
    echo       starting SeedVC Worker in a new window...
    start "SeedVC Worker (:8022)" cmd /k "title SeedVC Worker (:8022) && "%SEEDVC_PYTHON%" tools\seedvc_server.py --seedvc-repo "%SEEDVC_REPO%" --port 8022"
    echo       waiting for it to load the model...
    python tools\wait_for_seedvc.py "http://127.0.0.1:8022/health"
)

REM --- 2) Free port 8012 so the studio starts on fresh code -----------
echo [2/3] Freeing Tone Studio port (:8012) if a stale server is holding it...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8012" ^| findstr "LISTENING"') do (
    echo       killing stale PID %%p
    taskkill /PID %%p /F >nul 2>&1
)

REM --- 3) Launch the studio (foreground; this window shows its logs) ---
echo [3/3] Launching Tone Studio (:8012)...
echo       browser opens automatically once the backend answers /health.
start "" /b cmd /c "for /l %%i in (1,1,120) do (curl -s -o nul -m 2 http://localhost:8012/health && (start http://localhost:8012/test & exit) || timeout /t 2 >nul)"

python -m uvicorn app.main:app --host 0.0.0.0 --port %SERVICE_PORT%

echo.
echo Tone Studio stopped. Press any key to close.
pause >nul
