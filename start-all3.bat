@echo off
title START ALL 3 (Reload + LAN) - ThonburianTTS + SeedVC Tone Studio
chcp 65001 >nul

REM --- Auto-elevate to Admin (needed for firewall rules) ---------------
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges for firewall setup...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ======================================================================
echo   START ALL 3  -  SeedVC Worker (:8022) + Tone Studio (:8012)
echo   Mode: --reload  ^|  Network: open to LAN
echo ======================================================================
echo.

if exist "%~dp0voice-cloning-with-tones" (
    cd /d "%~dp0voice-cloning-with-tones"
) else (
    cd /d "%~dp0"
)

REM --- Paths / env -----------------------------------------------------
set SEEDVC_REPO=C:\Users\opendream002\Desktop\seed-vc
set SEEDVC_PYTHON=C:\Users\opendream002\Desktop\seed-vc\seedvc-venv\Scripts\python.exe
if not exist "%SEEDVC_PYTHON%" set SEEDVC_PYTHON=python

set FLOWTTS_SRC=C:\Users\opendream002\Desktop\thonburian\thonburian-tts
set SERVICE_PORT=8012
set SEEDVC_URL=http://127.0.0.1:8022

REM --- 0) Firewall rules ----------------------------------------------
echo [0/3] Ensuring Windows Firewall allows inbound TCP 8012 / 8022 ...
netsh advfirewall firewall show rule name="ToneStudio 8012" >nul 2>&1
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="ToneStudio 8012" dir=in action=allow protocol=TCP localport=8012 >nul 2>&1
    echo       added rule: ToneStudio 8012
) else (
    echo       rule already present: ToneStudio 8012
)
netsh advfirewall firewall show rule name="SeedVC Worker 8022" >nul 2>&1
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="SeedVC Worker 8022" dir=in action=allow protocol=TCP localport=8022 >nul 2>&1
    echo       added rule: SeedVC Worker 8022
) else (
    echo       rule already present: SeedVC Worker 8022
)

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

REM --- Collect all usable LAN IPv4 addresses ---------------------------
echo.
echo ======================================================================
echo   Tone Studio URLs
echo ----------------------------------------------------------------------
echo     Local   : http://localhost:8012/test
for /f "tokens=*" %%a in ('powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 ^| Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.254.*' } ^| Select-Object -ExpandProperty IPAddress"') do (
    echo     Network : http://%%a:8012/test
)
echo ======================================================================
echo.

REM --- 3) Launch the studio with --reload -----------------------------
echo [3/3] Launching Tone Studio (:8012) with --reload ...
start "" /b cmd /c "for /l %%i in (1,1,120) do (curl -s -o nul -m 2 http://localhost:8012/health && (start http://localhost:8012/test & exit) || timeout /t 2 >nul)"

python -m uvicorn app.main:app --host 0.0.0.0 --port %SERVICE_PORT% --reload --reload-dir app

echo.
echo Tone Studio stopped. Press any key to close.
pause >nul
