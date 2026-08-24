@echo off
title SiangTTS Webhook / Queue (Port 8010)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   SiangTTS Webhook / Queue  --  http://localhost:8010
echo ============================================================
echo   n8n / LiveAI contract, chunking, ffmpeg merge, upload,
echo   callback. Loads no model -- restarting it is cheap.
echo.
echo   Queue page:  http://localhost:8010/
echo   Swagger:     http://localhost:8010/docs
echo.
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found in "%CD%". Create it first:
    echo     uv venv --python 3.11 .venv
    echo     uv sync --extra serve
    echo.
    pause
    exit /b 1
)
curl -s -o nul --max-time 10 http://127.0.0.1:8020/health
if errorlevel 1 (
    echo WARNING: the GPU service on port 8020 is not answering.
    echo          The webhook will start, but every job fails until
    echo          start-gpu.bat is running and has printed "[gpu] ready".
    echo.
)
echo Listening on 0.0.0.0 -- there is NO authentication, so keep port
echo 8010 blocked at the firewall. Use 127.0.0.1 for local-only tests.
echo.
.venv\Scripts\python.exe -m uvicorn src.webhook:app --host 0.0.0.0 --port 8010
echo.
echo Webhook stopped.
pause
