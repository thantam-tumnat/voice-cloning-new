@echo off
title Thai TTS Tone Studio (Port 8011)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   Thai TTS Tone Studio  --  http://localhost:8011
echo ============================================================
echo   Tone annotation + voice cloning UI. Loads no model of its
echo   own -- generation goes to the shared GPU service on 8020.
echo.
echo   Web UI:   http://localhost:8011/
echo   Swagger:  http://localhost:8011/docs
echo.
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found in "%CD%". Create it first:
    echo     uv venv --python 3.11 .venv
    echo     uv pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
curl -s -o nul --max-time 10 http://127.0.0.1:8020/health
if errorlevel 1 (
    echo WARNING: the GPU service on port 8020 is not answering.
    echo          The studio will start, but /synthesize returns 503
    echo          until voice-cloning\start-gpu.bat is up.
    echo.
)
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8011 --reload
echo.
echo Tone Studio stopped.
pause
