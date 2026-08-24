@echo off
title Thai TTS Tone Studio (Port 8011)
cd /d "%~dp0"
echo Starting Thai TTS Tone Studio on port 8011...
echo.
echo Requires the shared SiangTTS GPU service on port 8020 -- see how_to_run.txt.
echo.
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Create it first:
    echo     uv venv --python 3.11 .venv
    echo     uv pip install -r requirements.txt
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8011 --reload
pause
