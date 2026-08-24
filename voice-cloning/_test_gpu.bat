@echo off
title SiangTTS GPU Service (Port 8020)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   SiangTTS GPU Service  --  http://127.0.0.1:8020
echo ============================================================
echo   The only process that loads VoxCPM2 + the Thai LoRA.
echo   Start this FIRST -- the webhook (8010) and the tone studio
echo   (8011) are HTTP clients of it.
echo.
echo   Model load takes 30-60 s. Ready when you see "[gpu] ready".
echo   Config comes from .env in this folder.
echo.
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found in "%CD%". Create it first:
    echo     uv venv --python 3.11 .venv
    echo     uv sync --extra serve
    echo.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m uvicorn src.gpu_service:app --host 127.0.0.1 --port 8020
echo.
echo GPU service stopped.
echo -- pause --
