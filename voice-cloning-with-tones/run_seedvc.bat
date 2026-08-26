@echo off
title SeedVC Worker (:8022)
echo ========================================================
echo   Starting SeedVC Voice Conversion Worker on Port 8022
echo ========================================================
set SEEDVC_REPO=C:\Users\opendream002\Desktop\seed-vc
set PYTHON_EXE=C:\Users\opendream002\Desktop\seed-vc\seedvc-venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" tools\seedvc_server.py --seedvc-repo "%SEEDVC_REPO%" --port 8022
pause
