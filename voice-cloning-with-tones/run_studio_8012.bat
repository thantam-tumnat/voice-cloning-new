@echo off
title Tone Studio (:8012) - Thonburian F5 + SeedVC
echo ========================================================
echo   Starting Tone Studio on http://localhost:8012
echo   Backend: Thonburian F5 + SeedVC
echo ========================================================
set FLOWTTS_SRC=C:\Users\opendream002\Desktop\thonburian\thonburian-tts
set SERVICE_PORT=8012
set SEEDVC_URL=http://127.0.0.1:8022

python -m uvicorn app.main:app --host 0.0.0.0 --port 8012 --reload
pause
