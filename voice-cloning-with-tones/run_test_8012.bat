@echo off
title Test Page (:8012) - Thonburian F5 + SeedVC
echo ========================================================
echo   Starting server on http://localhost:8012
echo   Test page will open when the backend is READY
echo   Backend: Thonburian F5 + SeedVC
echo ========================================================
set FLOWTTS_SRC=C:\Users\opendream002\Desktop\thonburian\thonburian-tts
set SERVICE_PORT=8012
set SEEDVC_URL=http://127.0.0.1:8022

REM Poll /health in the background; open the test page only once it responds 200
start "" /b cmd /c "for /l %%i in (1,1,120) do (curl -s -o nul -m 2 http://localhost:8012/health && (start http://localhost:8012/test & exit) || timeout /t 2 >nul)"

python -m uvicorn app.main:app --host 0.0.0.0 --port 8012 --reload
pause
