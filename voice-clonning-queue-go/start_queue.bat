@echo off
title SiangTTS Go Fiber Queue Service (:8020)
set PORT=8020
set PYTHON_GPU_URL=http://127.0.0.1:8021
echo Starting Go Fiber Queue Gateway on http://127.0.0.1:%PORT% (Targeting GPU: %PYTHON_GPU_URL%)...
go run main.go
pause
