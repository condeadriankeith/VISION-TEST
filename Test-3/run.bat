@echo off
title AR Hand-Controlled 3D Objects - Test 3
cd /d "%~dp0"

echo =======================================================
echo     AR Hand-Controlled 3D Object System  (Test-3)
echo =======================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in your system PATH.
    pause
    exit /b 1
)

echo [INFO] Launching application...
echo.
python main.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Application terminated unexpectedly.
    pause
)
