@echo off
title Hand Hologram - 3D Hovering Cubes
cd /d "%~dp0"

echo =======================================================
echo       Hand Gesture 3D Hologram Application
echo =======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10+ and ensure "Add Python to PATH" is checked.
    echo.
    pause
    exit /b 1
)

:: Ensure dependencies are installed
echo [INFO] Checking dependencies...
python -c "import cv2, mediapipe, numpy" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo [INFO] Launching application...
echo.
python main.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Application terminated unexpectedly.
    pause
)
