@echo off

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ========================================================
    echo ERROR: Python is not installed or not in PATH
    echo Please download and install Python from:
    echo https://www.python.org/downloads/
    echo ========================================================
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Create scripts directory if it doesn't exist
if not exist "scripts" mkdir scripts

REM Run the Python installation script
python scripts\install.py
pause
