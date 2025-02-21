@echo off

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ========================================================
    echo ERROR: Python is not installed or not in PATH
    echo Please download and install Python 3.11 from:
    echo https://www.python.org/downloads/windows/
    echo ========================================================
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Run the Python installation script
python scripts\install.py
if errorlevel 1 (
    echo Installation failed
    pause
    exit /b 1
)

echo Installation completed successfully!
echo To start using the application, activate the virtual environment:
echo     .venv\Scripts\activate.bat
pause
