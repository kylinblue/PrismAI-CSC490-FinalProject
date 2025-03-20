@echo off
setlocal

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ========================================================
    echo ERROR: Python is not installed
    echo Please install Python from:
    echo https://www.python.org/downloads/
    echo ========================================================
    exit /b 1
)

:: Run the Python Docker installation script
python scripts\docker_install.py
if %ERRORLEVEL% neq 0 (
    echo Docker setup failed
    exit /b 1
)

echo Docker setup completed successfully!
