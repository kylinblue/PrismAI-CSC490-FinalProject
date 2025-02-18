#!/bin/bash

# Ensure script fails on any error
set -e

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "========================================================"
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python using one of these methods:"
    echo ""
    echo "macOS (with homebrew):"
    echo "    brew install python@3.11"
    echo ""
    echo "Or download from:"
    echo "    https://www.python.org/downloads/"
    echo "========================================================"
    exit 1
fi

# Run the Python installation script
python3 scripts/install.py
if [ $? -ne 0 ]; then
    echo "Installation failed"
    exit 1
fi

echo "Installation completed successfully!"
echo "To start using the application, activate the virtual environment:"
echo "    source .venv/bin/activate"
