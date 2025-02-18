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
    echo "    brew install python3"
    echo ""
    echo "Or download from:"
    echo "    https://www.python.org/downloads/"
    echo "========================================================"
    exit 1
fi

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Run the Python installation script
python3 scripts/install.py
