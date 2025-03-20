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

# Run the Python Docker installation script
python3 scripts/docker_install.py
if [ $? -ne 0 ]; then
    echo "Docker setup failed"
    exit 1
fi

echo "Docker setup completed successfully!"
