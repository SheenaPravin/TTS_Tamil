#!/bin/bash
set -e

echo "=== TTS Tamil - Setup ==="

PYTHON_CMD="python3.11"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Using: $($PYTHON_CMD --version)"

# Use uv for fast dependency resolution (pip has macOS truststore bug)
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    brew install uv
fi

rm -rf venv
uv venv --python 3.11 venv
source venv/bin/activate
uv pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate the virtual environment: source venv/bin/activate"
echo "Run the server: python -m src.main"
echo "Run tests: pytest tests/"
