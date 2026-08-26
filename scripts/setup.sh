#!/bin/bash
set -e

echo "=== TTS Tamil - Setup ==="

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate the virtual environment: source venv/bin/activate"
echo "Run the server: python -m src.main"
echo "Run tests: pytest tests/"
