#!/bin/bash
set -e

echo "=== Downloading TTS Models ==="

mkdir -p models/.cache

echo "Model will be downloaded on first startup via Coqui TTS."
echo "Default model: tts_models/multilingual/multi-dataset/xtts_v2"
echo ""
echo "To pre-download, run:"
echo "  python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')\""
echo ""
echo "Alternative Tamil models:"
echo "  - tts_models/multilingual/multi-dataset/xtts_v2 (multilingual)"
echo "  - tts_models/en/ljspeech/tacotron2-DDC (English only)"

echo "=== Model download instructions complete ==="
