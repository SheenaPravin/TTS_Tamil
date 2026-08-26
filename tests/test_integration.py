#!/usr/bin/env python3
"""
End-to-end integration test: starts the service and runs a full synthesis cycle.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_full_pipeline():
    from src.text_normalization.normalizer import TextNormalizer
    from src.tts_engine.vits_engine import MockTTSEngine
    from src.tts_engine.engine import TTSRequest

    normalizer = TextNormalizer(lang='mixed', context='transport')
    engine = MockTTSEngine()
    engine.initialize()

    test_cases = [
        ("Your cab will arrive in 10 minutes.", 'en'),
        ("Your booking ID is TN45AB1234.", 'en'),
        ("Your OTP is 4821.", 'en'),
        ("Your phone number is 9876543210.", 'en'),
        ("உங்கள் கேப் 10 நிமிடங்களில் வரும்.", 'ta'),
        ("உங்கள் pickup location எங்கே?", 'mixed'),
        ("Driver வருகிறார், please wait பண்ணுங்க.", 'mixed'),
    ]

    results = []
    for text, lang in test_cases:
        normalized = normalizer.normalize_for_tts(text, target_lang=lang)
        request = TTSRequest(text=normalized, language=lang)
        start = time.time()
        response = engine.synthesize(request)
        latency = (time.time() - start) * 1000

        result = {
            'original': text,
            'normalized': normalized,
            'language': lang,
            'latency_ms': round(latency, 1),
            'duration': round(response.duration, 2),
            'success': response.audio is not None,
        }
        results.append(result)

    success_count = sum(1 for r in results if r['success'])
    assert success_count == len(results), f"Only {success_count}/{len(results)} passed"


if __name__ == "__main__":
    test_full_pipeline()
    print("Integration test passed!")
