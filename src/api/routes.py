from __future__ import annotations

import asyncio
import io
import re
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from loguru import logger

from .models import (
    TTSRequestBody,
    TTSHealthResponse,
    TTSModelInfo,
    TTSErrorResponse,
    TTSNormalizedRequest,
)

router = APIRouter(prefix="/api/v1", tags=["tts"])

_tts_engine = None
_tamil_engine = None
_normalizer = None
_streaming_synth = None
_start_time = None

TAMIL_RANGE = re.compile(r'[\u0B80-\u0BFF]')


def detect_language(text: str) -> str:
    has_tamil = bool(TAMIL_RANGE.search(text))
    has_english = bool(re.search(r'[A-Za-z]', text))
    if has_tamil and has_english:
        return 'mixed'
    if has_tamil:
        return 'ta'
    return 'en'


def init_routes(tts_engine, tamil_engine, normalizer, streaming_synth):
    global _tts_engine, _tamil_engine, _normalizer, _streaming_synth, _start_time
    _tts_engine = tts_engine
    _tamil_engine = tamil_engine
    _normalizer = normalizer
    _streaming_synth = streaming_synth
    _start_time = time.time()


@router.get("/health", response_model=TTSHealthResponse)
async def health_check():
    english_ready = _tts_engine and _tts_engine.is_ready
    tamil_ready = _tamil_engine and _tamil_engine.is_ready
    return TTSHealthResponse(
        status="healthy" if english_ready and tamil_ready else "degraded",
        model_loaded=english_ready,
        model_name="xtts_v2+mms-tts-tam",
        device=_tts_engine.model_info.get('device', 'unknown') if _tts_engine else 'unknown',
        uptime_seconds=time.time() - _start_time if _start_time else 0,
    )


@router.get("/model", response_model=TTSModelInfo)
async def model_info():
    if not _tts_engine:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    return TTSModelInfo(**_tts_engine.model_info)


@router.post("/synthesize")
async def synthesize_speech(body: TTSRequestBody, request: Request):
    if not _tts_engine or not _tts_engine.is_ready:
        raise HTTPException(status_code=503, detail="TTS engine not ready")

    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        detected_lang = body.language or detect_language(body.text)
        normalized_text = _normalizer.normalize_for_tts(body.text, target_lang=detected_lang)

        logger.info(
            f"[{request_id}] Synthesis request: lang={detected_lang}, "
            f"text_len={len(body.text)}, normalized_len={len(normalized_text)}"
        )

        body.language = detected_lang

        if body.stream:
            return await _handle_streaming(request_id, body, normalized_text)
        else:
            return await _handle_blocking(request_id, body, normalized_text, start_time)

    except Exception as e:
        logger.error(f"[{request_id}] Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_streaming(request_id: str, body: TTSRequestBody, normalized_text: str) -> StreamingResponse:
    from ..tts_engine.engine import TTSRequest

    request = TTSRequest(
        text=normalized_text,
        language=body.language,
        speaker_wav=body.speaker_wav,
        temperature=body.temperature,
        speed=body.speed,
        stream=True,
        request_id=request_id,
    )

    async def audio_stream():
        first_chunk_time = None
        async for chunk in _tts_engine.synthesize_streaming(request):
            if first_chunk_time is None:
                first_chunk_time = time.time()
                logger.info(f"[{request_id}] TTFA: {(first_chunk_time - time.time()) * 1000:.1f}ms")
            yield chunk.data

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={
            "X-Request-ID": request_id,
            "X-Language": body.language,
            "Content-Disposition": f'attachment; filename="tts_{request_id}.wav"',
        },
    )


async def _handle_blocking(
    request_id: str, body: TTSRequestBody, normalized_text: str, start_time: float
) -> Response:
    from ..tts_engine.engine import TTSRequest

    request = TTSRequest(
        text=normalized_text,
        language=body.language,
        speaker_wav=body.speaker_wav,
        temperature=body.temperature,
        speed=body.speed,
        stream=False,
        request_id=request_id,
    )

    loop = asyncio.get_event_loop()

    if body.language == 'ta' and _tamil_engine and _tamil_engine.is_ready:
        result = await loop.run_in_executor(None, _tamil_engine.synthesize, request)
    else:
        result = await loop.run_in_executor(None, _tts_engine.synthesize, request)

    total_latency = (time.time() - start_time) * 1000

    if result.audio is None:
        raise HTTPException(status_code=500, detail="Synthesis failed")

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, result.audio, result.sample_rate, format='WAV')
    wav_buffer.seek(0)
    audio_bytes = wav_buffer.read()

    logger.info(
        f"[{request_id}] Synthesis complete: "
        f"duration={result.duration:.2f}s, latency={total_latency:.1f}ms, "
        f"first_chunk={result.first_chunk_latency_ms:.1f}ms"
    )

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Request-ID": request_id,
            "X-Language": body.language,
            "X-Latency-Ms": f"{total_latency:.1f}",
            "X-First-Chunk-Latency-Ms": f"{result.first_chunk_latency_ms:.1f}",
            "X-Audio-Duration": f"{result.duration:.2f}",
            "Content-Disposition": f'attachment; filename="tts_{request_id}.wav"',
        },
    )


@router.post("/normalize", response_model=TTSNormalizedRequest)
async def normalize_text(body: TTSRequestBody):
    from ..text_normalization.tamil_normalizer import detect_language_segments

    detected_lang = body.language or detect_language(body.text)
    normalized_text = _normalizer.normalize_for_tts(body.text, target_lang=detected_lang)
    segments = detect_language_segments(normalized_text)

    return TTSNormalizedRequest(
        original_text=body.text,
        normalized_text=normalized_text,
        language=detected_lang,
        segments=[
            {"language": lang, "text": text} for lang, text in segments
        ] if segments else None,
    )


@router.post("/synthesize/multi")
async def synthesize_multi_language(requests: list[TTSRequestBody]):
    if not _tts_engine or not _tts_engine.is_ready:
        raise HTTPException(status_code=503, detail="TTS engine not ready")

    results = []
    for req in requests[:5]:
        try:
            normalized = _normalizer.normalize_for_tts(req.text, target_lang=req.language)
            from ..tts_engine.engine import TTSRequest
            tts_req = TTSRequest(
                text=normalized,
                language=req.language,
                speaker_wav=req.speaker_wav,
                temperature=req.temperature,
                speed=req.speed,
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _tts_engine.synthesize, tts_req)

            if result.audio is not None:
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, result.audio, result.sample_rate, format='WAV')
                wav_buffer.seek(0)
                import base64
                audio_b64 = base64.b64encode(wav_buffer.read()).decode()
                results.append({
                    "request_id": result.request_id,
                    "language": req.language,
                    "text": req.text,
                    "normalized_text": normalized,
                    "duration": result.duration,
                    "latency_ms": result.latency_ms,
                    "audio_base64": audio_b64,
                })
            else:
                results.append({
                    "request_id": result.request_id,
                    "language": req.language,
                    "error": "synthesis_failed",
                })
        except Exception as e:
            results.append({
                "request_id": str(uuid.uuid4()),
                "language": req.language,
                "error": str(e),
            })

    return {"results": results, "count": len(results)}
