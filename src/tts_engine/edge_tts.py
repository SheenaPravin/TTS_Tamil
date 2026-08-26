from __future__ import annotations

import asyncio
import io
import tempfile
import time
from collections import deque
from typing import AsyncIterator

import edge_tts
import numpy as np
import soundfile as sf
from loguru import logger

from .engine import TTSEngine, TTSRequest, TTSResponse, AudioChunk


EDGE_TTS_VOICES = {
    'ta': 'ta-IN-PallaviNeural',
    'en': 'en-US-JennyNeural',
}


class EdgeTTSEngine(TTSEngine):
    def __init__(self, voice: str | None = None, rate: str = '+0%', pitch: str = '+0Hz'):
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._ready = False
        self._active_requests: deque[str] = deque(maxlen=1000)
        self._synthesis_stats: dict = {
            'total_requests': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
        }

    def initialize(self) -> None:
        self._ready = True
        logger.info("Edge TTS engine ready (Microsoft Azure neural voices)")

    def _get_voice(self, language: str) -> str:
        if self._voice:
            return self._voice
        return EDGE_TTS_VOICES.get(language, 'en-US-JennyNeural')

    def _synthesize_sync(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_path = f.name

        async def _do():
            communicate = edge_tts.Communicate(text, voice, rate=self._rate, pitch=self._pitch)
            await communicate.save(tmp_path)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_do())
        finally:
            loop.close()

        audio, sample_rate = sf.read(tmp_path, dtype='float32')
        import os
        os.unlink(tmp_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio, sample_rate

    def _infer(self, request: TTSRequest) -> TTSResponse:
        start_time = time.time()
        voice = self._get_voice(request.language)

        try:
            audio, sample_rate = self._synthesize_sync(request.text, voice)

            if request.speed != 1.0 and request.speed > 0:
                indices = np.round(
                    np.arange(0, len(audio), request.speed)
                ).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

            latency_ms = (time.time() - start_time) * 1000
            duration = len(audio) / sample_rate

            self._synthesis_stats['total_requests'] += 1
            self._synthesis_stats['total_latency_ms'] += latency_ms
            self._synthesis_stats['avg_latency_ms'] = (
                self._synthesis_stats['total_latency_ms']
                / self._synthesis_stats['total_requests']
            )

            return TTSResponse(
                audio=audio,
                sample_rate=sample_rate,
                duration=duration,
                request_id=request.request_id,
                latency_ms=latency_ms,
                first_chunk_latency_ms=latency_ms,
                language=request.language,
            )

        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return TTSResponse(
                audio=None,
                sample_rate=24000,
                duration=0.0,
                request_id=request.request_id,
                latency_ms=latency_ms,
                first_chunk_latency_ms=latency_ms,
                language=request.language,
            )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._active_requests.append(request.request_id)
        try:
            return self._infer(request)
        finally:
            if request.request_id in self._active_requests:
                self._active_requests.remove(request.request_id)

    async def synthesize_streaming(
        self, request: TTSRequest, chunk_size: int = 4096
    ) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._infer, request)

        if result.audio is None:
            yield AudioChunk(data=b'', chunk_index=0, is_last=True,
                           request_id=request.request_id)
            return

        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, result.audio, result.sample_rate, format='WAV')
        audio_bytes.seek(0)
        full_data = audio_bytes.read()

        header_size = 44
        yield AudioChunk(data=full_data[:header_size], chunk_index=0,
                        is_last=False, request_id=request.request_id)

        chunk_index = 1
        offset = header_size
        while offset < len(full_data):
            end = min(offset + chunk_size, len(full_data))
            is_last = end >= len(full_data)
            yield AudioChunk(data=full_data[offset:end], chunk_index=chunk_index,
                           is_last=is_last, request_id=request.request_id)
            offset = end
            chunk_index += 1
            if not is_last:
                await asyncio.sleep(0.001)

    def cleanup(self) -> None:
        self._ready = False
        logger.info("Edge TTS engine cleaned up")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_info(self) -> dict:
        return {
            'model_name': 'edge-tts',
            'device': 'cloud',
            'ready': self._ready,
            'active_requests': len(self._active_requests),
            'stats': dict(self._synthesis_stats),
        }
