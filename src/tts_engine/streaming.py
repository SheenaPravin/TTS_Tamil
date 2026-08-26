from __future__ import annotations

import asyncio
import io
import struct
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from loguru import logger


class StreamingAudioBuffer:
    def __init__(self, sample_rate: int = 24000, chunk_duration_ms: float = 50.0):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.buffer: list[np.ndarray] = []
        self.total_samples: int = 0

    def add_audio(self, audio: np.ndarray) -> None:
        self.buffer.append(audio)
        self.total_samples += len(audio)

    def get_chunks(self) -> AsyncIterator[bytes]:
        if not self.buffer:
            return

        full_audio = np.concatenate(self.buffer)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, full_audio, self.sample_rate, format='WAV')
        wav_buffer.seek(0)

        header = wav_buffer.read(44)
        yield header

        chunk_count = 0
        while True:
            chunk = wav_buffer.read(self.chunk_size * 2)
            if not chunk:
                break
            chunk_count += 1
            yield chunk

    async def get_chunks_async(self) -> AsyncIterator[bytes]:
        if not self.buffer:
            return

        full_audio = np.concatenate(self.buffer)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, full_audio, self.sample_rate, format='WAV')
        wav_buffer.seek(0)

        header = wav_buffer.read(44)
        yield header

        chunk_count = 0
        while True:
            chunk = wav_buffer.read(self.chunk_size * 2)
            if not chunk:
                break
            chunk_count += 1
            yield chunk
            await asyncio.sleep(0.005)

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / self.sample_rate


class StreamingSynthesizer:
    def __init__(self, tts_engine, normalizer):
        self.tts_engine = tts_engine
        self.normalizer = normalizer
        self._pending_requests: dict[str, StreamingAudioBuffer] = {}

    async def synthesize_stream(
        self,
        text: str,
        language: str = 'en',
        speaker_wav: str | None = None,
        temperature: float = 0.7,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        from ..tts_engine.engine import TTSRequest

        if request_id is None:
            request_id = str(uuid.uuid4())

        normalized_text = self.normalizer.normalize_for_tts(text, target_lang=language)

        logger.info(
            f"Streaming synthesis: request_id={request_id}, "
            f"lang={language}, text_len={len(normalized_text)}"
        )

        request = TTSRequest(
            text=normalized_text,
            language=language,
            speaker_wav=speaker_wav,
            temperature=temperature,
            stream=True,
            request_id=request_id,
        )

        start_time = time.time()
        first_chunk_sent = False

        async for chunk in self.tts_engine.synthesize_streaming(request):
            yield chunk.data
            if not first_chunk_sent:
                ttfa = (time.time() - start_time) * 1000
                logger.info(f"TTFA for {request_id}: {ttfa:.1f}ms")
                first_chunk_sent = True

    async def synthesize_to_buffer(
        self,
        text: str,
        language: str = 'en',
        speaker_wav: str | None = None,
        temperature: float = 0.7,
    ) -> tuple[bytes, dict]:
        from ..tts_engine.engine import TTSRequest

        request_id = str(uuid.uuid4())
        normalized_text = self.normalizer.normalize_for_tts(text, target_lang=language)

        request = TTSRequest(
            text=normalized_text,
            language=language,
            speaker_wav=speaker_wav,
            temperature=temperature,
            stream=False,
            request_id=request_id,
        )

        start_time = time.time()
        result = self.tts_engine.synthesize(request)
        total_latency = (time.time() - start_time) * 1000

        if result.audio is None:
            return b'', {
                'request_id': request_id,
                'error': 'synthesis_failed',
                'latency_ms': total_latency,
            }

        import io
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, result.audio, result.sample_rate, format='WAV')
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()

        metadata = {
            'request_id': request_id,
            'sample_rate': result.sample_rate,
            'duration': result.duration,
            'latency_ms': total_latency,
            'first_chunk_latency_ms': result.first_chunk_latency_ms,
            'language': language,
            'text_length': len(text),
            'normalized_length': len(normalized_text),
        }

        return audio_bytes, metadata
