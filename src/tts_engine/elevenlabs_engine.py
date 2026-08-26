from __future__ import annotations

import io
import time
from collections import deque
from typing import AsyncIterator

import numpy as np
import soundfile as sf
import requests
from loguru import logger

from .engine import TTSEngine, TTSRequest, TTSResponse, AudioChunk
from .vits_engine import postprocess_audio


class ElevenLabsTTSEngine(TTSEngine):
    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self._ready = False
        self._active_requests: deque[str] = deque(maxlen=1000)
        self._synthesis_stats: dict = {
            'total_requests': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
        }

    def initialize(self) -> None:
        logger.info("Validating ElevenLabs API key...")
        try:
            resp = requests.get(
                f"{self.BASE_URL}/user",
                headers={"xi-api-key": self.api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"ElevenLabs ready: {data.get('first_name', 'User')}")
                self._ready = True
            elif resp.status_code == 401:
                logger.error("Invalid ElevenLabs API key")
                self._ready = False
            else:
                logger.error(f"ElevenLabs API error: {resp.status_code}")
                self._ready = False
        except Exception as e:
            logger.error(f"ElevenLabs connection failed: {e}")
            self._ready = False

    def _synthesize_audio(self, text: str, language: str = 'ta') -> tuple[np.ndarray, int]:
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

        resp = requests.post(
            f"{self.BASE_URL}/text-to-speech/{self.voice_id}",
            headers=headers,
            json=payload,
            params={"output_format": self.output_format},
            timeout=30,
        )
        resp.raise_for_status()

        audio_bytes = io.BytesIO(resp.content)
        audio, sample_rate = sf.read(audio_bytes, format='MP3' if 'mp3' in self.output_format else 'WAV')

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32), sample_rate

    def _infer(self, request: TTSRequest) -> TTSResponse:
        start_time = time.time()

        try:
            audio, sample_rate = self._synthesize_audio(
                request.text, request.language
            )

            audio = postprocess_audio(audio, sample_rate)

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
            logger.error(f"ElevenLabs synthesis failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return TTSResponse(
                audio=None,
                sample_rate=44100,
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
        import asyncio
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
        logger.info("ElevenLabs engine cleaned up")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_info(self) -> dict:
        return {
            'model_name': f'elevenlabs-{self.model_id}',
            'device': 'cloud',
            'ready': self._ready,
            'active_requests': len(self._active_requests),
            'stats': dict(self._synthesis_stats),
        }
