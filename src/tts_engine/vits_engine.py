from __future__ import annotations

import asyncio
import io
import struct
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from loguru import logger

from .engine import TTSEngine, TTSRequest, TTSResponse, AudioChunk


def postprocess_audio(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    if len(audio) == 0:
        return audio

    audio = audio.astype(np.float32)

    audio -= np.mean(audio)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.85

    fade_in = int(0.005 * sample_rate)
    fade_out = int(0.01 * sample_rate)
    if len(audio) > fade_in + fade_out:
        audio[:fade_in] *= np.linspace(0, 1, fade_in)
        audio[-fade_out:] *= np.linspace(1, 0, fade_out)

    return audio.astype(np.float32)


class CoquiTTSEngine(TTSEngine):
    XTTS_LANG_MAP = {
        'ta': 'en',
        'en': 'en',
        'mixed': 'en',
    }
    SUPPORTED_LANGUAGES = {'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi'}

    def __init__(
        self,
        model_name: str = 'tts_models/multilingual/multi-dataset/xtts_v2',
        device: str = 'cpu',
        max_concurrent: int = 20,
        cache_dir: str | None = None,
        speaker_wav: str | None = None,
        temperature: float = 0.5,
        repetition_penalty: float = 10.0,
        length_penalty: float = 1.0,
        speed: float = 1.0,
    ):
        self.model_name = model_name
        self.device = device
        self.max_concurrent = max_concurrent
        self.cache_dir = cache_dir
        self._speaker_wav = speaker_wav
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._length_penalty = length_penalty
        self._speed = speed
        self._model = None
        self._ready = False
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._active_requests: deque[str] = deque(maxlen=1000)
        self._synthesis_stats: dict = {
            'total_requests': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
        }

    def initialize(self) -> None:
        logger.info(f"Loading TTS model: {self.model_name} on {self.device}")
        start = time.time()
        try:
            from TTS.api import TTS as CoquiTTS
            self._model = CoquiTTS(
                self.model_name,
                gpu=self.device == 'cuda',
                progress_bar=False,
            )
            self._ready = True
            logger.info(
                f"Model loaded in {time.time() - start:.2f}s "
                f"on device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            self._model = None
            self._ready = False

    def _get_language(self, lang: str) -> str:
        mapped = self.XTTS_LANG_MAP.get(lang, 'en')
        if mapped not in self.SUPPORTED_LANGUAGES:
            return 'en'
        return mapped

    def _infer(self, request: TTSRequest) -> TTSResponse:
        start_time = time.time()
        lang = self._get_language(request.language)

        try:
            speaker_wav = request.speaker_wav or self._speaker_wav

            kwargs = {
                'text': request.text,
                'language': lang,
                'split_sentences': True,
            }
            if speaker_wav:
                kwargs['speaker_wav'] = speaker_wav

            wav_list = self._model.tts(**kwargs)

            audio = np.array(wav_list, dtype=np.float32)
            sample_rate = 24000
            if hasattr(self._model, 'synthesizer') and hasattr(
                self._model.synthesizer, 'output_sample_rate'
            ):
                sample_rate = self._model.synthesizer.output_sample_rate

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
            logger.error(f"Synthesis failed: {e}")
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
        result = await loop.run_in_executor(
            self._executor, self._infer, request
        )

        if result.audio is None:
            yield AudioChunk(
                data=b'', chunk_index=0, is_last=True,
                request_id=request.request_id,
            )
            return

        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, result.audio, result.sample_rate, format='WAV')
        audio_bytes.seek(0)
        full_data = audio_bytes.read()

        header_size = 44
        yield AudioChunk(
            data=full_data[:header_size],
            chunk_index=0,
            is_last=False,
            request_id=request.request_id,
        )

        chunk_index = 1
        offset = header_size
        while offset < len(full_data):
            end = min(offset + chunk_size, len(full_data))
            is_last = end >= len(full_data)
            yield AudioChunk(
                data=full_data[offset:end],
                chunk_index=chunk_index,
                is_last=is_last,
                request_id=request.request_id,
            )
            offset = end
            chunk_index += 1
            if not is_last:
                await asyncio.sleep(0.001)

    def cleanup(self) -> None:
        self._model = None
        self._ready = False
        self._executor.shutdown(wait=False)
        logger.info("TTS engine cleaned up")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_info(self) -> dict:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'ready': self._ready,
            'active_requests': len(self._active_requests),
            'stats': dict(self._synthesis_stats),
        }


class MockTTSEngine(TTSEngine):
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._ready = True
        self._active_requests: deque[str] = deque(maxlen=1000)
        self._synthesis_stats: dict = {
            'total_requests': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
        }

    def initialize(self) -> None:
        self._ready = True
        logger.info("Mock TTS engine initialized")

    def _generate_speech(self, text: str) -> np.ndarray:
        duration = max(0.5, len(text) * 0.05)
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        freq = 220
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        audio += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        envelope = np.exp(-3 * t / duration)
        audio *= envelope
        audio += np.random.normal(0, 0.01, len(audio))
        return audio.astype(np.float32)

    def _infer(self, request: TTSRequest) -> TTSResponse:
        start_time = time.time()
        audio = self._generate_speech(request.text)
        latency_ms = (time.time() - start_time) * 1000

        self._synthesis_stats['total_requests'] += 1
        self._synthesis_stats['total_latency_ms'] += latency_ms
        self._synthesis_stats['avg_latency_ms'] = (
            self._synthesis_stats['total_latency_ms']
            / self._synthesis_stats['total_requests']
        )

        return TTSResponse(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=len(audio) / self.sample_rate,
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

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_info(self) -> dict:
        return {
            'model_name': 'mock',
            'device': 'cpu',
            'ready': self._ready,
            'active_requests': len(self._active_requests),
            'stats': dict(self._synthesis_stats),
        }
