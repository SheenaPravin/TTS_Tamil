from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from loguru import logger

from .engine import TTSEngine, TTSRequest, TTSResponse, AudioChunk
from .vits_engine import postprocess_audio


class MMSTTSEngine(TTSEngine):
    def __init__(self, lang_code: str = 'tam', device: str = 'cpu'):
        self.lang_code = lang_code
        self.device = device
        self._model = None
        self._tokenizer = None
        self._ready = False
        self._active_requests: deque[str] = deque(maxlen=1000)
        self._synthesis_stats: dict = {
            'total_requests': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
        }

    def initialize(self) -> None:
        logger.info(f"Loading MMS-TTS model for language: {self.lang_code}")
        start = time.time()
        try:
            from transformers import AutoTokenizer, AutoModelForTextToWaveform
            model_name = f'facebook/mms-tts-{self.lang_code}'
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForTextToWaveform.from_pretrained(model_name)
            self._model.eval()
            self._ready = True
            logger.info(f"MMS-TTS loaded in {time.time() - start:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load MMS-TTS: {e}")
            self._model = None
            self._tokenizer = None
            self._ready = False

    def _infer(self, request: TTSRequest) -> TTSResponse:
        import torch
        start_time = time.time()

        try:
            inputs = self._tokenizer(request.text, return_tensors='pt')
            with torch.no_grad():
                output = self._model(**inputs).waveform

            audio = output.squeeze().numpy().astype(np.float32)
            sample_rate = self._model.config.sampling_rate

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
            logger.error(f"MMS-TTS synthesis failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return TTSResponse(
                audio=None,
                sample_rate=16000,
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
        import io as _io
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._infer, request)

        if result.audio is None:
            yield AudioChunk(data=b'', chunk_index=0, is_last=True,
                           request_id=request.request_id)
            return

        audio_bytes = _io.BytesIO()
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
        self._model = None
        self._tokenizer = None
        self._ready = False
        logger.info("MMS-TTS engine cleaned up")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_info(self) -> dict:
        return {
            'model_name': f'facebook/mms-tts-{self.lang_code}',
            'device': self.device,
            'ready': self._ready,
            'active_requests': len(self._active_requests),
            'stats': dict(self._synthesis_stats),
        }
