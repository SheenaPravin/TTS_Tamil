from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger


@dataclass
class TTSRequest:
    text: str
    language: str = 'en'
    speaker_wav: str | None = None
    temperature: float = 0.7
    speed: float = 1.0
    stream: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TTSResponse:
    audio: np.ndarray | None = None
    sample_rate: int = 24000
    duration: float = 0.0
    request_id: str = ''
    latency_ms: float = 0.0
    first_chunk_latency_ms: float = 0.0
    language: str = 'en'
    format: str = 'wav'


@dataclass
class AudioChunk:
    data: bytes
    chunk_index: int
    is_last: bool
    request_id: str


class TTSEngine(ABC):
    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        pass

    @abstractmethod
    def synthesize_streaming(
        self, request: TTSRequest, chunk_size: int = 4096
    ) -> AsyncIterator[AudioChunk]:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @property
    @abstractmethod
    def model_info(self) -> dict:
        pass
