from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "auto"
    max_concurrent: int = 20
    enable_streaming: bool = True
    log_level: str = "INFO"
    model_dir: str = "./models"
    cache_dir: str = "./models/.cache"
    sample_rate: int = 24000
    audio_format: str = "wav"
    tts_languages: Optional[dict[str, str]] = None
    tamil_speaker_wav: Optional[str] = None
    english_speaker_wav: Optional[str] = None

    model_config = {"env_prefix": "TTS_"}

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
