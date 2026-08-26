from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class TTSRequestBody(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000,
        description="Text to synthesize into speech"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language: 'en', 'ta', 'mixed'. Auto-detected if omitted."
    )
    speaker_wav: Optional[str] = Field(
        default=None,
        description="Path to speaker reference WAV for voice cloning"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Temperature for sampling"
    )
    speed: float = Field(
        default=1.0, ge=0.5, le=2.0,
        description="Speech speed multiplier"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the audio response"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Your cab will arrive in 10 minutes.",
                },
                {
                    "text": "உங்கள் கேப் 10 நிமிடங்களில் வரும்.",
                },
                {
                    "text": "உங்கள் pickup location எங்கே?",
                    "stream": True,
                },
            ]
        }
    }


class TTSHealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    uptime_seconds: float


class TTSModelInfo(BaseModel):
    model_name: str
    device: str
    ready: bool
    active_requests: int
    stats: dict


class TTSErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None


class TTSNormalizedRequest(BaseModel):
    original_text: str
    normalized_text: str
    language: str
    segments: list[dict] | None = None
