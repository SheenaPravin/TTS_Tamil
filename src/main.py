from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from .config.settings import get_settings
from .tts_engine.vits_engine import CoquiTTSEngine, MockTTSEngine
from .tts_engine.streaming import StreamingSynthesizer
from .text_normalization.normalizer import TextNormalizer
from .api.routes import router, init_routes

settings = get_settings()

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

app = FastAPI(
    title="Tamil/English/Tanglish TTS Service",
    description=(
        "Self-hosted Text-to-Speech service for Tamil, English, "
        "and Tanglish (Tamil-English code-mixed) speech generation. "
        "Optimized for real-time voice agent applications in "
        "transportation and taxi contact centers."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_engine = None
normalizer = None
streaming_synth = None


@app.on_event("startup")
async def startup_event():
    global tts_engine, normalizer, streaming_synth

    logger.info("Starting TTS service...")
    logger.info(f"Settings: host={settings.host}, port={settings.port}")
    logger.info(f"Model: {settings.model_name}, device={settings.resolve_device()}")

    normalizer = TextNormalizer(lang='mixed', context='transport')

    default_speaker = str(Path(__file__).parent.parent / "models" / "default_speaker.wav")

    use_mock = os.environ.get('TTS_USE_MOCK', 'false').lower() == 'true'
    if use_mock:
        logger.warning("Using Mock TTS engine (for development/testing)")
        tts_engine = MockTTSEngine(sample_rate=settings.sample_rate)
    else:
        tts_engine = CoquiTTSEngine(
            model_name=settings.model_name,
            device=settings.resolve_device(),
            max_concurrent=settings.max_concurrent,
            cache_dir=settings.cache_dir,
            speaker_wav=default_speaker,
        )

    loop = __import__('asyncio').get_event_loop()
    await loop.run_in_executor(None, tts_engine.initialize)

    streaming_synth = StreamingSynthesizer(tts_engine, normalizer)

    init_routes(tts_engine, normalizer, streaming_synth)

    logger.info(f"TTS service ready. Model ready: {tts_engine.is_ready}")


@app.on_event("shutdown")
async def shutdown_event():
    global tts_engine
    if tts_engine:
        tts_engine.cleanup()
    logger.info("TTS service shut down")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent.parent / "docs" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>TTS Tamil</h1><p>Dashboard not found. <a href='/docs'>Open API Docs</a></p>"
    )


app.include_router(router)


def run():
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    run()
