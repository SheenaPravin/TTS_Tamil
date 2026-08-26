import asyncio
import pytest
import httpx

from src.tts_engine.vits_engine import MockTTSEngine
from src.tts_engine.engine import TTSRequest


class TestMockTTSEngine:
    @pytest.fixture
    def engine(self):
        eng = MockTTSEngine()
        eng.initialize()
        return eng

    def test_engine_ready(self, engine):
        assert engine.is_ready

    def test_synthesize(self, engine):
        request = TTSRequest(
            text="Hello world",
            language='en',
            request_id="test-001",
        )
        response = engine.synthesize(request)
        assert response.audio is not None
        assert response.sample_rate == 24000
        assert response.duration > 0
        assert response.request_id == "test-001"

    def test_synthesize_tamil(self, engine):
        request = TTSRequest(
            text="வணக்கம்",
            language='ta',
            request_id="test-002",
        )
        response = engine.synthesize(request)
        assert response.audio is not None
        assert response.duration > 0

    def test_synthesize_mixed(self, engine):
        request = TTSRequest(
            text="Hello வணக்கம் world",
            language='mixed',
            request_id="test-003",
        )
        response = engine.synthesize(request)
        assert response.audio is not None

    def test_model_info(self, engine):
        info = engine.model_info
        assert info['model_name'] == 'mock'
        assert info['ready'] is True

    def test_stats(self, engine):
        request = TTSRequest(text="test", language='en')
        engine.synthesize(request)
        info = engine.model_info
        assert info['stats']['total_requests'] >= 1


class TestStreaming:
    @pytest.fixture
    def engine(self):
        eng = MockTTSEngine()
        eng.initialize()
        return eng

    @pytest.mark.asyncio
    async def test_streaming_synthesis(self, engine):
        request = TTSRequest(
            text="Hello world",
            language='en',
            stream=True,
        )
        chunks = []
        async for chunk in engine.synthesize_streaming(request):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert chunks[-1].is_last is True
