# Tamil/English/Tanglish Text-to-Speech System

A self-hosted, low-latency Text-to-Speech (TTS) system for **Tamil**, **English**, and **Tanglish** (Tamil-English code-mixed) speech generation. Optimized for real-time voice agent applications in transportation and taxi contact centers.

## Features

- **Multi-language TTS**: Tamil, English, and Tanglish code-mixed speech
- **Context-aware text normalization**: Handles booking IDs, OTPs, phone numbers, times, prices, distances, and abbreviations
- **Streaming API**: Time-to-first-audio (TTFA) optimized streaming synthesis
- **Concurrent requests**: Supports 15-20 concurrent TTS requests
- **Transportation-focused**: Pre-built normalization for taxi/ride-hailing domain

## Architecture

```
Client Request
     │
     ▼
┌─────────────────────────────────────────────┐
│              FastAPI Server                  │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ API Routes    │  │ Streaming Middleware  │ │
│  └──────┬───────┘  └──────────┬───────────┘ │
│         │                     │              │
│  ┌──────▼─────────────────────▼───────────┐ │
│  │        Text Normalization Pipeline      │ │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ │ │
│  │  │ Numbers  │ │ Booking  │ │ Tanglish│ │ │
│  │  │ & Time   │ │ IDs/OTP  │ │ Processor│ │ │
│  │  └─────────┘ └──────────┘ └─────────┘ │ │
│  └───────────────────┬────────────────────┘ │
│                      │                      │
│  ┌───────────────────▼────────────────────┐ │
│  │          TTS Engine Layer               │ │
│  │  ┌──────────┐ ┌──────────┐ ┌────────┐ │ │
│  │  │ Coqui    │ │ Code-    │ │Streaming│ │ │
│  │  │ TTS/VITS │ │ Switching│ │ Buffer │ │ │
│  │  └──────────┘ └──────────┘ └────────┘ │ │
│  └───────────────────┬────────────────────┘ │
│                      │                      │
│  ┌───────────────────▼────────────────────┐ │
│  │       Output (WAV / Stream)            │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Project Structure

```
TTS_Tamil/
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── config/
│   │   └── settings.py            # Configuration management
│   ├── text_normalization/
│   │   ├── normalizer.py          # Main normalization pipeline
│   │   ├── number_normalizer.py   # Number/phone/OTP conversion
│   │   └── tamil_normalizer.py    # Tamil language processing
│   ├── tts_engine/
│   │   ├── engine.py              # TTS engine abstraction
│   │   ├── vits_engine.py         # Coqui TTS / VITS implementation
│   │   ├── code_switch.py         # Tanglish code-switching processor
│   │   └── streaming.py           # Streaming audio synthesis
│   └── api/
│       ├── routes.py              # API endpoints
│       └── models.py              # Request/response schemas
├── benchmarks/
│   ├── benchmark.py               # Performance benchmarking
│   ├── quality_eval.py            # Quality evaluation templates
│   └── cost_analysis.py           # Infrastructure cost analysis
├── tests/
│   ├── test_normalizer.py         # Text normalization tests
│   ├── test_tts_engine.py         # TTS engine tests
│   ├── test_api.py                # API integration tests
│   └── test_integration.py        # End-to-end pipeline tests
├── scripts/
│   ├── setup.sh                   # Environment setup
│   └── download_models.sh         # Model download script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for GPU acceleration
- (Optional) Docker

### Installation

```bash
# Clone the repository
git clone https://github.com/SheenaPravin/TTS_Tamil.git
cd TTS_Tamil

# Setup environment
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Running the Server

```bash
# Development mode (mock engine, no GPU required)
TTS_USE_MOCK=true python -m src.main

# Production mode (requires model download)
python -m src.main
```

The server starts at `http://localhost:8000`.

### Docker Deployment

```bash
# Build and run
docker-compose up tts-service

# With mock engine (no GPU)
TTS_USE_MOCK=true docker-compose up tts-service

# Run benchmarks
docker-compose --profile benchmark run tts-benchmark
```

## API Reference

### POST `/api/v1/synthesize`

Synthesize text to speech.

**Request:**
```json
{
  "text": "Your cab will arrive in 10 minutes.",
  "language": "en",
  "stream": false,
  "temperature": 0.7,
  "speed": 1.0
}
```

**Language options:**
- `"en"` - English
- `"ta"` - Tamil
- `"mixed"` - Tanglish (code-mixed)

**Response:** WAV audio file with metadata headers:
- `X-Request-ID`: Unique request identifier
- `X-Latency-Ms`: Total synthesis latency in milliseconds
- `X-First-Chunk-Latency-Ms`: Time to first audio byte
- `X-Audio-Duration`: Duration of generated audio in seconds

**Streaming mode:** Set `"stream": true` for chunked WAV streaming.

### POST `/api/v1/normalize`

Preview text normalization output.

```json
{
  "text": "Your OTP is 4821.",
  "language": "en"
}
```

**Response:**
```json
{
  "original_text": "Your OTP is 4821.",
  "normalized_text": "Your OTP is four eight two one.",
  "language": "en"
}
```

### GET `/api/v1/health`

Health check endpoint.

### GET `/api/v1/model`

Model information and statistics.

## Text Normalization

The system handles transportation-specific text normalization:

| Input | Output |
|-------|--------|
| `TN45AB1234` | `T N 4 5 A B 1 2 3 4` (spelled out) |
| `OTP 4821` | `four eight two one` (digits individually) |
| `9876543210` | `nine eight seven six five four three two one zero` |
| `7:30 PM` | `seven thirty PM` (natural time) |
| `Rs 450` | `four hundred fifty rupees` |
| `12 km` | `twelve kilometers` |
| `auto` | `auto rickshaw` |

## Benchmarking

```bash
# Run performance benchmarks
python benchmarks/benchmark.py --url http://localhost:8000

# Run with different concurrency levels
python benchmarks/benchmark.py --concurrency 1 5 10 15 20

# Generate quality evaluation template
python benchmarks/quality_eval.py --output quality_report.json

# Run cost analysis
python benchmarks/cost_analysis.py --rtf 0.8 --avg-duration 5
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_normalizer.py -v

# Run integration tests
python tests/test_integration.py
```

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_HOST` | `0.0.0.0` | Server host |
| `TTS_PORT` | `8000` | Server port |
| `TTS_MODEL_NAME` | `tts_models/multilingual/multi-dataset/xtts_v2` | Model to use |
| `TTS_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda` |
| `TTS_MAX_CONCURRENT` | `20` | Max concurrent requests |
| `TTS_USE_MOCK` | `false` | Use mock engine for testing |

## Performance Targets

| Metric | Target |
|--------|--------|
| p99 TTS latency | ≤ 500 ms |
| Concurrent requests | 15-20 |
| Time-to-first-audio | < 300 ms |
| Deployment model | Self-hosted |
| Third-party TTS API | Not required |

## Model Options

### Primary: XTTS v2 (Coqui TTS)
- Multilingual including Tamil support
- Voice cloning with reference audio
- Streaming synthesis
- High quality code-mixed speech

### Alternatives Investigated
- **MMS-TTS (Meta)**: Massively Multilingual Speech
- **Bark**: Generative audio model
- **StyleTTS2**: Style-based TTS
- **VITS**: End-to-end TTS

## License

This project uses open-source models and frameworks. See individual model licenses for specific terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit a pull request
