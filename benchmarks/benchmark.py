#!/usr/bin/env python3
"""
Performance Benchmark for TTS Service
Measures latency, throughput, concurrency, and resource utilization.
"""
from __future__ import annotations

import asyncio
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    first_chunk_latencies_ms: list[float] = field(default_factory=list)
    audio_durations: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    concurrency_level: int = 0
    test_duration_seconds: float = 0.0

    @property
    def p50(self) -> float:
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0

    @property
    def p95(self) -> float:
        return np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0

    @property
    def p99(self) -> float:
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def avg_first_chunk(self) -> float:
        return statistics.mean(self.first_chunk_latencies_ms) if self.first_chunk_latencies_ms else 0

    @property
    def avg_audio_duration(self) -> float:
        return statistics.mean(self.audio_durations) if self.audio_durations else 0

    @property
    def total_audio_duration(self) -> float:
        return sum(self.audio_durations)

    @property
    def requests_per_second(self) -> float:
        return self.successful_requests / self.test_duration_seconds if self.test_duration_seconds > 0 else 0

    @property
    def rtf(self) -> float:
        if not self.audio_durations or not self.latencies_ms:
            return 0
        return self.avg_latency / (self.avg_audio_duration * 1000) if self.avg_audio_duration > 0 else 0

    def to_dict(self) -> dict:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'concurrency_level': self.concurrency_level,
            'test_duration_seconds': round(self.test_duration_seconds, 2),
            'latency': {
                'p50_ms': round(self.p50, 1),
                'p95_ms': round(self.p95, 1),
                'p99_ms': round(self.p99, 1),
                'avg_ms': round(self.avg_latency, 1),
                'max_ms': round(max(self.latencies_ms), 1) if self.latencies_ms else 0,
                'min_ms': round(min(self.latencies_ms), 1) if self.latencies_ms else 0,
            },
            'first_chunk_latency': {
                'avg_ms': round(self.avg_first_chunk, 1),
                'p95_ms': round(np.percentile(self.first_chunk_latencies_ms, 95), 1) if self.first_chunk_latencies_ms else 0,
            },
            'audio': {
                'avg_duration_s': round(self.avg_audio_duration, 2),
                'total_duration_s': round(self.total_audio_duration, 2),
            },
            'throughput': {
                'requests_per_second': round(self.requests_per_second, 2),
                'real_time_factor': round(self.rtf, 3),
            },
            'errors': self.errors[:10],
        }


TEST_SENTENCES = {
    'en': [
        "Your cab will arrive in 10 minutes.",
        "Your booking ID is TN45AB1234.",
        "Your OTP is 4821.",
        "Your phone number is 9876543210.",
        "Your cab will arrive at 7:30 PM.",
        "The fare is 450 rupees.",
        "The distance is 12.5 kilometers.",
        "Please confirm your pickup location.",
        "Your driver Mr. Kumar is on the way.",
        "Thank you for using our service.",
    ],
    'ta': [
        "உங்கள் கேப் 10 நிமிடங்களில் வரும்.",
        "உங்கள் புக்கிங் ID TN45AB1234.",
        "உங்கள் OTP 4821.",
        "உங்கள் கேப் மாலை 7 மணி 30 நிமிடத்திற்கு வரும்.",
        "கட்டணம் 450 ரூபாய்.",
        "தூரம் 12.5 கிலோமீட்டர்.",
        "தயவுசெய்து உங்கள் பிக்கப் இடத்தை உறுதிப்படுத்துங்கள்.",
        "உங்கள் டிரைவர் குமார் வருகிறார்.",
        "எங்கள் சேவையைப் பயன்படுத்தியதற்கு நன்றி.",
    ],
    'mixed': [
        "உங்கள் pickup location எங்கே?",
        "உங்கள் கேப் 10 minutes-ல் வரும்.",
        "Your booking ID TN45AB1234.",
        "OTP 4821 enter பண்ணுங்க.",
        "Driver வருகிறார், please wait பண்ணுங்க.",
        "Fare 450 rupees.",
        "Distance 12.5 kilometers.",
        "Drop location எங்கே?",
    ],
}


async def single_request(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    language: str,
    stream: bool = False,
) -> dict:
    start = time.time()
    try:
        payload = {
            "text": text,
            "language": language,
            "stream": stream,
        }
        if stream:
            async with client.stream(
                "POST", f"{base_url}/api/v1/synthesize",
                json=payload, timeout=30.0,
            ) as response:
                if response.status_code != 200:
                    return {"error": f"HTTP {response.status_code}", "latency_ms": 0}
                first_chunk_time = None
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    total_bytes += len(chunk)
                latency_ms = (time.time() - start) * 1000
                first_chunk_ms = (first_chunk_time - start) * 1000 if first_chunk_time else latency_ms
                return {
                    "latency_ms": latency_ms,
                    "first_chunk_ms": first_chunk_ms,
                    "bytes": total_bytes,
                    "status": "success",
                }
        else:
            response = await client.post(
                f"{base_url}/api/v1/synthesize",
                json=payload, timeout=30.0,
            )
            latency_ms = (time.time() - start) * 1000
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}", "latency_ms": latency_ms}
            first_chunk_ms = float(response.headers.get("X-First-Chunk-Latency-Ms", latency_ms))
            duration = float(response.headers.get("X-Audio-Duration", "0"))
            return {
                "latency_ms": latency_ms,
                "first_chunk_ms": first_chunk_ms,
                "audio_duration": duration,
                "bytes": len(response.content),
                "status": "success",
            }
    except Exception as e:
        return {"error": str(e), "latency_ms": (time.time() - start) * 1000}


async def run_benchmark(
    base_url: str,
    concurrency: int,
    requests_per_level: int,
    languages: list[str],
    stream: bool = False,
) -> BenchmarkResult:
    result = BenchmarkResult(concurrency_level=concurrency)

    sentences = []
    for lang in languages:
        if lang in TEST_SENTENCES:
            for i in range(requests_per_level):
                sentences.append(TEST_SENTENCES[lang][i % len(TEST_SENTENCES[lang])])

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(text: str, lang: str):
            async with sem:
                return await single_request(client, base_url, text, lang, stream)

        tasks = [bounded_request(text, lang) for text, lang in zip(
            [s for s in sentences for _ in range(1)],
            [lang for lang in languages for _ in range(requests_per_level)],
        )]
        # Fix: build tasks list properly
        task_list = []
        for i, s in enumerate(sentences):
            lang = languages[i // requests_per_level] if i < len(sentences) else languages[0]
            task_list.append(bounded_request(s, lang))

        results = await asyncio.gather(*task_list, return_exceptions=True)

    for r in results:
        result.total_requests += 1
        if isinstance(r, Exception):
            result.failed_requests += 1
            result.errors.append(str(r))
        elif isinstance(r, dict) and 'error' in r:
            result.failed_requests += 1
            result.errors.append(r['error'])
            result.latencies_ms.append(r.get('latency_ms', 0))
        else:
            result.successful_requests += 1
            result.latencies_ms.append(r['latency_ms'])
            result.first_chunk_latencies_ms.append(r['first_chunk_ms'])
            if 'audio_duration' in r:
                result.audio_durations.append(r['audio_duration'])

    result.test_duration_seconds = time.time() - start_time
    return result


async def health_check(base_url: str) -> bool:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{base_url}/api/v1/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(description="TTS Service Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="TTS service URL")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 5, 10, 15, 20])
    parser.add_argument("--requests", type=int, default=10, help="Requests per concurrency level")
    parser.add_argument("--languages", nargs="+", default=["en", "ta", "mixed"])
    parser.add_argument("--stream", action="store_true", help="Test streaming mode")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()

    print(f"Waiting for TTS service at {args.url}...")
    import time as _time
    for _ in range(30):
        if asyncio.run(health_check(args.url)):
            print("Service is ready!")
            break
        _time.sleep(2)
    else:
        print("Service not available. Using mock mode may be needed.")
        sys.exit(1)

    all_results = {}
    for concurrency in args.concurrency:
        print(f"\n--- Benchmarking with concurrency={concurrency} ---")
        result = asyncio.run(run_benchmark(
            base_url=args.url,
            concurrency=concurrency,
            requests_per_level=args.requests,
            languages=args.languages,
            stream=args.stream,
        ))
        all_results[concurrency] = result.to_dict()
        print(json.dumps(result.to_dict(), indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
