#!/usr/bin/env python3
"""
Cost Analysis for TTS Service
Estimates infrastructure cost per minute of generated audio.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class HardwareConfig:
    name: str
    gpu_count: int = 0
    gpu_type: str = ""
    vcpus: int = 0
    ram_gb: float = 0
    monthly_cost_usd: float = 0
    gpu_cost_per_hour: float = 0

    @property
    def cost_per_hour(self) -> float:
        return self.monthly_cost_usd / 730

    @property
    def cost_per_minute(self) -> float:
        return self.cost_per_hour / 60


@dataclass
class CostEstimate:
    hardware: str
    concurrency: int
    requests_per_second: float
    audio_minutes_per_hour: float
    cost_per_hour: float
    cost_per_minute: float
    cost_per_audio_minute: float
    rtf: float

    def to_dict(self) -> dict:
        return asdict(self)


HARDWARE_CONFIGS = [
    HardwareConfig(
        name="CPU-only (2 vCPU, 4GB RAM)",
        vcpus=2, ram_gb=4, monthly_cost_usd=15.0,
    ),
    HardwareConfig(
        name="CPU-only (4 vCPU, 8GB RAM)",
        vcpus=4, ram_gb=8, monthly_cost_usd=30.0,
    ),
    HardwareConfig(
        name="GPU (1x T4, 4 vCPU, 16GB RAM)",
        gpu_count=1, gpu_type="T4", vcpus=4, ram_gb=16,
        monthly_cost_usd=180.0,
    ),
    HardwareConfig(
        name="GPU (1x A10G, 8 vCPU, 32GB RAM)",
        gpu_count=1, gpu_type="A10G", vcpus=8, ram_gb=32,
        monthly_cost_usd=350.0,
    ),
    HardwareConfig(
        name="GPU (1x A100, 12 vCPU, 64GB RAM)",
        gpu_count=1, gpu_type="A100", vcpus=12, ram_gb=64,
        monthly_cost_usd=1500.0,
    ),
]


def estimate_cost(
    hardware: HardwareConfig,
    concurrency: int,
    rtf: float,
    avg_request_duration: float,
    utilization: float = 0.7,
) -> CostEstimate:
    effective_throughput = concurrency / max(rtf, 0.01) * utilization
    audio_minutes_per_hour = effective_throughput * avg_request_duration / 60

    requests_per_second = effective_throughput / avg_request_duration if avg_request_duration > 0 else 0

    cost_per_hour = hardware.cost_per_hour
    cost_per_minute = hardware.cost_per_minute
    cost_per_audio_minute = cost_per_hour / audio_minutes_per_hour if audio_minutes_per_hour > 0 else float('inf')

    return CostEstimate(
        hardware=hardware.name,
        concurrency=concurrency,
        requests_per_second=round(requests_per_second, 2),
        audio_minutes_per_hour=round(audio_minutes_per_hour, 1),
        cost_per_hour=round(cost_per_hour, 2),
        cost_per_minute=round(cost_per_minute, 4),
        cost_per_audio_minute=round(cost_per_audio_minute, 4),
        rtf=round(rtf, 3),
    )


def run_cost_analysis(
    rtf: float = 1.0,
    avg_request_duration: float = 5.0,
    utilizations: list[float] | None = None,
    output_path: str = "cost_analysis.json",
) -> dict:
    if utilizations is None:
        utilizations = [0.5, 0.7, 0.9]

    results = {
        'parameters': {
            'rtf': rtf,
            'avg_request_duration_s': avg_request_duration,
            'utilizations': utilizations,
        },
        'hardware_configs': [asdict(h) for h in HARDWARE_CONFIGS],
        'estimates': {},
    }

    for hw in HARDWARE_CONFIGS:
        hw_estimates = []
        for concurrency in [1, 5, 10, 15, 20]:
            for util in utilizations:
                estimate = estimate_cost(
                    hw, concurrency, rtf, avg_request_duration, util
                )
                hw_estimates.append({
                    'concurrency': concurrency,
                    'utilization': util,
                    **estimate.to_dict(),
                })
        results['estimates'][hw.name] = hw_estimates

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Cost analysis saved to {output_file}")
    return results


def print_summary(results: dict):
    print("\n" + "=" * 80)
    print("COST ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"RTF: {results['parameters']['rtf']}")
    print(f"Avg request duration: {results['parameters']['avg_request_duration_s']}s")
    print()

    for hw_name, estimates in results['estimates'].items():
        print(f"\n--- {hw_name} ---")
        for est in estimates:
            if est['utilization'] == 0.7:
                print(
                    f"  Concurrency={est['concurrency']:2d}: "
                    f"Audio min/hr={est['audio_minutes_per_hour']:6.1f}, "
                    f"Cost/hr=${est['cost_per_hour']:6.2f}, "
                    f"Cost/audio min=${est['cost_per_audio_minute']:6.4f}, "
                    f"RPS={est['requests_per_second']:5.2f}"
                )


def main():
    parser = argparse.ArgumentParser(description="TTS Cost Analysis")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor")
    parser.add_argument("--avg-duration", type=float, default=5.0, help="Avg audio duration (s)")
    parser.add_argument("--output", default="cost_analysis.json")
    args = parser.parse_args()

    results = run_cost_analysis(
        rtf=args.rtf,
        avg_request_duration=args.avg_duration,
        output_path=args.output,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
