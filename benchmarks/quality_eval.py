#!/usr/bin/env python3
"""
Quality Evaluation for TTS output.
Provides tools for both automated metrics and human evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PronunciationTestCase:
    text: str
    language: str
    expected_output: str | None = None
    category: str = 'general'
    description: str = ''


PRONUNCIATION_TEST_SUITE = [
    PronunciationTestCase(
        text="Your booking ID is TN45AB1234.",
        language='en',
        expected_output="T N 4 5 A B 1 2 3 4",
        category='booking_id',
        description="Booking ID should be spelled out character by character",
    ),
    PronunciationTestCase(
        text="Your OTP is 4821.",
        language='en',
        expected_output="four eight two one",
        category='otp',
        description="OTP digits should be spoken individually",
    ),
    PronunciationTestCase(
        text="Your phone number is 9876543210.",
        language='en',
        expected_output="nine eight seven six five four three two one zero",
        category='phone',
        description="Phone number digits should be spoken individually",
    ),
    PronunciationTestCase(
        text="Your cab will arrive at 7:30 PM.",
        language='en',
        description="Time should be spoken naturally",
        category='time',
    ),
    PronunciationTestCase(
        text="The fare is 450 rupees.",
        language='en',
        description="Price should be spoken as natural number",
        category='price',
    ),
    PronunciationTestCase(
        text="The distance is 12.5 kilometers.",
        language='en',
        description="Distance should be spoken naturally",
        category='distance',
    ),
    PronunciationTestCase(
        text="உங்கள் கேப் 10 நிமிடங்களில் வரும்.",
        language='ta',
        description="Tamil sentence with number",
        category='tamil_basic',
    ),
    PronunciationTestCase(
        text="உங்கள் OTP 4821.",
        language='ta',
        description="Tamil sentence with OTP",
        category='tamil_otp',
    ),
    PronunciationTestCase(
        text="உங்கள் pickup location எங்கே?",
        language='mixed',
        description="Tanglish code-mixed question",
        category='tanglish_basic',
    ),
    PronunciationTestCase(
        text="Driver வருகிறார், please wait பண்ணுங்க.",
        language='mixed',
        description="Tanglish code-mixed instruction",
        category='tanglish_instruction',
    ),
    PronunciationTestCase(
        text="உங்கள் கேப் Chennai Central-ல இருக்கா.",
        language='mixed',
        description="Tanglish with location",
        category='tanglish_location',
    ),
    PronunciationTestCase(
        text="Your cab will arrive in 10 minutes.",
        language='en',
        description="Basic English sentence",
        category='english_basic',
    ),
]


@dataclass
class QualityReport:
    total_test_cases: int = 0
    pronunciation_accuracy: float = 0.0
    categories_tested: list[str] = field(default_factory=list)
    language_coverage: dict[str, int] = field(default_factory=dict)
    human_evaluation_template: list[dict] = field(default_factory=list)
    mos_scale: str = "1-5 (1=Bad, 2=Poor, 3=Fair, 4=Good, 5=Excellent)"

    def to_dict(self) -> dict:
        return asdict(self)


def generate_human_eval_template(test_cases: list[PronunciationTestCase]) -> list[dict]:
    template = []
    for i, tc in enumerate(test_cases):
        template.append({
            'id': i + 1,
            'text': tc.text,
            'language': tc.language,
            'category': tc.category,
            'description': tc.description,
            'scores': {
                'pronunciation_accuracy': None,
                'naturalness': None,
                'intelligibility': None,
                'prosody': None,
                'code_switching_quality': None,
            },
            'notes': '',
        })
    return template


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)


def calculate_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[len(ref_chars)][len(hyp_chars)] / max(len(ref_chars), 1)


def run_evaluation(
    test_cases: list[PronunciationTestCase] | None = None,
    output_path: str = "quality_report.json",
):
    if test_cases is None:
        test_cases = PRONUNCIATION_TEST_SUITE

    report = QualityReport(
        total_test_cases=len(test_cases),
        categories_tested=list(set(tc.category for tc in test_cases)),
        language_coverage={
            lang: sum(1 for tc in test_cases if tc.language == lang)
            for lang in set(tc.language for tc in test_cases)
        },
        human_evaluation_template=generate_human_eval_template(test_cases),
    )

    output = {
        'report': report.to_dict(),
        'test_cases': [
            {
                'text': tc.text,
                'language': tc.language,
                'category': tc.category,
                'description': tc.description,
            }
            for tc in test_cases
        ],
        'mos_scale': report.mos_scale,
        'evaluation_instructions': {
            'pronunciation_accuracy': 'How accurately are individual words and phonemes pronounced? (1-5)',
            'naturalness': 'How natural does the speech sound overall? (1-5)',
            'intelligibility': 'How easily can the speech be understood? (1-5)',
            'prosody': 'How appropriate is the rhythm, stress, and intonation? (1-5)',
            'code_switching_quality': 'For Tanglish: How natural is the transition between Tamil and English? (1-5)',
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Quality evaluation template saved to {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description="TTS Quality Evaluation")
    parser.add_argument("--output", default="quality_report.json")
    parser.add_argument("--wer", nargs=2, metavar=("REFERENCE", "HYPOTHESIS"))
    args = parser.parse_args()

    if args.wer:
        ref, hyp = args.wer
        wer = calculate_wer(ref, hyp)
        cer = calculate_cer(ref, hyp)
        print(f"WER: {wer:.4f}")
        print(f"CER: {cer:.4f}")
    else:
        run_evaluation(output_path=args.output)


if __name__ == "__main__":
    main()
