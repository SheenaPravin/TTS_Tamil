from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from ..text_normalization.tamil_normalizer import (
    has_tamil_script,
    detect_language_segments,
    romanized_tamil_to_tamil,
    has_romanized_tamil,
)


@dataclass
class LanguageSegment:
    text: str
    language: str
    start: int
    end: int


class TanglishProcessor:
    TANGLED_PATTERNS = {
        'pickup': 'pickup',
        'drop': 'drop',
        'ride': 'ride',
        'cab': 'cab',
        'driver': 'driver',
        'trip': 'trip',
        'fare': 'fare',
        'book': 'book',
        'cancel': 'cancel',
        'confirm': 'confirm',
        'start': 'start',
        'stop': 'stop',
        'arrive': 'arrive',
        'location': 'location',
        'address': 'address',
        'route': 'route',
        'traffic': 'traffic',
        'wait': 'wait',
        'reach': 'reach',
        'call': 'call',
        'pay': 'pay',
        'cash': 'cash',
        'online': 'online',
        'night': 'night',
        'morning': 'morning',
        'charge': 'charge',
        'extra': 'extra',
        'rate': 'rate',
        'review': 'review',
        'feedback': 'feedback',
        'customer': 'customer',
        'service': 'service',
        'support': 'support',
        'company': 'company',
        'office': 'office',
        'help': 'help',
        'number': 'number',
        'ID': 'ID',
        'check': 'check',
        'number': 'number',
        'OTP': 'OTP',
        'GPS': 'GPS',
        'minutes': 'minutes',
        'km': 'km',
        'kilometers': 'kilometers',
        'done': 'done',
        'okay': 'okay',
        'sorry': 'sorry',
        'thank': 'thank',
        'please': 'please',
        'welcome': 'welcome',
    }

    def segment_tanglish(self, text: str) -> list[LanguageSegment]:
        segments: list[LanguageSegment] = []
        tokens = re.finditer(
            r'([\u0B80-\u0BFF]+(?:\s*[\u0B80-\u0BFF]+)*)'
            r'|([A-Za-z]+(?:\s*[A-Za-z]+)*)'
            r'|([0-9:.\-/+]+)'
            r'|(\s+)'
            r'|([^\s\w]+)',
            text,
            re.UNICODE,
        )

        pos = 0
        for m in tokens:
            start = m.start()
            end = m.end()

            if m.group(1):
                segments.append(LanguageSegment(
                    text=m.group(1), language='tamil',
                    start=start, end=end,
                ))
            elif m.group(2):
                word = m.group(2).strip()
                lang = self._classify_english_word(word)
                segments.append(LanguageSegment(
                    text=m.group(2), language=lang,
                    start=start, end=end,
                ))
            elif m.group(3):
                segments.append(LanguageSegment(
                    text=m.group(3), language='number',
                    start=start, end=end,
                ))
            elif m.group(4):
                segments.append(LanguageSegment(
                    text=m.group(4), language='space',
                    start=start, end=end,
                ))
            elif m.group(5):
                segments.append(LanguageSegment(
                    text=m.group(5), language='punct',
                    start=start, end=end,
                ))

        return segments

    def _classify_english_word(self, word: str) -> str:
        if word.lower() in self.TANGLED_PATTERNS:
            return 'english'
        return 'english'

    def generate_sequential_text(
        self, text: str, separator: str = ' '
    ) -> dict[str, list[str]]:
        segments = self.segment_tanglish(text)
        tamil_parts: list[str] = []
        english_parts: list[str] = []
        mixed_parts: list[str] = []

        for seg in segments:
            if seg.language == 'tamil':
                tamil_parts.append(seg.text)
                mixed_parts.append(seg.text)
            elif seg.language == 'english':
                english_parts.append(seg.text)
                mixed_parts.append(seg.text)
            elif seg.language == 'number':
                mixed_parts.append(seg.text)
            elif seg.language == 'space':
                if mixed_parts and mixed_parts[-1] != ' ':
                    mixed_parts.append(' ')

        return {
            'tamil': tamil_parts,
            'english': english_parts,
            'mixed': mixed_parts,
            'full': mixed_parts,
        }

    def detect_primary_language(self, text: str) -> str:
        segments = self.segment_tanglish(text)
        tamil_chars = 0
        english_chars = 0
        for seg in segments:
            if seg.language == 'tamil':
                tamil_chars += len(seg.text)
            elif seg.language == 'english':
                english_chars += len(seg.text)

        total = tamil_chars + english_chars
        if total == 0:
            return 'en'
        if tamil_chars / total > 0.6:
            return 'ta'
        if english_chars / total > 0.6:
            return 'en'
        return 'mixed'
