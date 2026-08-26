from __future__ import annotations

import re
from .number_normalizer import (
    number_to_english,
    number_to_tamil,
    digits_to_english_spoken,
    digits_to_tamil_spoken,
    ENGLISH_CARDINAL,
)
from .tamil_normalizer import (
    has_tamil_script,
    romanized_tamil_to_tamil,
    has_romanized_tamil,
    detect_language_segments,
)

BOOKING_ID_PATTERN = re.compile(
    r'\b([A-Z]{2}\d{2}[A-Z]{2}\d{4})\b'
    r'|\b([A-Z]{2}\d{6,})\b'
    r'|\b(BOOK[-\s]?\d{4,})\b',
    re.IGNORECASE,
)

OTP_PATTERN = re.compile(
    r'\b(?:OTP\s*(?:is\s*)?[:=]?\s*)?(\d{4,6})\b',
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r'(?:(?:\+91|91)[\s\-]?)?'
    r'([6-9]\d{9})\b'
)

TIME_PATTERN = re.compile(
    r'\b(\d{1,2})\s*[:\.]\s*(\d{2})\s*(AM|PM|am|pm)?\b'
)

DATE_PATTERN = re.compile(
    r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
)

PRICE_PATTERN = re.compile(
    r'(?:Rs\.?|₹|INR)\s*(\d+(?:,\d+)*(?:\.\d{1,2})?)'
    r'|(\d+(?:,\d+)*(?:\.\d{1,2})?)\s*(?:Rs\.?|₹|INR|rupees?)',
    re.IGNORECASE,
)

DISTANCE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometres|Km)',
    re.IGNORECASE,
)

VEHICLE_TYPES = {
    'auto': 'auto rickshaw',
    'sedan': 'sedan',
    'suv': 'S U V',
    'hatchback': 'hatchback',
    'cab': 'cab',
    'bike': 'bike',
    'taxi': 'taxi',
    'premium': 'premium cab',
    'mini': 'mini cab',
    'pool': 'pool',
    'share': 'share',
}


class TextNormalizer:
    def __init__(self, lang: str = 'mixed', context: str = 'transport'):
        self.lang = lang
        self.context = context

    def normalize(self, text: str) -> str:
        text = self._normalize_booking_ids(text)
        text = self._normalize_otps(text)
        text = self._normalize_phones(text)
        text = self._normalize_times(text)
        text = self._normalize_dates(text)
        text = self._normalize_prices(text)
        text = self._normalize_distances(text)
        text = self._normalize_vehicles(text)
        text = self._normalize_abbreviations(text)
        text = self._normalize_numbers(text)
        text = self._normalize_romanized_tamil(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_for_tts(self, text: str, target_lang: str = 'en') -> str:
        normalized = self.normalize(text)
        return normalized

    def _normalize_booking_ids(self, text: str) -> str:
        def replace_booking_id(match: re.Match) -> str:
            full = match.group(0)
            alpha_num = re.sub(r'[^A-Za-z0-9]', '', full)
            return ' '.join(alpha_num).upper()

        return BOOKING_ID_PATTERN.sub(replace_booking_id, text)

    def _normalize_otps(self, text: str) -> str:
        def replace_otp(match: re.Match) -> str:
            digits = match.group(1)
            prefix_match = re.search(
                r'(OTP\s*(?:is\s*)?[:=]?\s*)',
                text[:match.start() + len(match.group(0))],
                re.IGNORECASE,
            )
            prefix = prefix_match.group(1) if prefix_match else ''
            spoken = digits_to_english_spoken(digits)
            return prefix + spoken

        return OTP_PATTERN.sub(replace_otp, text)

    def _normalize_phones(self, text: str) -> str:
        def replace_phone(match: re.Match) -> str:
            number = match.group(1)
            return digits_to_english_spoken(number)

        return PHONE_PATTERN.sub(replace_phone, text)

    def _normalize_times(self, text: str) -> str:
        def replace_time(match: re.Match) -> str:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3)

            if ampm:
                ampm_lower = ampm.lower()
                if ampm_lower == 'am' and hour == 12:
                    hour = 0
                elif ampm_lower == 'pm' and hour != 12:
                    hour += 12

            if self._is_tamil_context(text):
                if hour == 0:
                    hour_text = 'பன்னிரண்டு'
                elif hour <= 12:
                    hour_text = number_to_tamil(hour)
                else:
                    hour_text = number_to_tamil(hour - 12 if hour > 12 else hour)
                minute_text = number_to_tamil(minute) if minute > 0 else ''
                ampm_tamil = 'காலை' if hour < 12 else 'மாலை'
                if minute > 0:
                    return f'{hour_text} மணி {minute_text} நிமிடம் {ampm_tamil}'
                return f'{hour_text} மணி {ampm_tamil}'
            else:
                hour_text = number_to_english(hour)
                if minute == 0:
                    return f"{hour_text} o'clock"
                minute_text = number_to_english(minute)
                ampm_str = f' {ampm}' if ampm else ''
                return f'{hour_text} {minute_text}{ampm_str}'

        return TIME_PATTERN.sub(replace_time, text)

    def _normalize_dates(self, text: str) -> str:
        def replace_date(match: re.Match) -> str:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            if year < 100:
                year += 2000

            months_en = [
                '', 'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            months_ta = [
                '', 'ஜனவரி', 'பிப்ரவரி', 'மார்ச்', 'ஏப்ரல்', 'மே', 'ஜூன்',
                'ஜூலை', 'ஆகஸ்ட்', 'செப்டம்பர்', 'அக்டோபர்', 'நவம்பர்', 'டிசம்பர்'
            ]

            if 1 <= month <= 12 and 1 <= day <= 31:
                if self._is_tamil_context(text):
                    return f'{months_ta[month]} {number_to_tamil(day)}, {number_to_tamil(year)}'
                return f'{months_en[month]} {number_to_english(day)}, {year}'
            return match.group(0)

        return DATE_PATTERN.sub(replace_date, text)

    def _normalize_prices(self, text: str) -> str:
        def replace_price(match: re.Match) -> str:
            amount_str = (match.group(1) or match.group(2)).replace(',', '')
            amount = float(amount_str)
            amount_int = int(amount)

            if self._is_tamil_context(text):
                return f'{number_to_tamil(amount_int)} ரூபாய்'
            return f'{number_to_english(amount_int)} rupees'

        return PRICE_PATTERN.sub(replace_price, text)

    def _normalize_distances(self, text: str) -> str:
        def replace_distance(match: re.Match) -> str:
            dist_str = match.group(1)
            dist = float(dist_str)
            dist_int = int(dist)

            if self._is_tamil_context(text):
                if dist == dist_int:
                    return f'{number_to_tamil(dist_int)} கிலோமீட்டர்'
                return f'{dist_str} கிலோமீட்டர்'
            if dist == dist_int:
                return f'{number_to_english(dist_int)} kilometers'
            return f'{dist_str} kilometers'

        return DISTANCE_PATTERN.sub(replace_distance, text)

    def _normalize_vehicles(self, text: str) -> str:
        for vtype, spoken in VEHICLE_TYPES.items():
            pattern = re.compile(rf'\b{re.escape(vtype)}s?\b', re.IGNORECASE)
            text = pattern.sub(spoken, text)
        return text

    def _normalize_abbreviations(self, text: str) -> str:
        abbreviations = {
            'OTP': 'O T P',
            'PIN': 'P I N',
            'GPS': 'G P S',
            'ID': 'I D',
            'SMS': 'S M S',
            'App': 'app',
            'app': 'app',
            'km': 'kilometers',
            'KM': 'kilometers',
            'mins': 'minutes',
            'min': 'minute',
        }
        for abbr, expansion in abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbr)}\b', expansion, text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match: re.Match) -> str:
            num_str = match.group(0)
            if num_str.endswith(('st', 'nd', 'rd', 'th')):
                return number_to_english(int(num_str[:-2]))
            num = int(num_str)
            if self._is_tamil_context(text):
                return number_to_tamil(num)
            return number_to_english(num)

        text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', replace_number, text)
        text = re.sub(r'(?<!\d)\b\d{1,6}\b(?!\d)', replace_number, text)
        return text

    def _normalize_romanized_tamil(self, text: str) -> str:
        if has_romanized_tamil(text):
            words = text.split()
            normalized_words = []
            for word in words:
                cleaned = re.sub(r'[^\w]', '', word)
                if has_romanized_tamil(cleaned):
                    tamil_word = romanized_tamil_to_tamil(cleaned)
                    suffix = word[len(cleaned):] if len(word) > len(cleaned) else ''
                    normalized_words.append(tamil_word + suffix)
                else:
                    normalized_words.append(word)
            return ' '.join(normalized_words)
        return text

    def _is_tamil_context(self, text: str) -> bool:
        tamil_ratio = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF') / max(len(text), 1)
        return tamil_ratio > 0.3 or self.lang == 'ta'
