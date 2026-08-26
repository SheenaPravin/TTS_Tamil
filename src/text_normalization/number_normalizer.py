from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

TAMIL_DIGITS = {
    '0': 'பூஜ்யம்', '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று',
    '4': 'நான்கு', '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு',
    '8': 'எட்டு', '9': 'ஒன்பது',
}

TAMIL_PLACE_VALUES = {
    10: 'பத்து',
    100: 'நூறு',
    1000: 'ஆயிரம்',
    100000: 'லட்சம்',
    10000000: 'கோடி',
}

ENGLISH_CARDINAL = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
    10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen',
    14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
    18: 'eighteen', 19: 'nineteen', 20: 'twenty', 30: 'thirty',
    40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy',
    80: 'eighty', 90: 'ninety',
}

ENGLISH_ORDINAL = {
    1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
    6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
}

TAMIL_ONES = [
    '', 'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து',
    'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது',
]

TAMIL_TEENS = [
    'பத்து', 'பதினொன்று', 'பன்னிரண்டு', 'பதிமூன்று',
    'பதினான்கு', 'பதினைந்து', 'பதினாறு', 'பதினேழு',
    'பதினெட்டு', 'பத்தொன்பது',
]

TAMIL_TENS = [
    '', 'பத்து', 'இருபது', 'முப்பது', 'நாற்பது', 'ஐம்பது',
    'அறுபது', 'எழுபது', 'எண்பது', 'தொண்ணூறு',
]


def number_to_english(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + number_to_english(-n)
    parts = []
    if n >= 1_000_000_000:
        parts.append(number_to_english(n // 1_000_000_000) + " billion")
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(number_to_english(n // 1_000_000) + " million")
        n %= 1_000_000
    if n >= 1_000:
        parts.append(number_to_english(n // 1_000) + " thousand")
        n %= 1_000
    if n >= 100:
        parts.append(ENGLISH_CARDINAL[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        tens = (n // 10) * 10
        ones = n % 10
        parts.append(ENGLISH_CARDINAL[tens])
        n = ones
    if 0 < n <= 19:
        parts.append(ENGLISH_CARDINAL[n])
    elif n == 0 and parts:
        pass
    return " ".join(parts)


def number_to_tamil(n: int) -> str:
    if n == 0:
        return TAMIL_ONES[0] or 'சுழி'
    if n < 0:
        return 'கழிவு ' + number_to_tamil(-n)

    parts: list[str] = []

    if n >= 10_00_000:
        crore_part = n // 10_00_000
        parts.append(number_to_tamil(crore_part) + ' கோடி')
        n %= 10_00_000

    if n >= 1_00_000:
        lakh_part = n // 1_00_000
        parts.append(number_to_tamil(lakh_part) + ' லட்சம்')
        n %= 1_00_000

    if n >= 1000:
        thousands = n // 1000
        if thousands == 1:
            parts.append('ஒரு ஆயிரம்')
        else:
            parts.append(number_to_tamil(thousands) + ' ஆயிரம்')
        n %= 1000

    if n >= 100:
        hundreds = n // 100
        if hundreds == 1:
            parts.append('நூறு')
        else:
            parts.append(TAMIL_ONES[hundreds] + ' நூறு')
        n %= 100

    if n >= 20:
        tens = n // 10
        ones = n % 10
        parts.append(TAMIL_TENS[tens])
        if ones:
            parts.append(TAMIL_ONES[ones])
    elif n > 0:
        if n < 10:
            parts.append(TAMIL_ONES[n])
        else:
            parts.append(TAMIL_TEENS[n - 10])

    return ' '.join(p for p in parts if p)


def digits_to_english_spoken(n: str) -> str:
    return ' '.join(ENGLISH_CARDINAL[int(d)] for d in n if d.isdigit())


def digits_to_tamil_spoken(n: str) -> str:
    return ' '.join(TAMIL_DIGITS[d] for d in n if d.isdigit())
