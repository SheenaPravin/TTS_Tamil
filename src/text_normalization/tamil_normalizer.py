from __future__ import annotations

import re


TAMIL_LIKE = re.compile(
    r'[\u0B80-\u0BFF]+',
    re.UNICODE,
)

LATIN_IN_TAMIL_CONTEXT = re.compile(
    r'(?<=[\u0B80-\u0BFF])\s*([A-Za-z]+(?:\s*[A-Za-z]+)*)'
    r'|'
    r'([A-Za-z]+(?:\s*[A-Za-z]+)*)\s*(?=[\u0B80-\u0BFF])',
    re.UNICODE,
)

TAMIL_SUFFIXES = {
    'ல': 'இல்',
    'இல்': 'இல்',
    'ஐ': 'ஐ',
    'அன்': 'அன்',
    'கள்': 'கள்',
    'ஆ': 'ஆ',
    'உடன்': 'உடன்',
    'இலிருந்து': 'இலிருந்து',
    'க்கு': 'க்கு',
    'இன்': 'இன்',
    'க்கும்': 'க்கும்',
    'களுக்கு': 'களுக்கு',
}

TAMIL_VOWEL_MARKS = {
    '\u0BBE': 'ா', '\u0BBF': 'ி', '\u0BC0': 'ீ',
    '\u0BC1': 'ு', '\u0BC2': 'ூ', '\u0BC6': 'ெ',
    '\u0BC7': 'ே', '\u0BC8': 'ை', '\u0BCA': 'ொ',
    '\u0BCB': 'ோ', '\u0BCC': 'ௌ',
}


def is_tamil_text(text: str) -> bool:
    return bool(TAMIL_LIKE.search(text))


def has_tamil_script(text: str) -> bool:
    return any('\u0B80' <= c <= '\u0BFF' for c in text)


def detect_language_segments(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    tokens = re.findall(r'[\u0B80-\u0BFF]+|[A-Za-z]+|[0-9]+|[^\s\w]+|\s+', text)

    current_lang = 'unknown'
    current_tokens: list[str] = []

    for token in tokens:
        if re.match(r'[\u0B80-\u0BFF]+', token):
            lang = 'tamil'
        elif re.match(r'[A-Za-z]+', token):
            lang = 'english'
        elif re.match(r'[0-9]+', token):
            lang = current_lang if current_lang != 'unknown' else 'neutral'
        elif re.match(r'\s+', token):
            current_tokens.append(token)
            continue
        else:
            lang = current_lang if current_lang != 'unknown' else 'neutral'

        if lang != current_lang and current_tokens:
            segments.append((current_lang, ''.join(current_tokens)))
            current_tokens = []
        current_lang = lang
        current_tokens.append(token)

    if current_tokens:
        segments.append((current_lang, ''.join(current_tokens)))

    return segments


def split_tanglish(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pattern = re.compile(
        r'([\u0B80-\u0BFF]+(?:\s*[\u0B80-\u0BFF]+)*)'
        r'|([A-Za-z]+(?:\s*[A-Za-z]+)*)'
        r'|([0-9]+(?:[0-9:.\-/]*)?)'
        r'|(\s+)'
        r'|([^\s\w]+)',
        re.UNICODE,
    )

    for m in pattern.finditer(text):
        if m.group(1):
            parts.append(('tamil', m.group(1)))
        elif m.group(2):
            parts.append(('english', m.group(2)))
        elif m.group(3):
            parts.append(('number', m.group(3)))
        elif m.group(4):
            parts.append(('space', m.group(4)))
        elif m.group(5):
            parts.append(('punct', m.group(5)))

    return parts


def normalize_tamil_punctuation(text: str) -> str:
    punct_map = {
        '?': '?',
        '!': '!',
        '.': '.',
        ',': ',',
        ':': ':',
        ';': ';',
    }
    for orig, repl in punct_map.items():
        text = text.replace(orig, repl)
    return text


def romanized_tamil_to_tamil(text: str) -> str:
    roman_map = {
        'vanakkam': 'வணக்கம்',
        'nandri': 'நன்றி',
        'podum': 'போதும்',
        'illa': 'இல்லை',
        'aama': 'ஆம்',
        'illai': 'இல்லை',
        'seri': 'சரி',
        'epdi': 'எப்படி',
        'naan': 'நான்',
        'nee': 'நீ',
        'avan': 'அவன்',
        'aval': 'அவள்',
        'nammal': 'நாம்',
        'avargal': 'அவர்கள்',
        'idhu': 'இது',
        'adhu': 'அது',
        'ingu': 'இங்கு',
        'angu': 'அங்கு',
        'ippo': 'இப்போது',
        'appothu': 'அப்போது',
        'inniki': 'இன்று',
        'naalaik': 'நாளை',
        'ku': 'க்கு',
        'oda': 'ஓட',
        'la': 'இல்',
        'il': 'இல்',
    }

    lower = text.lower().strip()
    if lower in roman_map:
        return roman_map[lower]
    return text


def has_romanized_tamil(text: str) -> bool:
    roman_tamil_indicators = [
        r'\b(illa|aama|seri|epdi|vanakkam|nandri)\b',
        r'\b(naan|nee|avan|aval|nammal)\b',
        r'\b(idhu|adhu|ingu|angu|ippo|appothu)\b',
        r'\b(inniki|naalaik|podum)\b',
        r'\b(ku|oda|la|il)\b(?=\s|$)',
    ]
    for pattern in roman_tamil_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
