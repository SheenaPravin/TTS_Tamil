import pytest
from src.text_normalization.number_normalizer import (
    number_to_english,
    number_to_tamil,
    digits_to_english_spoken,
    digits_to_tamil_spoken,
)


class TestNumberToEnglish:
    def test_zero(self):
        assert number_to_english(0) == "zero"

    def test_single_digits(self):
        assert number_to_english(5) == "five"
        assert number_to_english(9) == "nine"

    def test_teens(self):
        assert number_to_english(13) == "thirteen"
        assert number_to_english(19) == "nineteen"

    def test_tens(self):
        assert number_to_english(20) == "twenty"
        assert number_to_english(42) == "forty two"

    def test_hundreds(self):
        assert number_to_english(100) == "one hundred"
        assert number_to_english(256) == "two hundred fifty six"

    def test_thousands(self):
        assert number_to_english(1000) == "one thousand"
        assert number_to_english(1234) == "one thousand two hundred thirty four"

    def test_large_numbers(self):
        assert "million" in number_to_english(1000000)
        assert "billion" in number_to_english(1000000000)

    def test_negative(self):
        assert "minus" in number_to_english(-5)


class TestNumberToTamil:
    def test_zero(self):
        result = number_to_tamil(0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_digits(self):
        assert number_to_tamil(1) == "ஒன்று"
        assert number_to_tamil(5) == "ஐந்து"
        assert number_to_tamil(9) == "ஒன்பது"

    def test_teens(self):
        assert "பத்து" in number_to_tamil(10)
        assert "பன்னிரண்டு" in number_to_tamil(12)

    def test_tens(self):
        assert "இருபது" in number_to_tamil(20)
        assert "ஐம்பது" in number_to_tamil(50)

    def test_hundreds(self):
        assert "நூறு" in number_to_tamil(100)
        assert "இருநூறு" in number_to_tamil(200)

    def test_thousands(self):
        assert "ஆயிரம்" in number_to_tamil(1000)


class TestDigitsToSpoken:
    def test_english_digits(self):
        assert digits_to_english_spoken("4821") == "four eight two one"
        assert digits_to_english_spoken("123") == "one two three"

    def test_tamil_digits(self):
        result = digits_to_tamil_spoken("4821")
        assert "நான்கு" in result
        assert "எட்டு" in result

    def test_phone_number(self):
        result = digits_to_english_spoken("9876543210")
        assert result.startswith("nine eight seven")
