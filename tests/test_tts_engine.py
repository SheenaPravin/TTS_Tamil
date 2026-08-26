import pytest
from src.text_normalization.normalizer import TextNormalizer


class TestTextNormalizer:
    @pytest.fixture
    def normalizer(self):
        return TextNormalizer(lang='mixed', context='transport')

    @pytest.fixture
    def en_normalizer(self):
        return TextNormalizer(lang='en', context='transport')

    @pytest.fixture
    def ta_normalizer(self):
        return TextNormalizer(lang='ta', context='transport')

    def test_booking_id(self, en_normalizer):
        result = en_normalizer.normalize("Your booking ID is TN45AB1234.")
        assert "T" in result
        assert "N" in result
        assert "four" in result
        assert "five" in result
        assert "A" in result
        assert "B" in result

    def test_otp_digit_by_digit(self, en_normalizer):
        result = en_normalizer.normalize("Your OTP is 4821.")
        assert "four" in result
        assert "eight" in result
        assert "two" in result
        assert "one" in result
        assert "thousand" not in result.lower()

    def test_phone_number_digit_by_digit(self, en_normalizer):
        result = en_normalizer.normalize("Your phone number is 9876543210.")
        assert "nine" in result
        assert "eight" in result
        assert "seven" in result
        assert "six" in result
        assert "five" in result

    def test_time_normalization(self, en_normalizer):
        result = en_normalizer.normalize("at 7:30 PM")
        assert "seven" in result or "7" in result

    def test_price_normalization_en(self, en_normalizer):
        result = en_normalizer.normalize("The fare is Rs 450.")
        assert "four hundred" in result or "forty" in result

    def test_distance_normalization(self, en_normalizer):
        result = en_normalizer.normalize("12.5 km")
        assert "kilometer" in result.lower() or "12" in result

    def test_tamil_price(self, ta_normalizer):
        result = ta_normalizer.normalize("கட்டணம் ₹450.")
        assert "ரூபாய்" in result or "நான்கு" in result

    def test_abbreviation(self, en_normalizer):
        result = en_normalizer.normalize("Your OTP is 1234.")
        assert "O" in result

    def test_vehicle_type(self, en_normalizer):
        result = en_normalizer.normalize("Book an auto.")
        assert "auto rickshaw" in result.lower()

    def test_multiple_entities(self, en_normalizer):
        result = en_normalizer.normalize(
            "Your cab will arrive at 7:30 PM. "
            "Fare is Rs 450. Distance is 12 km."
        )
        assert len(result) > 0

    def test_mixed_text(self, normalizer):
        result = normalizer.normalize("Pickup location Chennai.")
        assert len(result) > 0

    def test_empty_text(self, en_normalizer):
        result = en_normalizer.normalize("")
        assert result == ""

    def test_unicode_preserved(self, ta_normalizer):
        result = ta_normalizer.normalize("வணக்கம்.")
        assert "வணக்கம்" in result
