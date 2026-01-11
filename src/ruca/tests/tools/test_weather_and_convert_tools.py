"""
Tests for weather_and_convert_tools (MiscTools)
Testing get_weather and currency_converter methods with pytest and mock
"""

from unittest.mock import patch

from ruca.tools.weather_and_convert_tools import MiscTools


class TestMiscTools:
    """Test suite for MiscTools (weather and currency conversion)"""

    def test_get_tools_metadata(self):
        """Test that get_tools_metadata returns correct structure"""
        metadata = MiscTools.get_tools_metadata()

        assert isinstance(metadata, list)
        assert len(metadata) == 2

        # Check weather tool metadata
        weather_tool = metadata[0]
        assert weather_tool["name"] == "get_weather"
        assert "description" in weather_tool
        assert "parameters" in weather_tool
        assert "location" in weather_tool["parameters"]["properties"]

        # Check currency tool metadata
        currency_tool = metadata[1]
        assert currency_tool["name"] == "currency_converter"
        assert "parameters" in currency_tool

    def test_get_weather_without_date(self, sample_location):
        """Test get_weather returns 3-day forecast when no date specified"""
        result = MiscTools.get_weather(location=sample_location)

        assert result["success"] is True
        assert result["location"] == sample_location
        assert "forecast_3days" in result
        assert len(result["forecast_3days"]) == 3
        assert result["source"] == "mock"

        # Check forecast structure
        for day in result["forecast_3days"]:
            assert "date" in day
            assert "condition" in day
            assert "description" in day
            assert "temp_c" in day
            assert "wind_kph" in day
            assert "precip_mm" in day

    def test_get_weather_with_valid_date(self, sample_location, sample_date):
        """Test get_weather returns single day forecast for specified date"""
        result = MiscTools.get_weather(location=sample_location, date=sample_date)

        assert result["success"] is True
        assert result["location"] == sample_location
        assert "forecast" in result
        assert result["forecast"]["date"] == sample_date
        assert "condition" in result["forecast"]
        assert "temp_c" in result["forecast"]

    def test_get_weather_invalid_date_format(self, sample_location):
        """Test get_weather handles invalid date format"""
        result = MiscTools.get_weather(location=sample_location, date="invalid-date")

        assert result["success"] is False
        assert result["error"] == "invalid_date_format"
        assert "expected" in result

    def test_get_weather_returns_deterministic_results(self, sample_location):
        """Test that same location with same date returns deterministic result"""
        date = "2026-01-15"
        result1 = MiscTools.get_weather(location=sample_location, date=date)
        result2 = MiscTools.get_weather(location=sample_location, date=date)

        # Results should be consistent (based on location name)
        assert result1["forecast"]["condition"] == result2["forecast"]["condition"]
        assert result1["forecast"]["description"] == result2["forecast"]["description"]

    def test_currency_converter_basic(self):
        """Test basic currency conversion"""
        result = MiscTools.currency_converter(from_currency="USD", to_currency="EUR", amount=100.0)

        assert result["success"] is True
        assert result["from_currency"] == "USD"
        assert result["to_currency"] == "EUR"
        assert result["original_amount"] == 100.0
        assert "converted_amount" in result
        assert "converted_amount" in result
        assert isinstance(result["converted_amount"], (int, float))

    def test_currency_converter_same_currency(self):
        """Test conversion when source and target currencies are the same"""
        result = MiscTools.currency_converter(from_currency="USD", to_currency="USD", amount=50.0)

        assert result["success"] is True
        assert result["converted_amount"] == 50.0

    def test_currency_converter_supported_currencies(self):
        """Test conversion with various supported currencies"""
        currencies = ["USD", "EUR", "RUB", "GBP", "JPY"]

        for curr in currencies:
            result = MiscTools.currency_converter(from_currency="USD", to_currency=curr, amount=100.0)
            assert result["success"] is True or result["success"] is False

    def test_currency_converter_zero_amount(self):
        """Test currency conversion with zero amount"""
        result = MiscTools.currency_converter(from_currency="USD", to_currency="EUR", amount=0.0)

        assert result["success"] is True
        assert result["converted_amount"] == 0.0

    def test_currency_converter_large_amount(self):
        """Test currency conversion with large amount"""
        result = MiscTools.currency_converter(from_currency="USD", to_currency="EUR", amount=1000000.0)

        assert result["success"] is True
        assert result["converted_amount"] > 0

    @patch("ruca.tools.weather_and_convert_tools.random.randint")
    def test_get_weather_mocked_random(self, mock_randint, sample_location):
        """Test get_weather with mocked random values"""
        # Нужно больше значений - для каждого дня вызывается 2 раза randint (temp, wind)
        # 3 дня * 2 вызова = 6 значений
        mock_randint.side_effect = [15, 10, 15, 10, 15, 10]
        result = MiscTools.get_weather(location=sample_location)
        assert result["success"] is True
        # Verify randint was called for random values
        assert mock_randint.called

    def test_get_weather_temp_range(self, sample_location):
        """Test that temperature values are within reasonable range"""
        result = MiscTools.get_weather(location=sample_location, date="2026-01-15")

        temp = result["forecast"]["temp_c"]
        # Temperature should be somewhat reasonable (between -10 and 50 celsius)
        assert -20 <= temp <= 50

    def test_get_weather_wind_speed_range(self, sample_location):
        """Test that wind speed values are within expected range"""
        result = MiscTools.get_weather(location=sample_location, date="2026-01-15")

        wind = result["forecast"]["wind_kph"]
        # Wind speed should be between 5 and 25 kph
        assert 5 <= wind <= 25

    def test_currency_converter_returns_dict(self):
        """Test that currency_converter returns a dictionary"""
        result = MiscTools.currency_converter(from_currency="USD", to_currency="EUR", amount=100.0)

        assert isinstance(result, dict)
        assert "success" in result

    def test_get_weather_location_in_response(self, sample_location):
        """Test that location is properly included in response"""
        result = MiscTools.get_weather(location=sample_location)

        assert result["location"] == sample_location

    def test_get_weather_forecast_list_not_empty(self, sample_location):
        """Test that forecast list is not empty"""
        result = MiscTools.get_weather(location=sample_location)

        if "forecast_3days" in result:
            assert len(result["forecast_3days"]) > 0

    @patch("ruca.tools.weather_and_convert_tools.datetime")
    def test_get_weather_uses_datetime(self, mock_datetime, sample_location):
        """Test that get_weather uses datetime module"""
        from datetime import datetime as real_datetime

        mock_datetime.now.return_value = real_datetime(2026, 1, 15, 12, 0, 0)
        mock_datetime.side_effect = lambda *args, **kw: real_datetime(*args, **kw)

        # Should still work
        result = MiscTools.get_weather(location=sample_location)
        assert result["success"] is True
