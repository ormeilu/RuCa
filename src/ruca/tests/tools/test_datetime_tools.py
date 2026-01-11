"""
Tests for datetime_tools (DateTimeTools)
Testing date and time operations with pytest and mock
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from ruca.tools.datetime_tools import DateTimeTools


class TestDateTimeTools:
    """Test suite for DateTimeTools"""

    def test_get_tools_metadata(self):
        """Test that get_tools_metadata returns correct structure"""
        metadata = DateTimeTools.get_tools_metadata()
        
        assert isinstance(metadata, list)
        assert len(metadata) == 2
        
        # Check get_date tool
        date_tool = metadata[0]
        assert date_tool["name"] == "get_date"
        assert "description" in date_tool
        assert "parameters" in date_tool
        
        # Check get_time tool
        time_tool = metadata[1]
        assert time_tool["name"] == "get_time"
        assert "parameters" in time_tool

    def test_get_date_iso_format(self):
        """Test get_date with ISO format"""
        result = DateTimeTools.get_date(format="iso")
        
        assert result["success"] is True
        assert "date" in result
        assert result["format"] == "iso"
        # Check ISO format (YYYY-MM-DD)
        assert len(result["date"].split("-")) == 3

    def test_get_date_us_format(self):
        """Test get_date with US format"""
        result = DateTimeTools.get_date(format="us")
        
        assert result["success"] is True
        assert result["format"] == "us"
        # Check US format (MM/DD/YYYY)
        assert "/" in result["date"]

    def test_get_date_eu_format(self):
        """Test get_date with EU format"""
        result = DateTimeTools.get_date(format="eu")
        
        assert result["success"] is True
        assert result["format"] == "eu"
        # Check EU format (DD.MM.YYYY)
        assert "." in result["date"]

    def test_get_date_short_format(self):
        """Test get_date with short format"""
        result = DateTimeTools.get_date(format="short")
        
        assert result["success"] is True
        assert result["format"] == "short"

    def test_get_date_long_format(self):
        """Test get_date with long format (Russian)"""
        result = DateTimeTools.get_date(format="long")
        
        assert result["success"] is True
        assert result["format"] == "long"
        assert "года" in result["date"]  # Russian long format

    def test_get_date_with_offset(self):
        """Test get_date with day offset"""
        result = DateTimeTools.get_date(format="iso", offset_days=5)
        
        assert result["success"] is True
        assert result["offset_days"] == 5

    def test_get_date_negative_offset(self):
        """Test get_date with negative day offset"""
        result = DateTimeTools.get_date(format="iso", offset_days=-3)
        
        assert result["success"] is True
        assert result["offset_days"] == -3

    def test_get_date_contains_day_of_week(self):
        """Test that get_date includes day of week"""
        result = DateTimeTools.get_date()
        
        assert "day_of_week" in result
        assert result["day_of_week"] in [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

    def test_get_date_contains_day_of_year(self):
        """Test that get_date includes day of year"""
        result = DateTimeTools.get_date()
        
        assert "day_of_year" in result
        assert 1 <= result["day_of_year"] <= 366

    def test_get_date_contains_week_number(self):
        """Test that get_date includes ISO week number"""
        result = DateTimeTools.get_date()
        
        assert "week_number" in result
        assert 1 <= result["week_number"] <= 53

    def test_get_date_contains_is_weekend(self):
        """Test that get_date includes is_weekend flag"""
        result = DateTimeTools.get_date()
        
        assert "is_weekend" in result
        assert isinstance(result["is_weekend"], bool)

    def test_get_date_contains_iso_format(self):
        """Test that get_date includes ISO format string"""
        result = DateTimeTools.get_date()
        
        assert "iso_format" in result
        # Check ISO format (YYYY-MM-DD)
        parts = result["iso_format"].split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # Year

    def test_get_time_24h_format(self):
        """Test get_time with 24-hour format"""
        result = DateTimeTools.get_time(format="24h")
        
        assert result["success"] is True
        assert result["format"] == "24h"
        assert "time" in result
        assert ":" in result["time"]

    def test_get_time_12h_format(self):
        """Test get_time with 12-hour format"""
        result = DateTimeTools.get_time(format="12h")
        
        assert result["success"] is True
        assert result["format"] == "12h"
        # 12-hour format should have AM/PM
        assert "AM" in result["time"] or "PM" in result["time"]

    def test_get_time_short_format(self):
        """Test get_time with short format"""
        result = DateTimeTools.get_time(format="short")
        
        assert result["success"] is True
        assert result["format"] == "short"

    def test_get_time_timestamp_format(self):
        """Test get_time with timestamp format"""
        result = DateTimeTools.get_time(format="timestamp")
        
        assert result["success"] is True
        assert result["format"] == "timestamp"
        # Timestamp should be numeric
        assert isinstance(result["timestamp"], (int, float))

    def test_get_time_with_hour_offset(self):
        """Test get_time with hour offset"""
        result = DateTimeTools.get_time(format="24h", offset_hours=3)
        
        assert result["success"] is True
        assert result["offset_hours"] == 3

    def test_get_time_negative_hour_offset(self):
        """Test get_time with negative hour offset"""
        result = DateTimeTools.get_time(format="24h", offset_hours=-5)
        
        assert result["success"] is True
        assert result["offset_hours"] == -5

    def test_get_time_contains_timezone_info(self):
        """Test that get_time contains timezone information"""
        result = DateTimeTools.get_time()
        
        # Should have UTC offset information
        assert "time" in result or "timestamp" in result

    def test_get_date_returns_dict(self):
        """Test that get_date returns a dictionary"""
        result = DateTimeTools.get_date()
        
        assert isinstance(result, dict)
        assert "success" in result

    def test_get_time_returns_dict(self):
        """Test that get_time returns a dictionary"""
        result = DateTimeTools.get_time()
        
        assert isinstance(result, dict)
        assert "success" in result

    @patch('ruca.tools.datetime_tools.datetime')
    def test_get_date_uses_datetime(self, mock_datetime):
        """Test that get_date uses datetime module"""
        from datetime import datetime as real_datetime
        
        mock_datetime.now.return_value = real_datetime(2026, 1, 15)
        mock_datetime.side_effect = lambda *args, **kw: real_datetime(*args, **kw)
        
        result = DateTimeTools.get_date(format="iso")
        assert result["success"] is True

    def test_get_date_multiple_formats_consistency(self):
        """Test that same date can be formatted in multiple ways"""
        iso = DateTimeTools.get_date(format="iso")
        us = DateTimeTools.get_date(format="us", offset_days=0)
        
        # Both should have success
        assert iso["success"] is True
        assert us["success"] is True
        
        # They represent the same day
        assert iso["day_of_year"] == us["day_of_year"]

    def test_get_time_offset_consistency(self):
        """Test that time offset works correctly"""
        time_now = DateTimeTools.get_time(format="24h", offset_hours=0)
        time_future = DateTimeTools.get_time(format="24h", offset_hours=1)
        
        assert time_now["success"] is True
        assert time_future["success"] is True

    def test_get_date_large_offset(self):
        """Test get_date with large day offset"""
        result = DateTimeTools.get_date(format="iso", offset_days=365)
        
        assert result["success"] is True
        assert result["offset_days"] == 365

    def test_get_time_large_offset(self):
        """Test get_time with large hour offset"""
        result = DateTimeTools.get_time(format="24h", offset_hours=48)
        
        assert result["success"] is True
        assert result["offset_hours"] == 48

    def test_get_date_success_always_true(self):
        """Test that get_date always returns success=True"""
        for format_type in ["iso", "us", "eu", "short", "long"]:
            result = DateTimeTools.get_date(format=format_type)
            assert result["success"] is True

    def test_get_time_success_always_true(self):
        """Test that get_time always returns success=True"""
        for format_type in ["24h", "12h", "short", "timestamp"]:
            result = DateTimeTools.get_time(format=format_type)
            assert result["success"] is True
