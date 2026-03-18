"""Date and time tools for the LLM agent benchmark."""

from datetime import datetime, timedelta
from typing import Any


class DateTimeTools:
    """Simplified toolkit for date and time operations."""

    @staticmethod
    def get_tools_metadata() -> list[dict[str, Any]]:
        """Return OpenAI-compatible metadata for date/time tools."""
        return [
            {
                "name": "get_date",
                "description": "Получает текущую дату с указанным смещением и форматом",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "enum": ["iso", "us", "eu", "long", "short"]},
                        "offset_days": {"type": "integer"},
                    },
                    "required": [],
                },
            },
            {
                "name": "get_time",
                "description": "Получает текущее время с указанным смещением и форматом",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "enum": ["24h", "12h", "short", "timestamp"]},
                        "offset_hours": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        ]

    @staticmethod
    def get_date(format: str = "iso", offset_days: int = 0) -> dict[str, Any]:
        """Return the current date in the requested format with an optional day offset."""
        base_date = datetime.now() + timedelta(days=offset_days)

        if format == "iso":
            formatted_date = base_date.strftime("%Y-%m-%d")
        elif format == "us":
            formatted_date = base_date.strftime("%m/%d/%Y")
        elif format == "eu":
            formatted_date = base_date.strftime("%d.%m.%Y")
        elif format == "short":
            formatted_date = base_date.strftime("%d.%m.%y")
        elif format == "long":
            months_ru = [
                "января",
                "февраля",
                "марта",
                "апреля",
                "мая",
                "июня",
                "июля",
                "августа",
                "сентября",
                "октября",
                "ноября",
                "декабря",
            ]
            weekdays_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
            formatted_date = f"{weekdays_ru[base_date.weekday()]}, {base_date.day} {months_ru[base_date.month - 1]} {base_date.year} года"

        return {
            "success": True,
            "date": formatted_date,
            "format": format,
            "offset_days": offset_days,
            "day_of_week": base_date.strftime("%A"),
            "day_of_year": base_date.timetuple().tm_yday,
            "week_number": base_date.isocalendar()[1],
            "is_weekend": base_date.weekday() >= 5,
            "iso_format": base_date.strftime("%Y-%m-%d"),
        }

    @staticmethod
    def get_time(format: str = "24h", offset_hours: int = 0) -> dict[str, Any]:
        """Return the current time in the requested format with an optional hour offset."""
        base_time = datetime.now() + timedelta(hours=offset_hours)

        if format == "24h":
            formatted_time = base_time.strftime("%H:%M:%S")
        elif format == "12h":
            formatted_time = base_time.strftime("%I:%M:%S %p")
        elif format == "short":
            formatted_time = base_time.strftime("%H:%M")
        elif format == "timestamp":
            formatted_time = str(int(base_time.timestamp()))

        # Determine part of day
        hour = base_time.hour
        period = (
            "morning"
            if 5 <= hour < 12
            else "afternoon"
            if 12 <= hour < 17
            else "evening"
            if 17 <= hour < 22
            else "night"
        )

        return {
            "success": True,
            "time": formatted_time,
            "format": format,
            "offset_hours": offset_hours,
            "hour": base_time.hour,
            "minute": base_time.minute,
            "second": base_time.second,
            "period": period,
            "is_business_hours": 9 <= base_time.hour < 18 and base_time.weekday() < 5,
            "timestamp": int(base_time.timestamp()),
            "iso_format": base_time.strftime("%H:%M:%S"),
        }


def register_datetime_tools(tool_registry) -> None:
    """Register date/time tools in the given tool registry.

    Args:
        tool_registry: Registry instance exposing a ``register_tool`` method.
    """
    tools_metadata = DateTimeTools.get_tools_metadata()
    executors = {
        "get_date": DateTimeTools.get_date,
        "get_time": DateTimeTools.get_time,
    }
    for tool_meta in tools_metadata:
        tool_name = tool_meta["name"]
        tool_registry.register_tool(tool_name, tool_meta, executors[tool_name])
    print(f"🕐 Зарегистрировано {len(tools_metadata)} инструментов даты/времени")
