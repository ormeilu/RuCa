"""Mock tools: weather forecast and currency converter.

All responses are deterministic / predetermined — no real API calls.
"""

import random
from datetime import datetime, timedelta
from typing import Any


class MiscTools:
    """Mock toolkit: weather and currency conversion."""

    @staticmethod
    def get_tools_metadata() -> list[dict[str, Any]]:
        """Return OpenAI-compatible metadata for every misc tool."""
        return [
            {
                "name": "get_weather",
                "description": "Возвращает погодный прогноз (мок).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Город или геолокация"},
                        "date": {"type": "string", "description": "Дата в формате YYYY-MM-DD (опционально)"},
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "currency_converter",
                "description": "Конвертирует сумму из одной валюты в другую (локальные курсы).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_currency": {"type": "string", "description": "Исходная валюта (USD, EUR, RUB, etc.)"},
                        "to_currency": {"type": "string", "description": "Целевая валюта"},
                        "amount": {"type": "number", "description": "Сумма для конвертации"},
                    },
                    "required": ["from_currency", "to_currency", "amount"],
                },
            },
        ]

    @staticmethod
    def get_weather(location: str, date: str = None) -> dict[str, Any]:
        """Mock: return a weather forecast for the given date (or the next 3 days)."""
        # Validate date format if provided
        try:
            target = datetime.strptime(date, "%Y-%m-%d").date() if date else None
        except (TypeError, ValueError):
            return {"success": False, "error": "invalid_date_format", "expected": "YYYY-MM-DD"}

        base_conditions = [
            {"cond": "clear", "desc": "Ясно", "temp_avg": 20},
            {"cond": "partly_cloudy", "desc": "Переменная облачность", "temp_avg": 17},
            {"cond": "rain", "desc": "Дождь", "temp_avg": 14},
            {"cond": "storm", "desc": "Гроза", "temp_avg": 13},
            {"cond": "snow", "desc": "Снег", "temp_avg": -2},
        ]

        def make_day(cond_template: dict[str, Any], day_offset: int, use_date=None) -> dict[str, Any]:
            temp = cond_template["temp_avg"] + random.randint(-4, 4)
            d = use_date if use_date is not None else (datetime.now().date() + timedelta(days=day_offset))
            return {
                "date": d.isoformat(),
                "condition": cond_template["cond"],
                "description": cond_template["desc"],
                "temp_c": temp,
                "wind_kph": random.randint(5, 25),
                "precip_mm": random.choice([0, 0, 0.5, 2, 5]),
            }

        # If a specific date is provided — return a single day with deterministic logic
        if target:
            idx = target.toordinal() % len(base_conditions)
            cond = base_conditions[idx]
            forecast = make_day(cond, 0, use_date=target)
            return {"success": True, "location": location, "forecast": forecast, "source": "mock"}

        # Otherwise — return a short 3-day forecast
        forecast_list = []
        for i in range(3):
            cond = base_conditions[(i + len(location)) % len(base_conditions)]
            forecast_list.append(make_day(cond, i))

        return {"success": True, "location": location, "forecast_3days": forecast_list, "source": "mock"}

    @staticmethod
    def currency_converter(from_currency: str, to_currency: str, amount: float) -> dict[str, Any]:
        """Mock: simple converter with fixed exchange rates."""
        # Local exchange-rate table relative to USD
        rates_to_usd = {"USD": 1.0, "EUR": 1.08, "RUB": 0.012, "GBP": 1.25, "JPY": 0.0067, "CNY": 0.14}

        fc = from_currency.upper()
        tc = to_currency.upper()

        if fc not in rates_to_usd or tc not in rates_to_usd:
            return {"success": False, "error": "unsupported_currency", "supported": list(rates_to_usd.keys())}

        # Convert via USD
        # Interpretation: rates_to_usd[x] = 1 unit of x -> USD value
        usd_amount = amount * rates_to_usd[fc]
        target_amount = usd_amount / rates_to_usd[tc]

        return {
            "success": True,
            "from_currency": fc,
            "to_currency": tc,
            "original_amount": amount,
            "converted_amount": round(target_amount, 4),
            "rate_via_usd": {"from_to_usd": rates_to_usd[fc], "to_to_usd": rates_to_usd[tc]},
            "source": "mock_rates_2025",
        }


def register_misc_tools(tool_registry) -> None:
    """Register misc tools (weather & currency) in the given tool registry.

    Args:
        tool_registry: Registry instance exposing a ``register_tool`` method.
    """
    tools_metadata = MiscTools.get_tools_metadata()

    executors = {"get_weather": MiscTools.get_weather, "currency_converter": MiscTools.currency_converter}

    for tool_meta in tools_metadata:
        name = tool_meta["name"]
        tool_registry.register_tool(name, tool_meta, executors[name])

    print(f"✅ Зарегистрировано {len(tools_metadata)} misc инструментов (weather & currency)")
