import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import dotenv
from openai import OpenAI

from settings import OpenAISettings

# Подключаем папку с инструментами рядом со скриптом
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

dotenv.load_dotenv(dotenv.find_dotenv())

try:
    from json_parser import process_all_queries, system_prompt as default_system_prompt
except ImportError:
    process_all_queries = None
    default_system_prompt = "Ты — ассистент, вызывающий инструменты."


class BenchmarkAgent:

    def __init__(
        self,
        model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        *,
        use_retail: bool = True,
        use_weather: bool = True,
        use_translate: bool = True,
        use_calculator: bool = True,
        use_trash: bool = True,
        use_airbnb: bool = True,
        use_flights: bool = True,
        verbose: bool = True,
    ) -> None:
        """Инициализирует агента для бенчмарка.

        Args:
            model: Идентификатор модели для OpenAI-совместимого API.
            use_retail: Подключать ли ecommerce-инструменты.
            use_weather: Подключать ли погодные/валютные инструменты.
            use_translate: Подключать ли переводчик.
            use_calculator: Подключать ли калькулятор.
            use_trash: Подключать ли вспомогательные trash-инструменты.
            use_airbnb: Запускать ли MCP-сервер Airbnb.
            use_flights: Запускать ли MCP-сервер flights.
            verbose: Выводить ли дополнительную диагностику при старте.
        """
        self.model = model
        self.verbose = verbose
        self.openai_client = self._build_client()
        self.openai_tools: List[Dict[str, Any]] = []
        self.executors: Dict[str, Any] = {}
        self.mcp_clients: Dict[str, Any] = {}

        if use_retail:
            from retail_tools import EcommerceTools

            retail_tools = EcommerceTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_tools(retail_tools))
            self.executors.update(self._get_retail_executors())

        if use_weather:
            from weather_and_convert import MiscTools

            weather_tools = MiscTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_tools(weather_tools))
            self.executors.update(self._get_weather_executors())

        if use_translate:
            from trans import TranslateTools

            translate_tools = TranslateTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_tools(translate_tools))
            self.executors.update(self._get_translate_executors())

        if use_calculator:
            from caclute import CalculatorTool

            calculator_tools = CalculatorTool.get_tools_metadata()
            self.openai_tools.extend(self._convert_tools(calculator_tools))
            self.executors.update(self._get_calculator_executors())

        if use_trash:
            from trash import NullTools

            trash_tools = NullTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_tools(trash_tools))
            self.executors.update(self._get_trash_executors())

        # MCP-серверы поднимаем только при явном запросе.
        if use_airbnb or use_flights:
            self._setup_mcp_clients(use_airbnb=use_airbnb, use_flights=use_flights)
            self._register_mcp_tools()

        if self.verbose:
            print(f"Агент инициализирован с {len(self.openai_tools)} инструментами")

    def _build_client(self) -> OpenAI:
        """Создаёт аутентифицированный клиент OpenAI.

        Returns:
            OpenAI: Клиент с заранее настроенным base URL.
        """
        settings = OpenAISettings()
        return OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url),
        )

    def _convert_tools(self, tools_meta: List[Any], *, strict: bool = False) -> List[Dict[str, Any]]:
        """Преобразует произвольные описания тулзов в формат OpenAI.

        Args:
            tools_meta: Исходные метаданные от поставщиков инструментов.
            strict: Запрещать ли дополнительные параметры в схемах.

        Returns:
            List[Dict[str, Any]]: Описания функций, совместимые с OpenAI.
        """
        converted: List[Dict[str, Any]] = []
        for tool in tools_meta:
            params = tool.get("parameters") or tool.get("inputSchema") or {}
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": params.get("type", "object"),
                        "properties": params.get("properties", {}),
                        "required": params.get("required", []),
                    },
                },
            }
            if strict:
                openai_tool["function"]["parameters"]["additionalProperties"] = False
                openai_tool["function"]["strict"] = True
            converted.append(openai_tool)
        return converted

    def _get_retail_executors(self) -> Dict[str, Any]:
        """Возвращает исполняемые методы для ретейл тулзов."""
        from retail_tools import EcommerceTools

        method_names = [
            "cancel_order",
            "search_products",
            "return_order",
            "place_order",
            "track_order",
            "update_address",
            "add_to_cart",
            "remove_from_cart",
            "update_payment_method",
            "apply_discount_code",
            "get_order_history",
            "schedule_delivery",
            "update_profile",
            "contact_support",
        ]
        return {
            name: getattr(EcommerceTools, name)
            for name in method_names
            if hasattr(EcommerceTools, name)
        }

    def _get_weather_executors(self) -> Dict[str, Any]:
        """Возвращает исполняемые методы для погодных/валютных тулзов."""
        from weather_and_convert import MiscTools

        return {
            "get_weather": MiscTools.get_weather,
            "currency_converter": MiscTools.currency_converter,
        }

    def _get_translate_executors(self) -> Dict[str, Any]:
        """Возвращает исполняемые методы для тулзов перевода."""
        from trans import TranslateTools

        return {"translate": TranslateTools.translate}

    def _get_calculator_executors(self) -> Dict[str, Any]:
        """Возвращает исполняемые методы для калькуляторных тулзов."""
        from caclute import CalculatorTool

        return {"calculator": CalculatorTool.calculate}

    def _get_trash_executors(self) -> Dict[str, Any]:
        """Возвращает исполняемые методы для мусорных тулзов."""
        from trash import NullTools

        return {
            name: getattr(NullTools, name)
            for name in dir(NullTools)
            if not name.startswith("_")
        }

    def _setup_mcp_clients(self, *, use_airbnb: bool, use_flights: bool) -> None:
        """Поднимает нужные MCP-клиенты в текущем процессе.

        Args:
            use_airbnb: Создавать ли клиент Airbnb MCP.
            use_flights: Создавать ли клиент flights MCP.
        """
        from fastmcp import Client

        loop = asyncio.get_event_loop()

        if use_airbnb:
            airbnb_config = {
                "mcpServers": {
                    "airbnb": {
                        "command": "npx",
                        "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
                    }
                }
            }
            airbnb_client = Client(airbnb_config)
            loop.run_until_complete(airbnb_client.__aenter__())
            self.mcp_clients["airbnb"] = airbnb_client

        if use_flights:
            duffel_api_key = os.getenv("DUFFEL_API_KEY_LIVE", "")
            flights_path = os.path.join(os.path.dirname(__file__), "flights-mcp")
            flights_config = {
                "mcpServers": {
                    "flights": {
                        "command": "uv",
                        "args": ["--directory", flights_path, "run", "flights-mcp"],
                        "env": {"DUFFEL_API_KEY_LIVE": duffel_api_key},
                    }
                }
            }
            flights_client = Client(flights_config)
            loop.run_until_complete(flights_client.__aenter__())
            self.mcp_clients["flights"] = flights_client

    def _register_mcp_tools(self) -> None:
        """Запрашивает манифесты MCP-клиентов и регистрирует инструменты."""
        if not self.mcp_clients:
            return

        loop = asyncio.get_event_loop()

        for name, client in self.mcp_clients.items():
            try:
                listing = loop.run_until_complete(client.list_tools())
            except Exception as exc:
                if self.verbose:
                    print(f"Не удалось получить инструменты MCP '{name}': {exc}")
                continue

            tools_meta = getattr(listing, "tools", listing) or []
            for tool in tools_meta:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": self._safe_attr(tool, "name"),
                        "description": self._safe_attr(tool, "description"),
                        "parameters": self._extract_schema(tool),
                    },
                }
                self.openai_tools.append(openai_tool)

    @staticmethod
    def _safe_attr(tool: Any, key: str) -> str:
        """Безопасно читает атрибут или ключ из описания тула."""
        if isinstance(tool, dict):
            return str(tool.get(key, ""))
        return str(getattr(tool, key, ""))

    @staticmethod
    def _extract_schema(tool: Any) -> Dict[str, Any]:
        """Нормализует произвольные схемы до вида JSON."""
        def _normalize(schema: Any) -> Dict[str, Any]:
            if schema is None:
                return {"type": "object", "properties": {}, "required": []}
            if isinstance(schema, dict):
                normalized = dict(schema)
            elif hasattr(schema, "model_dump"):
                normalized = schema.model_dump()
            elif hasattr(schema, "dict"):
                normalized = schema.dict()
            else:
                normalized = {}
            normalized.setdefault("type", "object")
            normalized.setdefault("properties", {})
            normalized.setdefault("required", [])
            return normalized

        schema = None
        if isinstance(tool, dict):
            schema = tool.get("input_schema") or tool.get("inputSchema") or tool.get("parameters")
        else:
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
        return _normalize(schema)

    def get_tools_info(self) -> List[Dict[str, str]]:
        """Возвращает краткие метаданные обо всех зарегистрированных тулах.

        Returns:
            List[Dict[str, str]]: Названия и описания для логирования.
        """
        return [
            {
                "name": tool["function"].get("name", ""),
                "description": tool["function"].get("description", ""),
            }
            for tool in self.openai_tools
        ]

    async def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Выполняет единичный вызов тула локально или через MCP.

        Args:
            tool_name: Название вызываемого инструмента.
            arguments: Распаршенные аргументы из ответа модели.

        Returns:
            Dict[str, Any]: Эхо-вызов и фактический результат.
        """
        try:
            if tool_name in self.executors:
                result = self.executors[tool_name](**arguments)
                return {
                    "tool_call": {"name": tool_name, "parameters": arguments},
                    "result": result,
                }

            if self.mcp_clients:
                for _, client in self.mcp_clients.items():
                    try:
                        client_result = await client.call_tool(tool_name, arguments)
                        if hasattr(client_result, "content"):
                            buffer = ""
                            for item in client_result.content:
                                if hasattr(item, "text"):
                                    buffer += item.text
                            return {
                                "tool_call": {"name": tool_name, "parameters": arguments},
                                "result": buffer,
                            }
                        return {
                            "tool_call": {"name": tool_name, "parameters": arguments},
                            "result": str(client_result),
                        }
                    except Exception:
                        continue

            return {
                "tool_call": {"name": tool_name, "parameters": arguments},
                "result": {"error": "Tool not found"},
            }
        except Exception as exc:
            return {
                "tool_call": {"name": tool_name, "parameters": arguments},
                "result": {"error": str(exc)},
            }

    def run_single_query(
        self,
        *,
        user_query: str,
        system_prompt: str,
        query_id: str,
    ) -> Dict[str, Any]:
        """Делает один вызов модели и при необходимости запускает тул.

        Args:
            user_query: Текстовый запрос из датасета.
            system_prompt: Общий системный промпт бенчмарка.
            query_id: Идентификатор запроса для логов/метрик.

        Returns:
            Dict[str, Any]: Структурированный ответ с деталями выполнения тула.
        """
        loop = asyncio.get_event_loop()
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            tools=self.openai_tools,
            tool_choice="auto",
        )

        serialized = {
            "content": getattr(response.choices[0].message, "content", None),
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
                for call in getattr(response.choices[0].message, "tool_calls", [])
            ],
        }

        tool_calls = getattr(response.choices[0].message, "tool_calls", [])
        if tool_calls:
            call = tool_calls[0]
            arguments = json.loads(call.function.arguments)
            tool_result = loop.run_until_complete(
                self._execute_tool_call(call.function.name, arguments)
            )
            return {
                "tool_call": tool_result["tool_call"],
                "tool_result": tool_result["result"],
                "user_message": None,
                "clarification_question": None,
                "assistant_response": serialized.get("content"),
                "internal": {
                    "query_id": query_id,
                    "raw_response": serialized,
                },
            }

        return {
            "tool_call": None,
            "tool_result": None,
            "user_message": serialized.get("content"),
            "clarification_question": None,
            "assistant_response": serialized.get("content"),
            "internal": {
                "query_id": query_id,
                "raw_response": serialized,
            },
        }


def run_benchmark(
    *,
    system_prompt: str,
    inputs_for_llm: List[Dict[str, Any]],
    inputs_for_logging: List[Dict[str, Any]],
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    use_retail: bool = True,
    use_weather: bool = True,
    use_translate: bool = True,
    use_calculator: bool = True,
    use_trash: bool = False,
    use_airbnb: bool = False,
    use_flights: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    
    """
    Запускает все запросы через агента и собирает результаты в словарь.
    
    Args:
        system_prompt: Системный промпт для агента
        inputs_for_llm: Список запросов для модели
        inputs_for_logging: Список ground truth данных
        model: Название модели
        verbose: Выводить прогресс
    
    Returns:
        Словарь с результатами бенчмарка
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print("RUSSIAN TOOL ВЫЗОВ БЕНЧМАРКА")
        print(f"{'=' * 70}")
        print(f"Модель: {model}")
        print(f"Всего запросов: {len(inputs_for_llm)}")
        print(f"{'=' * 70}\n")

    agent = BenchmarkAgent(
        model=model,
        use_retail=use_retail,
        use_weather=use_weather,
        use_translate=use_translate,
        use_calculator=use_calculator,
        use_trash=use_trash,
        use_airbnb=use_airbnb,
        use_flights=use_flights,
        verbose=False,
    )

    if verbose:
        print(f"Агент инициализирован с {len(agent.get_tools_info())} инструментами\n")

    results: Dict[str, Any] = {}
    started = time.time()

    # Прогоняем через агента
    for idx, (request_data, ground_truth) in enumerate(
        zip(inputs_for_llm, inputs_for_logging),
        1,
    ):
        query_id = request_data["id"]

        if verbose:
            print(f"[{idx}/{len(inputs_for_llm)}] Обработка: {query_id}")
            print(f"Запрос: {request_data['user_query'][:80]}...")

        try:
            # Вызываем агента
            agent_output = agent.run_single_query(
                user_query=request_data["user_query"],
                system_prompt=system_prompt,
                query_id=query_id,
            )

            if verbose:
                if agent_output["tool_call"]:
                    print(f"Инструмент: {agent_output['tool_call']['name']}")
                elif agent_output["clarification_question"]:
                    print("Требуется уточнение")
                elif agent_output["user_message"]:
                    print("Текстовый ответ")
                else:
                    print("Нет вывода")

        except Exception as exc:
            if verbose:
                print(f"Ошибка: {exc}")
            agent_output = {
                "tool_call": None,
                "clarification_question": None,
                "user_message": None,
                "internal": {
                    "reasoning": None,
                    "raw_response": None,
                    "errors": f"RUNNER_ERROR: {exc}",
                },
            }

        results[query_id] = {
            "id": query_id,
            "user_query": request_data["user_query"],
            "agent_response": agent_output,
            "ground_truth": {
                "expected_tool": ground_truth["expected_tool"],
                "expected_parameters": ground_truth["expected_parameters"],
                "requires_clarification": ground_truth["requires_clarification"],
                "skills": ground_truth["skills"],
            },
        }

        if verbose:
            print()

    elapsed = time.time() - started
    if verbose and inputs_for_llm:
        print(f"{'=' * 70}")
        print(f"Бенчмарк завершен за {elapsed:.2f} секунд")
        print(f"Среднее время на запрос: {elapsed / len(inputs_for_llm):.2f}s")
        print(f"{'=' * 70}\n")

    return results


def save_results(results: Dict[str, Any], filename: str = "benchmark_results.json") -> None:
    """Сохраняет результаты бенчмарка на диск.

    Args:
        results: Словарь с выводом агента.
        filename: Имя JSON-файла назначения.
    """
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в: {filename}")


def print_statistics(results: Dict[str, Any]) -> None:
    """
    Выводит базовую статистику по результатам.
    
    Args:
        results: Словарь с результатами бенчмарка
    """
    total = len(results)
    tool_calls = 0
    clarifications = 0
    text_responses = 0
    errors = 0

    for entry in results.values():
        agent_response = entry["agent_response"]
        if agent_response.get("tool_call"):
            tool_calls += 1
        elif agent_response.get("clarification_question"):
            clarifications += 1
        elif agent_response.get("user_message"):
            text_responses += 1

        if agent_response.get("internal", {}).get("errors"):
            errors += 1

    if not total:
        print("\nНет данных для статистики\n")
        return

    print(f"\n{'=' * 70}")
    print("Стастистика")
    print(f"{'=' * 70}")
    print(f"Всего запросов:        {total}")
    print(f"Вызовы инструментов:           {tool_calls} ({tool_calls / total * 100:.1f}%)")
    print(f"Уточняющие вопросы:       {clarifications} ({clarifications / total * 100:.1f}%)")
    print(f"Текстовые ответы:       {text_responses} ({text_responses / total * 100:.1f}%)")
    print(f"Ошибки:               {errors} ({errors / total * 100:.1f}%)")
    print(f"{'=' * 70}\n")


def _load_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """Загружает датасет через `json_parser` и делит его на два списка.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Ключи `inputs_for_llm` и
        `inputs_for_logging`, как в `json_parser.process_all_queries`.

    Raises:
        RuntimeError: Если `json_parser` недоступен для импорта.
    """
    if process_all_queries is None:
        raise RuntimeError(
            "json_parser.py не найден"
        )

    inputs_for_llm, inputs_for_logging = process_all_queries(  # type: ignore
        system_prompt=default_system_prompt,
    )
    return {
        "inputs_for_llm": inputs_for_llm,
        "inputs_for_logging": inputs_for_logging,
    }


def _dataset_requires_tool(
    entries: List[Dict[str, Any]],
    keywords: Tuple[str, ...],
) -> bool:
    """Проверяет, встречаются ли указанные ключевые слова в ground truth.

    Args:
        entries: Строки ground truth с полем `expected_tool`.
        keywords: Подстроки, по которым решаем, нужен ли MCP.

    Returns:
        bool: True, если найден хотя бы один инструмент с таким словом.
    """

    lowered = tuple(keyword.lower() for keyword in keywords)
    for entry in entries:
        expected_tool = str(entry.get("expected_tool", "")).lower()
        if any(keyword in expected_tool for keyword in lowered):
            return True
    return False


def main() -> None:
    """Парсит CLI-аргументы, включает нужные тулзы и запускает бенчмарк."""
    parser = argparse.ArgumentParser(description="Run Russian Tool Calling Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
        help="Model name to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output filename",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--airbnb",
        action="store_true",
        help="Enable Airbnb MCP tools",
    )
    parser.add_argument(
        "--flights",
        action="store_true",
        help="Enable flights MCP tools",
    )

    args = parser.parse_args()

    dataset = _load_dataset()
    inputs_for_llm = dataset["inputs_for_llm"]
    inputs_for_logging = dataset["inputs_for_logging"]

    auto_airbnb = _dataset_requires_tool(inputs_for_logging, ("airbnb",))
    auto_flights = _dataset_requires_tool(inputs_for_logging, ("flight", "duffel"))

    use_airbnb = args.airbnb or auto_airbnb
    use_flights = args.flights or auto_flights

    if use_airbnb and not args.airbnb and auto_airbnb:
        print("Airbnb MCP Подключен")
    if use_flights and not args.flights and auto_flights:
        print("Flights MCP Подключен")

    results = run_benchmark(
        system_prompt=default_system_prompt,
        inputs_for_llm=inputs_for_llm,
        inputs_for_logging=inputs_for_logging,
        model=args.model,
        use_airbnb=use_airbnb,
        use_flights=use_flights,
        verbose=not args.quiet,
    )

    save_results(results, filename=args.output)
    print_statistics(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nБенчмарк проведен! Проверь {args.output} для подробных результатов.")


if __name__ == "__main__":
    main()
