import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Optional

import dotenv
import yaml
from openai import OpenAI

from settings import OpenAISettings
from config_loader import get_all_models, resolve_model_params

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

dotenv.load_dotenv(dotenv.find_dotenv())

try:
    from json_parser import process_all_queries, system_prompt as default_system_prompt
except ImportError:
    process_all_queries = None
    default_system_prompt = "Ты — ассистент, вызывающий инструменты."
#Выбор модели wwwwwwww
# ========================================================================
DEFAULT_MODEL = "minimaxai/minimax-m2"
# DEFAULT_MODEL = "openai/gpt-oss-20b"
# DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
# DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
# DEFAULT_MODEL = "openai/gpt-oss-120b"
# ========================================================================

class BenchmarkAgent:

    def __init__(self):
        """Базовый конструктор - не используйте напрямую, используйте create()"""
        pass

    @classmethod
    async def create(
        cls,
        model: str = DEFAULT_MODEL,
        *,
        use_retail: bool = True,
        use_weather: bool = True,
        use_translate: bool = True,
        use_calculator: bool = True,
        use_trash: bool = True,
        use_aviation: bool = True,
        use_datetime: bool = True,
        use_airbnb: bool = False,
        verbose: bool = True,
    ):
        """Асинхронный конструктор для BenchmarkAgent."""
        instance = cls()
        
        instance.model = model
        instance.verbose = verbose
        instance.openai_client = cls._build_client()
        instance.openai_tools = []
        instance.executors = {}
        instance.mcp_clients = {}

        if use_retail:
            from retail_tools import EcommerceTools
            retail_tools = EcommerceTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(retail_tools))
            instance.executors.update(instance._get_retail_executors())

        if use_weather:
            from weather_and_convert import MiscTools
            weather_tools = MiscTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(weather_tools))
            instance.executors.update(instance._get_weather_executors())

        if use_translate:
            from trans import TranslateTools
            translate_tools = TranslateTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(translate_tools))
            instance.executors.update(instance._get_translate_executors())

        if use_calculator:
            from caclute import CalculatorTool
            calculator_tools = CalculatorTool.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(calculator_tools))
            instance.executors.update(instance._get_calculator_executors())

        if use_trash:
            from trash import NullTools
            trash_tools = NullTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(trash_tools))
            instance.executors.update(instance._get_trash_executors())
            
        if use_aviation:
            from avia_tools import AviationTools
            aviation_tools = AviationTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(aviation_tools))
            instance.executors.update(instance._get_aviation_executors())
            
        if use_datetime:
            from datetime_tools import DateTimeTools
            datetime_tools = DateTimeTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(datetime_tools))
            instance.executors.update(instance._get_datetime_executors())

        if use_airbnb:
            await instance._setup_mcp_clients(use_airbnb=True)
            await instance._register_mcp_tools()

        if verbose:
            print(f"Агент инициализирован с {len(instance.openai_tools)} инструментами")
        
        return instance

    @staticmethod
    def _build_client() -> OpenAI:
        """Создаёт аутентифицированный клиент OpenAI."""
        settings = OpenAISettings()
        return OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url),
        )

    def _convert_tools(self, tools_meta: List[Any], *, strict: bool = False) -> List[Dict[str, Any]]:
        """Преобразует произвольные описания тулзов в формат OpenAI."""
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
        from retail_tools import EcommerceTools
        method_names = [
            "cancel_order", "search_products", "return_order", "place_order",
            "track_order", "update_address", "add_to_cart", "remove_from_cart",
            "update_payment_method", "apply_discount_code", "get_order_history",
            "schedule_delivery", "update_profile", "contact_support",
        ]
        return {
            name: getattr(EcommerceTools, name)
            for name in method_names
            if hasattr(EcommerceTools, name)
        }

    def _get_aviation_executors(self) -> Dict[str, Any]:
        from avia_tools import AviationTools
        method_names = [
            "BookingService", "FlightStatusService", "CheckInService",
            "UpgradeService", "PaymentService", "LoyaltyService",
            "BaggageService", "SeatMapService", "RefundService",
            "RebookingService", "AncillariesService", "InsuranceService",
            "CargoService", "LostAndFoundService", "OpsService",
            "HotelService", "CompensationService", "IdentityVerificationService",
        ]
        return {
            name: getattr(AviationTools, name)
            for name in method_names
            if hasattr(AviationTools, name)
        }

    def _get_datetime_executors(self) -> Dict[str, Any]:
        from datetime_tools import DateTimeTools
        return {
            "get_date": DateTimeTools.get_date,
            "get_time": DateTimeTools.get_time,
        }

    def _get_weather_executors(self) -> Dict[str, Any]:
        from weather_and_convert import MiscTools
        return {
            "get_weather": MiscTools.get_weather,
            "currency_converter": MiscTools.currency_converter,
        }

    def _get_translate_executors(self) -> Dict[str, Any]:
        from trans import TranslateTools
        return {"translate": TranslateTools.translate}

    def _get_calculator_executors(self) -> Dict[str, Any]:
        from caclute import CalculatorTool
        return {"calculator": CalculatorTool.calculate}

    def _get_trash_executors(self) -> Dict[str, Any]:
        from trash import NullTools
        return {
            name: getattr(NullTools, name)
            for name in dir(NullTools)
            if not name.startswith("_")
        }

    async def _setup_mcp_clients(self, *, use_airbnb: bool) -> None:
        from fastmcp import Client
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
            await airbnb_client.__aenter__()
            self.mcp_clients["airbnb"] = airbnb_client

    async def _register_mcp_tools(self) -> None:
        if not self.mcp_clients:
            return
        for name, client in self.mcp_clients.items():
            try:
                listing = await client.list_tools()
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
        if isinstance(tool, dict):
            return str(tool.get(key, ""))
        return str(getattr(tool, key, ""))

    @staticmethod
    def _extract_schema(tool: Any) -> Dict[str, Any]:
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
        return [
            {
                "name": tool["function"].get("name", ""),
                "description": tool["function"].get("description", ""),
            }
            for tool in self.openai_tools
        ]

    async def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Выполняет единичный вызов тула локально или через MCP."""
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

    async def run_single_query_async(
        self,
        *,
        user_query: str,
        system_prompt: str,
        query_id: str,
        max_chain_length: int = 5,
        temperature: float = 0.5,
        top_p: float = None,
        top_k: int = None,
        seed: int = None,
    ) -> Dict[str, Any]:
        """
        Асинхронный вызов модели с поддержкой цепочек инструментов.
        
        Args:
            user_query: Запрос пользователя
            system_prompt: Системный промпт
            query_id: ID запроса
            max_chain_length: Максимальная длина цепочки вызовов
            
        Returns:
            Dict с полями:
                - tool_calls: List[Dict] - список вызовов (для цепочек) ИЛИ None
                - tool_call: Dict - один вызов (для обратной совместимости)
                - tool_results: List - результаты всех вызовов
                - tool_result: Any - результат последнего вызова
                - is_chain: bool - флаг цепочки
                - user_message: str|None
                - clarification_question: str|None
                - assistant_response: str
                - internal: Dict
        """
        settings = OpenAISettings()
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        all_tool_calls = []
        all_tool_results = []
        conversation_history = []
        
        import aiohttp
        
        for iteration in range(max_chain_length):
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": temperature,
            }
            
            if top_p is not None:
                payload["top_p"] = top_p
            if top_k is not None:
                payload["top_k"] = top_k

            if self.openai_tools:
                payload["tools"] = self.openai_tools
                payload["tool_choice"] = "auto"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    data = await resp.json()

            message = data["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            
            # Сохраняем историю
            conversation_history.append({
                "iteration": iteration,
                "response": content
            })

            # Парсим JSON из ответа
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {}

            tool_call_data = parsed.get("tool_call")
            
            # Если нет вызова инструмента - заканчиваем цепочку
            if not tool_call_data or not tool_call_data.get("called"):
                break

            # Выполняем инструмент
            tool_name = tool_call_data["tool_name"]
            params = tool_call_data.get("parameters", {})
            tool_result = await self._execute_tool_call(tool_name, params)
            
            all_tool_calls.append(tool_result["tool_call"])
            all_tool_results.append(tool_result["result"])

            # Добавляем результат в историю для следующей итерации
            messages.append({
                "role": "assistant",
                "content": content
            })
            messages.append({
                "role": "user",
                "content": f"Результат выполнения {tool_name}: {json.dumps(tool_result['result'], ensure_ascii=False)}"
            })

            # Если в parsed есть признак завершения цепочки
            if parsed.get("chain_complete", False):
                break

        # Формируем итоговый ответ
        is_chain = len(all_tool_calls) > 1
        
        # Сокращаем reasoning
        internal = parsed.get("internal", {})
        if "reasoning" in internal:
            internal["reasoning"] = internal["reasoning"].split(".")[0]
        
        internal["conversation_history"] = conversation_history
        internal["iterations"] = len(all_tool_calls)

        result = {
            # Новые поля для цепочек
            "tool_calls": all_tool_calls if all_tool_calls else None,
            "tool_results": all_tool_results if all_tool_results else None,
            "is_chain": is_chain,
            
            # Старые поля для обратной совместимости
            "tool_call": all_tool_calls[-1] if all_tool_calls else None,
            "tool_result": all_tool_results[-1] if all_tool_results else None,
            
            # Остальные поля
            "user_message": None if all_tool_calls else parsed.get("user_message"),
            "clarification_question": parsed.get("clarification_question"),
            "assistant_response": content,
            "internal": internal,
        }

        return result

    def run_single_query(
        self,
        *,
        user_query: str,
        system_prompt: str,
        query_id: str,
    ) -> Dict[str, Any]:
        """Синхронная обёртка для run_single_query_async"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.run_single_query_async(
                user_query=user_query,
                system_prompt=system_prompt,
                query_id=query_id
            )
        )


async def run_benchmark_async(
    *,
    system_prompt: str,
    inputs_for_llm: List[Dict[str, Any]],
    inputs_for_logging: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    use_retail: bool = True,
    use_weather: bool = True,
    use_translate: bool = True,
    use_calculator: bool = True,
    use_trash: bool = True,
    use_aviation: bool = True,
    use_datetime: bool = True,
    use_airbnb: bool = False,
    verbose: bool = True,
    max_concurrent: int = 8,
    temperature: float = 0.5,
    top_p: float = None,
    top_k: int = None,
    seed: int = None,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n{'=' * 70}")
        print("RUSSIAN TOOL ВЫЗОВ БЕНЧМАРКА (ASYNC WITH CHAINS)")
        print(f"{'=' * 70}")
        print(f"Модель: {model}")
        print(f"Всего запросов: {len(inputs_for_llm)}")
        print(f"Параллельных запросов: {max_concurrent}")
        print(f"{'=' * 70}\n")

    settings = OpenAISettings()
    
    agent = await BenchmarkAgent.create(
        model=model,
        use_retail=use_retail,
        use_weather=use_weather,
        use_translate=use_translate,
        use_calculator=use_calculator,
        use_trash=use_trash,
        use_aviation=use_aviation,
        use_datetime=use_datetime,
        use_airbnb=use_airbnb,
        verbose=False,
    )

    if verbose:
        print(f"Агент инициализирован с {len(agent.get_tools_info())} инструментами\n")

    results: Dict[str, Any] = {}
    started = time.time()

    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_query(idx: int, request_data: Dict[str, Any], ground_truth: Dict[str, Any]):
        async with semaphore:
            query_id = request_data["id"]

            if verbose:
                print(f"[{idx}/{len(inputs_for_llm)}] Обработка: {query_id}")
                print(f"Запрос: {request_data['user_query'][:80]}...")

            try:
                agent_output = await agent.run_single_query_async(
                    user_query=request_data["user_query"],
                    system_prompt=system_prompt,
                    query_id=query_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )

                if verbose:
                    if agent_output.get("is_chain"):
                        print(f"Цепочка из {len(agent_output['tool_calls'])} инструментов: " + 
                              " -> ".join([tc['name'] for tc in agent_output['tool_calls']]))
                    elif agent_output["tool_call"]:
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
                    "tool_calls": None,
                    "tool_results": None,
                    "is_chain": False,
                    "clarification_question": None,
                    "user_message": None,
                    "internal": {
                        "reasoning": None,
                        "raw_response": None,
                        "errors": f"RUNNER_ERROR: {exc}",
                    },
                }

            return query_id, {
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

    tasks = [
        process_single_query(idx, request_data, ground_truth)
        for idx, (request_data, ground_truth) in enumerate(
            zip(inputs_for_llm, inputs_for_logging), 1
        )
    ]

    completed_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in completed_results:
        if isinstance(result, Exception):
            if verbose:
                print(f"Ошибка при обработке: {result}")
            continue
        query_id, query_result = result
        results[query_id] = query_result

        if verbose:
            print()

    elapsed = time.time() - started
    if verbose and inputs_for_llm:
        print(f"{'=' * 70}")
        print(f"Бенчмарк завершен за {elapsed:.2f} секунд")
        print(f"Среднее время на запрос: {elapsed / len(inputs_for_llm):.2f}s")
        print(f"{'=' * 70}\n")

    # Добавляем метаинформацию о конфигурации
    api_key = settings.openai_api_key.get_secret_value()
    api_key_masked = api_key[:4] + "***" if len(api_key) > 4 else "***"
    
    results_with_config = {
        "config": {
            "model": model,
            "base_url": str(settings.openai_base_url),
            "api_key_prefix": api_key_masked,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        },
        "results": results,
    }

    return results_with_config


def save_results(results: Dict[str, Any], filename: str = "benchmark_results.json") -> None:
    # Если results содержит 'config' и 'results', сохраняем оба
    if "config" in results and "results" in results:
        output = results
    else:
        # Иначе оборачиваем в структуру с config
        output = {
            "config": {
                "model": "unknown",
                "base_url": "unknown",
                "api_key_prefix": "***",
                "temperature": 0.5,
                "top_p": None,
                "top_k": None,
            },
            "results": results,
        }
    
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в: {filename}")


def print_statistics(results: Dict[str, Any]) -> None:
    # Извлекаем результаты, если они обёрнуты в config
    if "results" in results:
        actual_results = results["results"]
    else:
        actual_results = results
    
    total = len(actual_results)
    tool_calls = 0
    chain_calls = 0
    clarifications = 0
    text_responses = 0
    errors = 0

    for entry in actual_results.values():
        agent_response = entry["agent_response"]
        
        if agent_response.get("is_chain"):
            chain_calls += 1
        elif agent_response.get("tool_call"):
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
    print("Статистика")
    print(f"{'=' * 70}")
    print(f"Всего запросов:              {total}")
    print(f"Одиночные вызовы:            {tool_calls} ({tool_calls / total * 100:.1f}%)")
    print(f"Цепочки инструментов:        {chain_calls} ({chain_calls / total * 100:.1f}%)")
    print(f"Уточняющие вопросы:          {clarifications} ({clarifications / total * 100:.1f}%)")
    print(f"Текстовые ответы:            {text_responses} ({text_responses / total * 100:.1f}%)")
    print(f"Ошибки:                      {errors} ({errors / total * 100:.1f}%)")
    print(f"{'=' * 70}\n")


def _load_dataset() -> Dict[str, List[Dict[str, Any]]]:
    if process_all_queries is None:
        raise RuntimeError("json_parser.py не найден")
    inputs_for_llm, inputs_for_logging = process_all_queries(
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
    lowered = tuple(keyword.lower() for keyword in keywords)
    for entry in entries:
        expected_tool = str(entry.get("expected_tool", "")).lower()
        if any(keyword in expected_tool for keyword in lowered):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Russian Tool Calling Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output filename")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--airbnb", action="store_true", help="Enable Airbnb MCP tools")
    parser.add_argument("--concurrent", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature parameter")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p parameter")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k parameter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    dataset = _load_dataset()
    inputs_for_llm = dataset["inputs_for_llm"]
    inputs_for_logging = dataset["inputs_for_logging"]

    auto_airbnb = _dataset_requires_tool(inputs_for_logging, ("airbnb",))
    use_airbnb = args.airbnb or auto_airbnb

    if use_airbnb and not args.airbnb and auto_airbnb:
        print("Airbnb MCP Подключен")

    results = asyncio.run(
        run_benchmark_async(
            system_prompt=default_system_prompt,
            inputs_for_llm=inputs_for_llm,
            inputs_for_logging=inputs_for_logging,
            model=args.model,
            use_airbnb=use_airbnb,
            verbose=not args.quiet,
            max_concurrent=args.concurrent,
            use_retail=True,
            use_weather=True,
            use_translate=True,
            use_calculator=True,
            use_trash=True,
            use_aviation=True,
            use_datetime=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
    )

    print(f"\nDEBUG: Config from args:")
    print(f"  temperature={args.temperature}")
    print(f"  top_p={args.top_p}")
    print(f"  top_k={args.top_k}")
    print(f"  seed={args.seed}")
    print(f"DEBUG: Config from results:")
    if "config" in results:
        print(f"  temperature={results['config'].get('temperature')}")
        print(f"  top_p={results['config'].get('top_p')}")
        print(f"  top_k={results['config'].get('top_k')}")
        print(f"  seed={results['config'].get('seed')}")
    print()

    save_results(results, filename=args.output)
    print_statistics(results)
    print(f"\nБенчмарк проведен! Проверь {args.output} для подробных результатов.")


if __name__ == "__main__":
    main()