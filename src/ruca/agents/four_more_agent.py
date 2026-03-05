"""Benchmark agent module for evaluating LLM tool-calling capabilities.

Provides :class:`BenchmarkAgent` that orchestrates tool registration, execution,
and chained tool-call evaluation against a ground-truth dataset.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import dotenv
from openai import OpenAI

from ruca.settings import OpenAISettings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

dotenv.load_dotenv(dotenv.find_dotenv())

try:
    from ruca.utils import process_all_queries
    from ruca.utils import system_prompt as default_system_prompt
except ImportError:
    process_all_queries = None
    default_system_prompt = "Ты — ассистент, вызывающий инструменты."

DEFAULT_MODEL: str = "openai/gpt-oss-20b"


class BenchmarkAgent:
    """Agent that registers tool sets, calls an LLM, and executes tool invocations.

    Use the async factory :meth:`create` instead of the constructor directly.
    """

    def __init__(self) -> None:
        """Base constructor — do not use directly, use :meth:`create` instead."""
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
    ) -> "BenchmarkAgent":
        """Async factory that creates and fully initialises a BenchmarkAgent.

        Args:
            model: Model identifier string.
            use_retail: Register e-commerce tools.
            use_weather: Register weather / currency tools.
            use_translate: Register translation tools.
            use_calculator: Register calculator tools.
            use_trash: Register null / distractor tools.
            use_aviation: Register aviation tools.
            use_datetime: Register date-time tools.
            use_airbnb: Connect to the Airbnb MCP server.
            verbose: Print initialisation diagnostics.

        Returns:
            Fully initialised :class:`BenchmarkAgent` instance.
        """
        instance = cls()

        instance.model = model
        instance.verbose = verbose
        instance.openai_client = cls._build_client()
        instance.openai_tools = []
        instance.executors = {}
        instance.mcp_clients = {}

        if use_retail:
            from ruca.tools import EcommerceTools

            retail_tools = EcommerceTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(retail_tools))
            instance.executors.update(instance._get_retail_executors())

        if use_weather:
            from ruca.tools import MiscTools

            weather_tools = MiscTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(weather_tools))
            instance.executors.update(instance._get_weather_executors())

        if use_translate:
            from ruca.tools import TranslateTools

            translate_tools = TranslateTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(translate_tools))
            instance.executors.update(instance._get_translate_executors())

        if use_calculator:
            from ruca.tools import CalculatorTool

            calculator_tools = CalculatorTool.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(calculator_tools))
            instance.executors.update(instance._get_calculator_executors())

        if use_trash:
            from ruca.tools import NullTools

            trash_tools = NullTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(trash_tools))
            instance.executors.update(instance._get_trash_executors())

        if use_aviation:
            from ruca.tools import AviationTools

            aviation_tools = AviationTools.get_tools_metadata()
            instance.openai_tools.extend(instance._convert_tools(aviation_tools))
            instance.executors.update(instance._get_aviation_executors())

        if use_datetime:
            from ruca.tools import DateTimeTools

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
        """Build an authenticated OpenAI client from application settings."""
        settings = OpenAISettings()
        return OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url),
        )

    def _convert_tools(self, tools_meta: list[Any], *, strict: bool = False) -> list[dict[str, Any]]:
        """Convert arbitrary tool metadata dicts into OpenAI function-calling format.

        Args:
            tools_meta: List of tool description dicts.
            strict: If ``True``, enable strict mode (no additional properties).

        Returns:
            List of OpenAI-compatible tool definitions.
        """
        converted: list[dict[str, Any]] = []
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

    def _get_retail_executors(self) -> dict[str, Any]:
        """Return a name -> callable mapping for e-commerce tool methods."""
        from ruca.tools import EcommerceTools

        method_names: list[str] = [
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
        return {name: getattr(EcommerceTools, name) for name in method_names if hasattr(EcommerceTools, name)}

    def _get_aviation_executors(self) -> dict[str, Any]:
        """Return a name -> callable mapping for aviation tool methods."""
        from ruca.tools import AviationTools

        method_names: list[str] = [
            "BookingService",
            "FlightStatusService",
            "CheckInService",
            "UpgradeService",
            "PaymentService",
            "LoyaltyService",
            "BaggageService",
            "SeatMapService",
            "RefundService",
            "RebookingService",
            "AncillariesService",
            "InsuranceService",
            "CargoService",
            "LostAndFoundService",
            "OpsService",
            "HotelService",
            "CompensationService",
            "IdentityVerificationService",
        ]
        return {name: getattr(AviationTools, name) for name in method_names if hasattr(AviationTools, name)}

    def _get_datetime_executors(self) -> dict[str, Any]:
        """Return a name -> callable mapping for date/time tool methods."""
        from ruca.tools import DateTimeTools

        return {
            "get_date": DateTimeTools.get_date,
            "get_time": DateTimeTools.get_time,
        }

    def _get_weather_executors(self) -> dict[str, Any]:
        """Return a name -> callable mapping for weather / currency tool methods."""
        from ruca.tools import MiscTools

        return {
            "get_weather": MiscTools.get_weather,
            "currency_converter": MiscTools.currency_converter,
        }

    def _get_translate_executors(self) -> dict[str, Any]:
        """Return a name -> callable mapping for translation tool methods."""
        from ruca.tools import TranslateTools

        return {"translate": TranslateTools.translate}

    def _get_calculator_executors(self) -> dict[str, Any]:
        """Return a name -> callable mapping for calculator tool methods."""
        from ruca.tools import CalculatorTool

        return {"calculator": CalculatorTool.calculate}

    def _get_trash_executors(self) -> dict[str, Any]:
        """Return a name -> callable mapping for null / distractor tool methods."""
        from ruca.tools import NullTools

        return {name: getattr(NullTools, name) for name in dir(NullTools) if not name.startswith("_")}

    async def _setup_mcp_clients(self, *, use_airbnb: bool) -> None:
        """Initialise and connect MCP (Model Context Protocol) clients.

        Args:
            use_airbnb: If ``True``, start and connect to the Airbnb MCP server.
        """
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
        """Fetch tool definitions from all connected MCP clients and register them."""
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
        """Safely retrieve a string attribute from a tool (dict or object)."""
        if isinstance(tool, dict):
            return str(tool.get(key, ""))
        return str(getattr(tool, key, ""))

    @staticmethod
    def _extract_schema(tool: Any) -> dict[str, Any]:
        """Extract and normalise the JSON-Schema parameters block from a tool."""

        def _normalize(schema: Any) -> dict[str, Any]:
            """Ensure *schema* is a valid ``object``-type JSON Schema dict."""
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

    def get_tools_info(self) -> list[dict[str, str]]:
        """Return a lightweight summary (name + description) for every registered tool."""
        return [
            {
                "name": tool["function"].get("name", ""),
                "description": tool["function"].get("description", ""),
            }
            for tool in self.openai_tools
        ]

    async def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call locally or via an MCP client.

        Args:
            tool_name: Registered name of the tool to invoke.
            arguments: Keyword arguments to forward to the tool.

        Returns:
            Dict with ``tool_call`` (name + parameters) and ``result``.
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

    async def run_single_query_async(
        self,
        *,
        user_query: str,
        system_prompt: str,
        query_id: str,
        max_chain_length: int = 5,
        temperature: float = 0.5,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Send a user query to the model and handle chained tool calls.

        The method iterates up to *max_chain_length* times, forwarding tool
        results back into the conversation until the model stops requesting
        tools or signals chain completion.

        Args:
            user_query: The end-user query text.
            system_prompt: System-level instruction for the model.
            query_id: Unique identifier of the query (used for logging).
            max_chain_length: Maximum number of tool-call iterations.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling probability mass.
            top_k: Top-k sampling parameter.
            seed: Random seed for reproducibility.

        Returns:
            A dict containing:
                - **tool_calls** – list of tool-call dicts (chained) or ``None``.
                - **tool_call** – last single tool-call dict (backward compat) or ``None``.
                - **tool_results** – list of all tool results or ``None``.
                - **tool_result** – result of the last tool call or ``None``.
                - **is_chain** – ``True`` when more than one tool was called.
                - **user_message** – plain-text response when no tool was called.
                - **clarification_question** – follow-up question from the model.
                - **assistant_response** – raw assistant content string.
                - **internal** – diagnostics (reasoning, token counts, history).
        """
        settings = OpenAISettings()
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[Any] = []
        conversation_history: list[dict[str, Any]] = []
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0
        total_tokens_used: int = 0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

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

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp,
            ):
                data = await resp.json()

            # Extract per-request token usage
            usage: dict[str, int] = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Accumulate token counts across chain iterations
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens_used += total_tokens

            message: dict[str, Any] = data["choices"][0]["message"]
            content: str = message.get("content") or message.get("reasoning_content") or ""

            # Persist iteration details for diagnostics
            conversation_history.append(
                {
                    "iteration": iteration,
                    "response": content,
                    "tokens": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }
            )

            # Attempt to parse a structured JSON response from the model
            try:
                parsed: dict[str, Any] = json.loads(content)
            except Exception:
                parsed = {}

            tool_call_data: dict[str, Any] | None = parsed.get("tool_call")

            # Stop the chain if the model did not request a tool
            if not tool_call_data or not tool_call_data.get("called"):
                break

            # Execute the requested tool
            tool_name: str = tool_call_data["tool_name"]
            params: dict[str, Any] = tool_call_data.get("parameters", {})
            tool_result: dict[str, Any] = await self._execute_tool_call(tool_name, params)

            all_tool_calls.append(tool_result["tool_call"])
            all_tool_results.append(tool_result["result"])

            # Feed the tool result back into the conversation for the next iteration
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Результат выполнения {tool_name}: {json.dumps(tool_result['result'], ensure_ascii=False)}",
                }
            )

            # Honour explicit chain-completion signal from the model
            if parsed.get("chain_complete", False):
                break

        # Build the final response payload
        is_chain: bool = len(all_tool_calls) > 1

        # Truncate reasoning to the first sentence for brevity
        internal: dict[str, Any] = parsed.get("internal", {})
        if "reasoning" in internal:
            internal["reasoning"] = internal["reasoning"].split(".")[0]

        internal["conversation_history"] = conversation_history
        internal["iterations"] = len(all_tool_calls)
        internal["tokens"] = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens_used,
        }

        result: dict[str, Any] = {
            # Chain-level fields
            "tool_calls": all_tool_calls if all_tool_calls else None,
            "tool_results": all_tool_results if all_tool_results else None,
            "is_chain": is_chain,
            # Backward-compatible single-call fields
            "tool_call": all_tool_calls[-1] if all_tool_calls else None,
            "tool_result": all_tool_results[-1] if all_tool_results else None,
            # Miscellaneous fields
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
    ) -> dict[str, Any]:
        """Synchronous wrapper around :meth:`run_single_query_async`."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.run_single_query_async(user_query=user_query, system_prompt=system_prompt, query_id=query_id)
        )


async def run_benchmark_async(
    *,
    system_prompt: str,
    inputs_for_llm: list[dict[str, Any]],
    inputs_for_logging: list[dict[str, Any]],
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
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the full benchmark suite asynchronously.

    Initialises a :class:`BenchmarkAgent`, fans out queries with a concurrency
    semaphore, and collects results together with run configuration metadata.

    Args:
        system_prompt: System-level prompt prepended to every query.
        inputs_for_llm: Per-query dicts sent to the model.
        inputs_for_logging: Matching ground-truth dicts used for evaluation.
        model: Model identifier string.
        use_retail: Register e-commerce tools.
        use_weather: Register weather / currency tools.
        use_translate: Register translation tools.
        use_calculator: Register calculator tools.
        use_trash: Register null / distractor tools.
        use_aviation: Register aviation tools.
        use_datetime: Register date-time tools.
        use_airbnb: Connect to the Airbnb MCP server.
        verbose: Print progress diagnostics.
        max_concurrent: Maximum number of simultaneous requests.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability mass.
        top_k: Top-k sampling parameter.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``config`` (run parameters) and ``results`` (per-query outcomes).
    """
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

    results: dict[str, Any] = {}
    started: float = time.time()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_query(
        idx: int, request_data: dict[str, Any], ground_truth: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Process one benchmark query under the concurrency semaphore."""
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
                        print(
                            f"Цепочка из {len(agent_output['tool_calls'])} инструментов: "
                            + " -> ".join([tc["name"] for tc in agent_output["tool_calls"]])
                        )
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
        for idx, (request_data, ground_truth) in enumerate(zip(inputs_for_llm, inputs_for_logging, strict=True), 1)
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

    elapsed: float = time.time() - started
    if verbose and inputs_for_llm:
        print(f"{'=' * 70}")
        print(f"Бенчмарк завершен за {elapsed:.2f} секунд")
        print(f"Среднее время на запрос: {elapsed / len(inputs_for_llm):.2f}s")
        print(f"{'=' * 70}\n")

    # Attach run configuration metadata to the results
    api_key: str = settings.openai_api_key.get_secret_value()
    api_key_masked: str = api_key[:4] + "***" if len(api_key) > 4 else "***"

    results_with_config: dict[str, Any] = {
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


def save_results(results: dict[str, Any], filename: str = "benchmark_results.json") -> None:
    """Persist benchmark results to a JSON file.

    If *results* already contains ``config`` and ``results`` keys the dict is
    written as-is; otherwise it is wrapped in a default config envelope.

    Args:
        results: Benchmark output dict.
        filename: Destination file path.
    """
    if "config" in results and "results" in results:
        output: dict[str, Any] = results
    else:
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


def print_statistics(results: dict[str, Any]) -> None:
    """Print a summary table of benchmark outcomes and token usage.

    Args:
        results: Benchmark output dict (optionally wrapped with ``config``).
    """
    # Unwrap if the results dict includes a config envelope
    actual_results: dict[str, Any] = results.get("results", results)
    total: int = len(actual_results)
    tool_calls: int = 0
    chain_calls: int = 0
    clarifications: int = 0
    text_responses: int = 0
    errors: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens_count: int = 0

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

        # Accumulate token usage across all queries
        tokens_info: dict[str, int] = agent_response.get("internal", {}).get("tokens", {})
        total_prompt_tokens += tokens_info.get("prompt_tokens", 0)
        total_completion_tokens += tokens_info.get("completion_tokens", 0)
        total_tokens_count += tokens_info.get("total_tokens", 0)

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
    print(f"{'-' * 70}")
    print("ТОКЕНЫ:")
    print(f"Токены в промпте:            {total_prompt_tokens:,}")
    print(f"Токены в ответах:            {total_completion_tokens:,}")
    print(f"Всего токенов:               {total_tokens_count:,}")
    if total > 0:
        print(f"Среднее токенов на запрос:   {total_tokens_count / total:.0f}")
    print(f"{'=' * 70}\n")


def _load_dataset() -> dict[str, list[dict[str, Any]]]:
    """Load and preprocess query datasets via :func:`process_all_queries`.

    Returns:
        Dict with ``inputs_for_llm`` and ``inputs_for_logging`` lists.

    Raises:
        RuntimeError: If the query-processing utility is unavailable.
    """
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
    entries: list[dict[str, Any]],
    keywords: tuple[str, ...],
) -> bool:
    """Check whether any entry's expected tool name contains one of *keywords*."""
    lowered: tuple[str, ...] = tuple(keyword.lower() for keyword in keywords)
    for entry in entries:
        expected_tool = str(entry.get("expected_tool", "")).lower()
        if any(keyword in expected_tool for keyword in lowered):
            return True
    return False


def main() -> None:
    """CLI entry point: parse arguments, load dataset, run benchmark, and save results."""
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

    print("\nDEBUG: Config from args:")
    print(f"  temperature={args.temperature}")
    print(f"  top_p={args.top_p}")
    print(f"  top_k={args.top_k}")
    print(f"  seed={args.seed}")
    print("DEBUG: Config from results:")
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
