import os
import sys
import json
import asyncio
import dotenv
from typing import Any, Dict, List
from openai import OpenAI
from settings import OpenAISettings


dotenv.load_dotenv(dotenv.find_dotenv())


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))


class BenchmarkAgent:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        use_retail: bool = True,
        use_weather: bool = True,
        use_translate: bool = True,
        use_calculator: bool = True,
        use_trash: bool = False,
        use_airbnb: bool = False,
        use_flights: bool = False,
        verbose: bool = False
    ):
        self.model = model
        self.verbose = verbose
        self.openai_client = self._get_client()
        self.openai_tools: List[Dict[str, Any]] = []
        self.executors: Dict[str, Any] = {}
        self.mcp_clients: Dict[str, Any] = {}

      
        if use_retail:
            from retail_tools import EcommerceTools
            retail_tools = EcommerceTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_retail_tools(retail_tools))
            self.executors.update(self._get_retail_executors())

        if use_weather:
            from weather_and_convert import MiscTools
            weather_tools = MiscTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_retail_tools(weather_tools))
            self.executors.update(self._get_weather_executors())

        if use_translate:
            from trans import TranslateTools
            translate_tools = TranslateTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_retail_tools(translate_tools))
            self.executors.update(self._get_translate_executors())

        if use_calculator:
            from caclute import CalculatorTool
            calculator_tools = CalculatorTool.get_tools_metadata()
            self.openai_tools.extend(self._convert_retail_tools(calculator_tools))
            self.executors.update(self._get_calculator_executors())

        if use_trash:
            from trash import NullTools
            trash_tools = NullTools.get_tools_metadata()
            self.openai_tools.extend(self._convert_retail_tools(trash_tools))
            self.executors.update(self._get_trash_executors())

        if use_airbnb or use_flights:
            from fastmcp import Client
            self._setup_mcp_clients(use_airbnb, use_flights)

        if self.verbose:
            print(f"✅ Agent initialized with {len(self.openai_tools)} tools")

    def _get_client(self) -> OpenAI:
        settings = OpenAISettings()
        return OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url)
        )

    # -------------------- Конвертеры инструментов --------------------
    def _convert_retail_tools(self, tools_meta: List[Any], strict: bool = False) -> List[Dict[str, Any]]:
        openai_tools = []
        for t in tools_meta:
            params = t.get("parameters") or t.get("inputSchema") or {}
            tool_dict = {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": {
                        "type": params.get("type", "object"),
                        "properties": params.get("properties", {}),
                        "required": params.get("required", [])
                    }
                }
            }
            if strict:
                tool_dict["function"]["parameters"]["additionalProperties"] = False
                tool_dict["function"]["strict"] = True
            openai_tools.append(tool_dict)
        return openai_tools

    # -------------------- Executors --------------------
    def _get_retail_executors(self):
        from retail_tools import EcommerceTools
        methods = [
            "cancel_order", "search_products", "return_order", "place_order",
            "track_order", "update_address", "add_to_cart", "remove_from_cart",
            "update_payment_method", "apply_discount_code", "get_order_history",
            "schedule_delivery", "update_profile", "contact_support"
        ]
        return {name: getattr(EcommerceTools, name) for name in methods if hasattr(EcommerceTools, name)}

    def _get_weather_executors(self):
        from weather_and_convert import MiscTools
        return {"get_weather": MiscTools.get_weather, "currency_converter": MiscTools.currency_converter}

    def _get_translate_executors(self):
        from trans import TranslateTools
        return {"translate": TranslateTools.translate}

    def _get_calculator_executors(self):
        from caclute import CalculatorTool
        return {"calculator": CalculatorTool.calculate}

    def _get_trash_executors(self):
        from trash import NullTools
        names = [n for n in dir(NullTools) if not n.startswith("_")]
        return {n: getattr(NullTools, n) for n in names}

    # -------------------- MCP --------------------
    def _setup_mcp_clients(self, use_airbnb: bool, use_flights: bool):
        from fastmcp import Client

        if use_airbnb:
            mcp_config = {
                "mcpServers": {
                    "airbnb": {
                        "command": "npx",
                        "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
                    }
                }
            }
            airbnb_client = Client(mcp_config)
            asyncio.get_event_loop().run_until_complete(airbnb_client.__aenter__())
            self.mcp_clients["airbnb"] = airbnb_client

        if use_flights:
            duffel_api_key = os.getenv("DUFFEL_API_KEY_LIVE", "")
            flights_path = os.path.join(os.path.dirname(__file__), "flights-mcp")
            flights_config = {
                "mcpServers": {
                    "flights": {
                        "command": "uv",
                        "args": ["--directory", flights_path, "run", "flights-mcp"],
                        "env": {"DUFFEL_API_KEY_LIVE": duffel_api_key}
                    }
                }
            }
            flights_client = Client(flights_config)
            asyncio.get_event_loop().run_until_complete(flights_client.__aenter__())
            self.mcp_clients["flights"] = flights_client

    # -------------------- Вспомогательные методы --------------------
    def get_tools_info(self):
        return [{"name": t["function"]["name"], "description": t["function"]["description"]} for t in self.openai_tools]

    async def _execute_tool_call(self, tool_name: str, arguments: dict):
    
        try:
            if tool_name in self.executors:
                result = self.executors[tool_name](**arguments)
                return {
                    "tool_call": {"name": tool_name, "parameters": arguments},
                    "result": result
                }

            elif self.mcp_clients:
                for client_name, mcp_client in self.mcp_clients.items():
                    try:
                        result = await mcp_client.call_tool(tool_name, arguments)
                        if hasattr(result, "content"):
                            result_text = ""
                            for content_item in result.content:
                                if hasattr(content_item, "text"):
                                    result_text += content_item.text
                            return {"tool_call": {"name": tool_name, "parameters": arguments}, "result": result_text}
                        else:
                            return {"tool_call": {"name": tool_name, "parameters": arguments}, "result": str(result)}
                    except Exception:
                        continue

            # Если инструмент не найден
            return {"tool_call": {"name": tool_name, "parameters": arguments}, "result": {"error": "Tool not found"}}

        except Exception as e:
            return {"tool_call": {"name": tool_name, "parameters": arguments}, "result": {"error": str(e)}}

    def get_tools_info(self) -> List[Dict[str, Any]]:
        
        return [
            {
                "name": t.get("function", {}).get("name"),
                "description": t.get("function", {}).get("description", "")
            }
            for t in self.openai_tools
        ]
    # -------------------- Основной метод --------------------
    def run_single_query(self, user_query: str, system_prompt: str, query_id: str) -> Dict[str, Any]:
    
        loop = asyncio.get_event_loop()
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            tools=self.openai_tools,
            tool_choice="auto"
        )

        serialized_response = {
            "content": getattr(response.choices[0].message, "content", None),
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in getattr(response.choices[0].message, "tool_calls", [])
            ]
        }

        tool_calls = getattr(response.choices[0].message, "tool_calls", [])

        if tool_calls:
            tool_call = tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name

          
            tool_result = loop.run_until_complete(self._execute_tool_call(tool_name, arguments))

            return {
                    "tool_call": tool_result["tool_call"],
                    "tool_result": tool_result["result"],  
                    "user_message": None,
                    "clarification_question": None,
                    "assistant_response": serialized_response.get("content"),
                    "internal": {
                        "query_id": query_id,
                        "raw_response": serialized_response
                    }
                }



