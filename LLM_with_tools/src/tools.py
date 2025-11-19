from openai import OpenAI
from settings import OpenAISettings
import dotenv
from typing import Any, Dict, List
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

dotenv.load_dotenv(dotenv.find_dotenv())


def get_client() -> OpenAI:
    settings = OpenAISettings()
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=str(settings.openai_base_url),
    )


def _normalize_tool_metadata(tool) -> Dict[str, Any]:
    if hasattr(tool, 'model_dump'):
        return tool.model_dump()
    if isinstance(tool, dict):
        return tool
    try:
        return dict(tool)
    except Exception:
        return {}


def convert_tools_metadata_to_openai(tools_meta: List[Any], *, input_key: str = "inputSchema", strict: bool = False) -> List[Dict[str, Any]]:
    openai_tools: List[Dict[str, Any]] = []

    for raw in tools_meta:
        tool = _normalize_tool_metadata(raw)
        input_schema = tool.get(input_key, {}) or {}

        params = {
            "type": input_schema.get("type", "object"),
            "properties": input_schema.get("properties", {}),
            "required": input_schema.get("required", []),
        }

        if strict:
            params["additionalProperties"] = False

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": params,
                **({"strict": True} if strict else {}),
            },
        })

    return openai_tools


def convert_mcp_tools_to_openai(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
    return convert_tools_metadata_to_openai(mcp_tools, input_key="inputSchema", strict=True)


def convert_retail_tools_to_openai(retail_tools_metadata: List[Dict[str, Any]], strict: bool = False) -> List[Dict[str, Any]]:
    return convert_tools_metadata_to_openai(retail_tools_metadata, input_key="parameters", strict=strict)


def convert_retail_tools_to_openai(retail_tools_metadata: List[Dict[str, Any]], strict: bool = False) -> List[Dict[str, Any]]:
    openai_tools = []
    
    for retail_tool in retail_tools_metadata:
        parameters = retail_tool.get("parameters", {})
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": retail_tool.get("name", ""),
                "description": retail_tool.get("description", ""),
                "parameters": {
                    "type": parameters.get("type", "object"),
                    "properties": parameters.get("properties", {}),
                    "required": parameters.get("required", [])
                }
            }
        }
        
        if strict:
            openai_tool["function"]["parameters"]["additionalProperties"] = False
            openai_tool["function"]["strict"] = True
        
        openai_tools.append(openai_tool)
    
    return openai_tools


def get_retail_executors():
    """Return a mapping of retail tool names to executor callables.

    Uses a simple list of expected method names and maps them to `EcommerceTools`.
    This keeps the mapping compact and easy to extend.
    """
    from retail_tools import EcommerceTools

    method_names = [
        "cancel_order", "search_products", "return_order", "place_order", "track_order",
        "update_address", "add_to_cart", "remove_from_cart", "update_payment_method",
        "apply_discount_code", "get_order_history", "schedule_delivery", "update_profile",
        "contact_support",
    ]

    executors = {}
    for name in method_names:
        if hasattr(EcommerceTools, name):
            executors[name] = getattr(EcommerceTools, name)

    return executors


def get_weather_executors():
    """Weather/currency executors."""
    from weather_and_convert import MiscTools
    return {"get_weather": MiscTools.get_weather, "currency_converter": MiscTools.currency_converter}


def get_trash_executors():
    """Null/trash executors (test stubs)."""
    from trash import NullTools

    names = [
        "cancel_order_T", "search_products_T", "return_order_T", "place_order_T", "track_order_T",
        "update_address_T", "add_to_cart_T", "remove_from_cart_T", "update_payment_method_T",
        "apply_discount_code_T", "get_order_history_T", "schedule_delivery_T", "update_profile_T",
        "contact_support_T", "calculator_T", "get_weather_T", "translate_T", "currency_converter_T",
        "get_time_T", "get_date_T", "ping_T", "noop_T", "faulty_gateway_T",
    ]

    return {n: getattr(NullTools, n) for n in names if hasattr(NullTools, n)}


def get_translate_executors():
    from trans import TranslateTools
    return {"translate": TranslateTools.translate}


def get_calculator_executors():
    from caclute import CalculatorTool
    return {"calculator": CalculatorTool.calculate}


async def execute_tool_call(tool_call, executors, mcp_clients=None):
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    print(f"\nCalling: {tool_name}")
    print(f"Args: {json.dumps(arguments, ensure_ascii=False)}")
    
    try:
        if tool_name in executors:
            result = executors[tool_name](**arguments)
            print(f"Success: {json.dumps(result, ensure_ascii=False)}")
            return (tool_call.id, json.dumps(result, ensure_ascii=False))
        
        elif mcp_clients:
            for client_name, mcp_client in mcp_clients.items():
                try:
                    result = await mcp_client.call_tool(tool_name, arguments)
                    if hasattr(result, 'content'):
                        result_text = ""
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                result_text += content_item.text
                        print(f"Success ({client_name}): {result_text[:200]}...")
                        return (tool_call.id, result_text)
                    else:
                        result_str = str(result)
                        print(f"Success ({client_name}): {result_str[:200]}...")
                        return (tool_call.id, result_str)
                except Exception as client_error:
                    continue
            
            error_msg = f"Tool {tool_name} not found in any MCP client"
            print(f"  ⚠️  {error_msg}")
            return (tool_call.id, json.dumps({"error": error_msg}, ensure_ascii=False))
        
        else:
            error_msg = f"Tool {tool_name} not found"
            print(f"  ⚠️  {error_msg}")
            return (tool_call.id, json.dumps({"error": error_msg}, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}")
        return (tool_call.id, json.dumps({"error": str(e)}, ensure_ascii=False))


async def process_tool_calls(response, messages, openai_tools, executors, openai_client, model="Qwen/Qwen3-Next-80B-A3B-Instruct", max_iterations=5, mcp_clients=None):

    iteration = 0
    
    while response.choices[0].message.tool_calls and iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"🔧 Tool calls iteration {iteration}:")
        print('='*60)
        
        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": []
        }
        
        for tool_call in response.choices[0].message.tool_calls:
            assistant_message["tool_calls"].append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
            
            tool_call_id, result_content = await execute_tool_call(tool_call, executors, mcp_clients)
            
            messages.append(assistant_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content
            })
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto"
        )
        
        if response.choices[0].message.content:
            print(f"\nModel: {response.choices[0].message.content}")
    
    return response


async def interactive_chat(use_airbnb=False, use_retail=True, use_flights=False, use_weather=False, use_trash=False, use_translate=False, use_calculator=False):
    
    openai_client = get_client()
    openai_tools = []
    executors = {}
    mcp_clients = {} 
    
    # Загрузка инструментов ретейла
    if use_retail:
        from retail_tools import EcommerceTools
        retail_tools = EcommerceTools.get_tools_metadata()
        retail_openai_tools = convert_retail_tools_to_openai(retail_tools, strict=False)
        openai_tools.extend(retail_openai_tools)
        executors.update(get_retail_executors())
        print(f"Loaded {len(retail_tools)} retail tools")
    
    # Загрузка инструментов погоды и валюты
    if use_weather:
        from weather_and_convert import MiscTools
        weather_tools = MiscTools.get_tools_metadata()
        weather_openai_tools = convert_retail_tools_to_openai(weather_tools, strict=False)
        openai_tools.extend(weather_openai_tools)
        executors.update(get_weather_executors())
        print(f"Loaded {len(weather_tools)} weather/currency tools")
    
    # Загрузка инструмента перевода
    if use_translate:
        from trans import TranslateTools
        translate_tools = TranslateTools.get_tools_metadata()
        translate_openai_tools = convert_retail_tools_to_openai(translate_tools, strict=False)
        openai_tools.extend(translate_openai_tools)
        executors.update(get_translate_executors())
        print(f"Loaded {len(translate_tools)} translate tool")
    
    # Загрузка инструмента калькулятора
    if use_calculator:
        from caclute import CalculatorTool
        calculator_tools = CalculatorTool.get_tools_metadata()
        calculator_openai_tools = convert_retail_tools_to_openai(calculator_tools, strict=False)
        openai_tools.extend(calculator_openai_tools)
        executors.update(get_calculator_executors())
        print(f"Loaded {len(calculator_tools)} calculator tool")
    
    # Загрузка инструментов trash (null)
    if use_trash:
        from trash import NullTools
        trash_tools = NullTools.get_tools_metadata()
        trash_openai_tools = convert_retail_tools_to_openai(trash_tools, strict=False)
        openai_tools.extend(trash_openai_tools)
        executors.update(get_trash_executors())
        print(f"Loaded {len(trash_tools)} trash (null) tools")
    
    # Загрузка инструментов Airbnb MCP
    if use_airbnb:
        from fastmcp import Client
        
        mcp_config = {
            "mcpServers": {
                "airbnb": {
                    "command": "npx",
                    "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
                }
            }
        }
        
        airbnb_client = Client(mcp_config)
        await airbnb_client.__aenter__()
        mcp_clients["airbnb"] = airbnb_client
        
        mcp_tools_response = await airbnb_client.list_tools()
        if hasattr(mcp_tools_response, 'tools'):
            mcp_tools = list(mcp_tools_response.tools)
        else:
            mcp_tools = mcp_tools_response
        
        mcp_openai_tools = convert_mcp_tools_to_openai(mcp_tools)
        openai_tools.extend(mcp_openai_tools)
        print(f" Loaded {len(mcp_tools)} Airbnb MCP tools")
    
    # Загрузка инструментов Flights MCP
    if use_flights:
        from fastmcp import Client

        duffel_api_key = os.getenv("DUFFEL_API_KEY_LIVE", "")
        if not duffel_api_key:
            print("Warning: DUFFEL_API_KEY_LIVE not set in environment")
        
        flights_mcp_path = os.path.join(os.path.dirname(__file__), "flights-mcp")
        
        flights_config = {
            "mcpServers": {
                "flights": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        flights_mcp_path,
                        "run",
                        "flights-mcp"
                    ],
                    "env": {
                        "DUFFEL_API_KEY_LIVE": duffel_api_key
                    }
                }
            }
        }
        
        flights_client = Client(flights_config)
        await flights_client.__aenter__()
        mcp_clients["flights"] = flights_client
        
        flights_tools_response = await flights_client.list_tools()
        if hasattr(flights_tools_response, 'tools'):
            flights_tools = list(flights_tools_response.tools)
        else:
            flights_tools = flights_tools_response
        
        flights_openai_tools = convert_mcp_tools_to_openai(flights_tools)
        openai_tools.extend(flights_openai_tools)
        print(f"Loaded {len(flights_tools)} Flights MCP tools")
    
    
    print("\n" + "-"*70)
    print("Доступные инструменты:")
    for i, tool in enumerate(openai_tools, 1):
        tool_func = tool.get("function", {})
        print(f"  {i:2d}. {tool_func.get('name', 'unknown'):<25} - {tool_func.get('description', 'No description')[:40]}")
    
    print("\n" + "="*70)
    print("Напиши 'list' чтобы увидеть все инструменты")
    print("Напиши 'clear' если хочешь очистить историю")
    print("Напиши 'quit' чтобы выйти")
    print("="*70)
    
    messages = []
    
    try:
        while True:
            print("\n" + "-"*70)
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'выход', 'q']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() in ['list', 'список']:
                print("\nAvailable tools:")
                for i, tool in enumerate(openai_tools, 1):
                    tool_func = tool.get("function", {})
                    print(f"  {i:2d}. {tool_func.get('name', 'unknown'):<25} - {tool_func.get('description', 'No description')}")
                continue
            
            if user_input.lower() in ['clear', 'очистить']:
                messages = []
                print("History cleared")
                continue
            
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            print("\nLLM thinking...")
            
            try:
                response = openai_client.chat.completions.create(
                    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )
                
                if response.choices[0].message.tool_calls:
                    print(f"\nLLM selected {len(response.choices[0].message.tool_calls)} tool(s)")
                    response = await process_tool_calls(
                        response, messages, openai_tools, executors, openai_client, mcp_clients=mcp_clients
                    )
                else:
                    if response.choices[0].message.content:
                        print(f"\nAssistant: {response.choices[0].message.content}")
                        
                if response.choices[0].message.content:
                    messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    })
                    
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
    
    finally:
        for client_name, mcp_client in mcp_clients.items():
            try:
                await mcp_client.__aexit__(None, None, None)
            except:
                pass

async def example_simple_tools():
    mcp_tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a given location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Paris, France"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location", "units"]
            }
        },
        {
            "name": "multiply",
            "description": "Multiply two numbers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    ]

    openai_tools = convert_mcp_tools_to_openai(mcp_tools)
    client = get_client()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Paris, France in celsius?"
            }
        ],
        tools=openai_tools,
        tool_choice="auto"
    )
    
    print("Response:")
    print(response.choices[0].message)
    
    if response.choices[0].message.tool_calls:
        print("\nTool calls:")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"  - {tool_call.function.name}({tool_call.function.arguments})")


if __name__ == "__main__":
    import sys
    
    use_airbnb = False
    use_retail = False
    use_flights = False
    use_weather = False
    use_trash = False
    use_translate = False
    use_calculator = False
    mode = "interactive"
    explicit_tools = False
    
    for arg in sys.argv[1:]:
        if arg in ["--airbnb", "-a"]:
            use_airbnb = True
            explicit_tools = True
        elif arg in ["--retail", "-r"]:
            use_retail = True
            explicit_tools = True
        elif arg in ["--flights", "-f"]:
            use_flights = True
            explicit_tools = True
        elif arg in ["--weather", "-w"]:
            use_weather = True
            explicit_tools = True
        elif arg in ["--trash", "-t"]:
            use_trash = True
            explicit_tools = True
        elif arg in ["--translate", "-tr"]:
            use_translate = True
            explicit_tools = True
        elif arg in ["--calculator", "-c"]:
            use_calculator = True
            explicit_tools = True
        elif arg in ["--all"]:
            use_airbnb = True
            use_retail = True
            use_flights = True
            use_weather = True
            use_trash = True
            use_translate = True
            use_calculator = True
            explicit_tools = True
        elif arg == "--simple":
            mode = "simple"
    
    if not explicit_tools:
        use_airbnb = True
        use_retail = True
        use_weather = True
        use_translate = True
        use_calculator = True

    if mode == "interactive":
        print("\nStarting interactive mode...")
        
        tool_sets = []
        if use_retail:
            tool_sets.append("Retail (14)")
        if use_airbnb:
            tool_sets.append("Airbnb MCP (2)")
        if use_flights:
            tool_sets.append("Flights MCP (?)")
        if use_weather:
            tool_sets.append("Weather (2)")
        if use_translate:
            tool_sets.append("Translate (1)")
        if use_calculator:
            tool_sets.append("Calculator (1)")
        if use_trash:
            tool_sets.append("Trash (23)")
        
        if tool_sets:
            print(f"   Tools: {' + '.join(tool_sets)}")
        else:
            print("   No tools selected")
        
        asyncio.run(interactive_chat(
            use_airbnb=use_airbnb,
            use_retail=use_retail,
            use_flights=use_flights,
            use_weather=use_weather,
            use_trash=use_trash,
            use_translate=use_translate,
            use_calculator=use_calculator
        ))
    elif mode == "simple":
        print("\n🔧 SIMPLE TOOLS EXAMPLE")
        print("="*60)
        print("\nRunning simple example with hardcoded tools...")
        print("\n💡 Use --help to see all options")
        print()
        asyncio.run(example_simple_tools())