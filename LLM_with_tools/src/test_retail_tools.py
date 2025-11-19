#!/usr/bin/env python3
"""
Test script demonstrating retail e-commerce tools integration with OpenAI
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from tools import get_client, convert_retail_tools_to_openai
from retail_tools import EcommerceTools
import json


async def test_scenario_1():
    """Сценарий 1: Поиск и добавление товара в корзину"""
    print("\n" + "="*60)
    print("СЦЕНАРИЙ 1: Поиск и покупка товара")
    print("="*60)
    
    retail_tools_metadata = EcommerceTools.get_tools_metadata()
    openai_tools = convert_retail_tools_to_openai(retail_tools_metadata, strict=False)
    
    executors = {
        "search_products": EcommerceTools.search_products,
        "add_to_cart": EcommerceTools.add_to_cart,
        "place_order": EcommerceTools.place_order,
    }
    
    openai_client = get_client()
    
    messages = [
        {
            "role": "user",
            "content": "Найди мне смартфон до 300 долларов и добавь самый дешевый в корзину"
        }
    ]
    
    response = openai_client.chat.completions.create(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto"
    )
    
    # Process tool calls
    for _ in range(3):  # Max 3 iterations
        if not response.choices[0].message.tool_calls:
            break
            
        for tool_call in response.choices[0].message.tool_calls:
            print(f"\n🔧 Tool: {tool_call.function.name}")
            args = json.loads(tool_call.function.arguments)
            print(f"📝 Args: {json.dumps(args, ensure_ascii=False)}")
            
            executor = executors.get(tool_call.function.name)
            if executor:
                result = executor(**args)
                print(f"✅ Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        response = openai_client.chat.completions.create(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto"
        )
    
    print(f"\n💬 Final Response: {response.choices[0].message.content}")


async def test_scenario_2():
    """Сценарий 2: Отслеживание и отмена заказа"""
    print("\n" + "="*60)
    print("СЦЕНАРИЙ 2: Управление заказами")
    print("="*60)
    
    retail_tools_metadata = EcommerceTools.get_tools_metadata()
    openai_tools = convert_retail_tools_to_openai(retail_tools_metadata, strict=False)
    
    executors = {
        "track_order": EcommerceTools.track_order,
        "cancel_order": EcommerceTools.cancel_order,
        "get_order_history": EcommerceTools.get_order_history,
    }
    
    openai_client = get_client()
    
    messages = [
        {
            "role": "user",
            "content": "Проверь статус моего заказа ORD12345, а потом отмени его"
        }
    ]
    
    response = openai_client.chat.completions.create(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto"
    )
    
    # Process tool calls
    for _ in range(3):
        if not response.choices[0].message.tool_calls:
            break
            
        for tool_call in response.choices[0].message.tool_calls:
            print(f"\n🔧 Tool: {tool_call.function.name}")
            args = json.loads(tool_call.function.arguments)
            print(f"📝 Args: {json.dumps(args, ensure_ascii=False)}")
            
            executor = executors.get(tool_call.function.name)
            if executor:
                result = executor(**args)
                print(f"✅ Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        response = openai_client.chat.completions.create(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto"
        )
    
    print(f"\n💬 Final Response: {response.choices[0].message.content}")


async def test_scenario_3():
    """Сценарий 3: Применение промокода"""
    print("\n" + "="*60)
    print("СЦЕНАРИЙ 3: Использование промокода")
    print("="*60)
    
    retail_tools_metadata = EcommerceTools.get_tools_metadata()
    openai_tools = convert_retail_tools_to_openai(retail_tools_metadata, strict=False)
    
    executors = {
        "apply_discount_code": EcommerceTools.apply_discount_code,
    }
    
    openai_client = get_client()
    
    messages = [
        {
            "role": "user",
            "content": "Примени промокод SAVE10 к моему заказу"
        }
    ]
    
    response = openai_client.chat.completions.create(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto"
    )
    
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            print(f"\n🔧 Tool: {tool_call.function.name}")
            args = json.loads(tool_call.function.arguments)
            print(f"📝 Args: {json.dumps(args, ensure_ascii=False)}")
            
            executor = executors.get(tool_call.function.name)
            if executor:
                result = executor(**args)
                print(f"✅ Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        response = openai_client.chat.completions.create(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto"
        )
    
    print(f"\n💬 Final Response: {response.choices[0].message.content}")


async def main():
    """Run all test scenarios"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ RETAIL E-COMMERCE TOOLS")
    print("="*60)
    
    try:
        await test_scenario_1()
        await test_scenario_2()
        await test_scenario_3()
        
        print("\n" + "="*60)
        print("✅ Все тесты успешно выполнены!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
