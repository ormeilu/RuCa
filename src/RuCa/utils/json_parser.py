import json
from pathlib import Path
import os

# Юзер вставляет свой путь к папке с JSON файлами
QUERIES_FOLDER = os.path.join(os.path.dirname(__file__), "..", "querries")

#читаем все json файлы из папки
def read_json_files(folder_path):
    all_queries = []
    json_files = list(Path(folder_path).glob("*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for key, value in data.items():
                if isinstance(value, list):
                    all_queries.extend(value)

    return all_queries

#проверяем, что все нужные поля есть
def validate_item(item):
    if "id" not in item:
        return False
    if "query" not in item or not item["query"].strip():
        return False
    if "expected_tool" not in item:
        return False
    if "expected_parameters" not in item:
        return False
    if "requires_clarification" not in item:
        return False
    if "skills" not in item:
        return False
    return True

#приводим данные к единому виду
def normalize_item(item):

    expected_params = item.get("expected_parameters") or {}
    normalized_params = {}
    for key, value in expected_params.items():
        if isinstance(value, str):
            normalized_params[key] = value.lower().strip()
        else:
            normalized_params[key] = value
            
    normalized = {
        "id": str(item.get("id", "")).strip(),
        "query": item.get("query", "").strip(),
        "expected_tool": (item.get("expected_tool") or "").strip().lower(),
        "expected_parameters": normalized_params,
        "requires_clarification": item.get("requires_clarification", False),
        "skills": item.get("skills", [])
    }

    normalized["query"] = " ".join(normalized["query"].split())
    
    return normalized

#готовим список из всех запросов (для подачи и для оценки по метрикам)
def prepare_for_llm(items, system_prompt):
    inputs_for_llm = []
    inputs_for_logging = []

    for item in items:
        input_for_llm = {
            "id": item["id"],
            "system_prompt": system_prompt,
            "user_query": item["query"]
        }

        input_log = {
            "id": item["id"],
            "user_query": item["query"],
            
            "expected_tool": item["expected_tool"],
            "expected_parameters": item["expected_parameters"],
            "requires_clarification": item["requires_clarification"],
            "skills": item["skills"]
        }
        
        inputs_for_llm.append(input_for_llm)
        inputs_for_logging.append(input_log)
    
    
    return inputs_for_llm, inputs_for_logging

#выполняем все функции
def process_all_queries(folder_path=QUERIES_FOLDER, system_prompt=""):
    all_queries = read_json_files(folder_path)
    
    valid_queries = [item for item in all_queries if validate_item(item)]
    
    normalized_queries = [normalize_item(item) for item in valid_queries]
    
    inputs_for_llm, inputs_for_logging = prepare_for_llm(normalized_queries, system_prompt)
    
    return inputs_for_llm, inputs_for_logging

system_prompt = """
Ты — агент, который ДОЛЖЕН строго возвращать JSON-объект.
Никакого текста вне JSON.

Всегда возвращай JSON строго следующей структуры:

{
    "id": "<ID запроса>",
    "tool_call": {
        "tool_name": "...",
        "parameters": { ... },
        "called": true/false
    },
    "clarification_question": null или строка,
    "user_message": null или строка,
    "internal": {
        "reasoning": "<ОДНО короткое предложение>",
        "errors": null
    }
}

Правила:
1. Если можешь вызвать инструмент — вызывай. "tool_call.called": true.
2. Если данных мало — НЕ вызывай инструмент, а задай вопрос (clarification_question).
3. НЕ заворачивай JSON в строки. НЕ используй markdown. НЕ пиши текст вокруг.
4. "internal.reasoning" должен быть ОДНОЙ КОРОТКОЙ ФРАЗОЙ (до 15 слов).
5. "assistant_response": должен быть не более нескольких КОРОТКИХ ФРАЗ (2-3 предложения до 15 слов).
5. Параметры инструмента должны быть только те, что есть в его спецификации.
6. Если вызываешь инструмент — "user_message": null.
7. Если НЕ вызываешь — "tool_call": null.
8. Ты можешь вызывать инструменты ПОСЛЕДОВАТЕЛЬНО.
    После получения результата инструмента ты можешь вызвать следующий инструмент,
    если это необходимо для ответа пользователю.
    Ты обязан следовать этим правилам и НИКОГДА не выводить JSON как текст.

"""
inputs_for_llm, inputs_for_logging = process_all_queries(
        system_prompt=system_prompt
    ) 