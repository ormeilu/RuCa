import json
from pathlib import Path
import os

# Юзер вставляет свой путь к папке с JSON файлами
# QUERIES_FOLDER = os.path.join(os.path.dirname(__file__), "queries")

QUERIES_FOLDER = "queries"
#"/home/alena-kuriatnikova/ML модуль/benchmark/queries"

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
    normalized = {
        "id": str(item.get("id", "")).strip(),
        "query": item.get("query", "").strip(),
        "expected_tool": (item.get("expected_tool") or "").strip().lower(),
        "expected_parameters": item.get("expected_parameters", {}),
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
  "chain_complete": true/false,
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
5. "user_message": должен быть не более нескольких КОРОТКИХ ФРАЗ (2-3 предложения до 15 слов).
6. Параметры инструмента должны быть только те, что есть в его спецификации.
7. Если вызываешь инструмент — "user_message": null.
8. Если НЕ вызываешь — "tool_call": null.

ПОДДЕРЖКА ЦЕПОЧЕК ИНСТРУМЕНТОВ:
9. Если задача требует НЕСКОЛЬКИХ инструментов последовательно:
   - Вызови ПЕРВЫЙ инструмент
   - Установи "chain_complete": false
   - После получения результата ты получишь новый запрос с результатом
   - Вызови СЛЕДУЮЩИЙ инструмент с учетом предыдущего результата
   - Когда все инструменты выполнены — установи "chain_complete": true

Примеры цепочек:
- "Найди товар и переведи цену в USD" → search_products, потом currency_converter
- "Узнай погоду и переведи на английский" → get_weather, потом translate
- "Вычисли сумму и переведи в евро" → calculator, потом currency_converter

10. Для цепочек: используй результат предыдущего инструмента в параметрах следующего.
11. Если это последний/единственный инструмент в цепочке — "chain_complete": true.


"""
if __name__ == "__main__":
    
    
    
    inputs_for_llm, inputs_for_logging = process_all_queries(
        system_prompt=system_prompt
    ) 
