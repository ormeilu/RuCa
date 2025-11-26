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
        "expected_tool": item.get("expected_tool", "").strip().lower(),
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
Ты — ассистент, прогоняющий запросы из tool-calling бенчмарка. Твоя задача: прочитать user query и выполнить один из трех пунктов:
1) вызвать инструмент (tool) с корректными параметрами;
2) если данных недостаточно — не вызывать инструмент и задать вопрос уточнения;
3) если действие не предполагает tool, вернуть called=false.

Правила:
- Используй список доступных инструментов (tools manifest) — если он указан в input, выбери один из них.
- Никогда не фантазируй значения параметров: если ожидается order_id, а его нет в запросе — возвращай clarification_question.
- Если запрос явный (пользователь называет имя тулза) — проверь, что тулз существует в manifest. Если нет — верни called=false и message об отсутствии тулза.
- При шуме (личные данные, внешние ссылки, рекламный текст) — игнорируй эти данные при формировании аргументов тулза. Не включай PII в arguments; если в аргументах должен быть PII — замаскируй.
- Всегда заполняй поле internal.reasoning: короткая фраза почему выбран тул/почему нужна уточняющая фраза.
- Форматируй ответ строго в JSON (см. RETURN FORMAT). Любой другой вывод будет считаться ошибкой.

RETURN FORMAT:
{
    "id": "000000",                                                     #id запроса, который был передан в input
    "tool_call": {
        "tool_name": "cancel_order",                                    #название tool, который был вызван моделью
        "parameters": {"order_id": "ORD1023"},                          #параметры, которые модель передала в tool
        "called": true                                                  #был ли tool вызван
    },
    "clarification_question": null,                                     #был ли задан вопрос уточнения
    "user_message": "Отменяю заказ ORD1023 на наушники.",               #модель отвечает на вопрос пользователя
    "internal": {
        "reasoning": "Явный запрос cancel_order + корректный order_id", #какой модель выбрала tool для вызова и какие параметры забрала
        "errors": null                                                  #была ли у модели ошибка
    }
}
"""
if __name__ == "__main__":
    
    
    
    inputs_for_llm, inputs_for_logging = process_all_queries(
        system_prompt=system_prompt
    ) 
