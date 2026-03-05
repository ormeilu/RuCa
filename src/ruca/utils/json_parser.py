"""Query parser for loading and processing benchmark queries from JSON files.

This module handles:
- Reading structured queries from JSON files
- Validating query structure and required fields
- Normalizing query data for consistent processing
- Preparing queries for LLM input and metric evaluation
"""

import json
import os
from pathlib import Path
from typing import Any

# Queries folder location - relative to this module's parent directory
QUERIES_FOLDER: str = os.path.join(os.path.dirname(__file__), "..", "querries")

# читаем все json файлы из папки
def read_json_files(folder_path: str) -> list[dict[str, Any]]:
    """Read and parse all JSON files from a folder.
    
    Loads all .json files from the specified folder and flattens lists
    found in the data into a single query list.
    
    Args:
        folder_path: Path to the folder containing JSON files
        
    Returns:
        List of all query objects found in the JSON files
    """
    all_queries: list[dict[str, Any]] = []
    json_files: list[Path] = list(Path(folder_path).glob("*.json"))

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data: dict[str, Any] | list[Any] | None = json.load(f)

            if isinstance(data, dict):
                for _key, value in data.items():
                    if isinstance(value, list):
                        all_queries.extend(value)
            elif isinstance(data, list):
                all_queries.extend(data)

    return all_queries


def validate_item(item: dict[str, Any]) -> bool:
    """Validate that a query item has all required fields.
    
    Required fields:
    - id: Unique identifier
    - query: User's query text (non-empty)
    - expected_tool: Tool/function name expected to be called
    - expected_parameters: Parameters for the tool
    - requires_clarification: Boolean indicating if clarification is needed
    - skills: List of applicable metrics/skills
    
    Args:
        item: Query item to validate
        
    Returns:
        True if item contains all required fields, False otherwise
    """
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
    return "skills" in item


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize and standardize a query item for processing.
    
    Performs the following normalizations:
    - Convert IDs to strings and strip whitespace
    - Lowercase and strip query text
    - Lowercase and strip tool names
    - Normalize parameter keys (lowercase) and string values (lowercase/stripped)
    - Collapse multiple whitespaces in query text
    
    Args:
        item: Raw query item from JSON
        
    Returns:
        Normalized query item with standardized formatting
    """
    expected_params: dict[str, Any] | None = item.get("expected_parameters") or {}
    normalized_params: dict[str, Any] = {}
    for key, value in expected_params.items():
        if isinstance(value, str):
            normalized_params[key] = value.lower().strip()
        else:
            normalized_params[key] = value

    normalized: dict[str, Any] = {
        "id": str(item.get("id", "")).strip(),
        "query": item.get("query", "").strip(),
        "expected_tool": (item.get("expected_tool") or "").strip().lower(),
        "expected_parameters": normalized_params,
        "requires_clarification": item.get("requires_clarification", False),
        "skills": item.get("skills", []),
    }

    # Collapse multiple whitespaces in query
    normalized["query"] = " ".join(normalized["query"].split())

    return normalized


def prepare_for_llm(
    items: list[dict[str, Any]],
    system_prompt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare query items for LLM input and metric evaluation.
    
    Separates data into two formats:
    - inputs_for_llm: Contains system prompt and user query for LLM processing
    - inputs_for_logging: Contains all data needed for metric evaluation
    
    Args:
        items: List of normalized query items
        system_prompt: System prompt to include in LLM input
        
    Returns:
        Tuple of (inputs_for_llm, inputs_for_logging)
    """
    inputs_for_llm: list[dict[str, Any]] = []
    inputs_for_logging: list[dict[str, Any]] = []

    for item in items:
        input_for_llm: dict[str, Any] = {
            "id": item["id"],
            "system_prompt": system_prompt,
            "user_query": item["query"]
        }

        input_log: dict[str, Any] = {
            "id": item["id"],
            "user_query": item["query"],
            "expected_tool": item["expected_tool"],
            "expected_parameters": item["expected_parameters"],
            "requires_clarification": item["requires_clarification"],
            "skills": item["skills"],
        }

        inputs_for_llm.append(input_for_llm)
        inputs_for_logging.append(input_log)

    return inputs_for_llm, inputs_for_logging


def process_all_queries(
    folder_path: str = QUERIES_FOLDER,
    system_prompt: str = ""
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process all benchmark queries from JSON files.
    
    Performs complete pipeline:
    1. Read all JSON files from folder
    2. Validate query structure
    3. Normalize query data
    4. Prepare for LLM and metric evaluation
    
    Args:
        folder_path: Path to folder containing query JSON files
        system_prompt: System prompt to include for LLM processing
        
    Returns:
        Tuple of (inputs_for_llm, inputs_for_logging)
    """
    all_queries: list[dict[str, Any]] = read_json_files(folder_path)

    valid_queries: list[dict[str, Any]] = [item for item in all_queries if validate_item(item)]

    normalized_queries: list[dict[str, Any]] = [normalize_item(item) for item in valid_queries]

    inputs_for_llm, inputs_for_logging = prepare_for_llm(normalized_queries, system_prompt)

    return inputs_for_llm, inputs_for_logging


system_prompt: str = """
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


def get_inputs_for_logging() -> list[dict[str, Any]]:
    """lazily load and return input data for metric evaluation.
    
    This function is called on demand to avoid loading data during import.
    
    Returns:
        List of input examples with expected values for metric evaluation
    """
    _, inputs = process_all_queries(system_prompt=system_prompt)
    return inputs


def get_inputs_for_llm() -> list[dict[str, Any]]:
    """Lazily load and return input data for LLM processing.
    
    This function is called on demand to avoid loading data during import.
    
    Returns:
        List of input examples formatted for LLM processing
    """
    inputs, _ = process_all_queries(system_prompt=system_prompt)
    return inputs
