from typing import Any
import pandas as pd
import json
from json_parser import inputs_for_logging
from output_parser import outputs_for_logging

#проверяем совместимы ли id запроса и ответа от модели
def validate_ids(inputs_for_logging: list[dict[str, Any]], outputs_for_logging: list[dict[str, Any]]) -> bool:
    """
    Проверяет, что ID в inputs и outputs совпадают и идут в том же порядке.
    Raises ValueError если есть несовпадения.
    """
    if len(inputs_for_logging) != len(outputs_for_logging):
        raise ValueError(
            f"Количество inputs ({len(inputs_for_logging)}) != outputs ({len(outputs_for_logging)})"
        )

    mismatches = []
    for i, (inp, out) in enumerate(zip(inputs_for_logging, outputs_for_logging)):
        inp_id = inp.get("id")
        out_id = out.get("id")

        if inp_id != out_id:
            mismatches.append(
                f"Позиция {i}: input_id='{inp_id}' != output_id='{out_id}'"
            )

    if mismatches:
        error_msg = "Несовпадение ID:\n" + "\n".join(mismatches)
        raise ValueError(error_msg)

    return True


# ---------- вспомогательная функция: нужна ли метрика ----------
def _is_metric_enabled(inp: dict[str, Any], metric_name: str) -> bool:
    """
    Возвращает True, если в поле skills есть metric_name.
    Если skills отсутствует – считаем, что метрика НЕ нужна.
    """
    skills = inp.get("skills", [])
    return metric_name in skills


# ---------- метрики ----------

def decision(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """Да/нет: ожидался вызов и он был / не ожидался и не был."""
    if not _is_metric_enabled(inp, "Decision"):
        return None

    expected = bool(inp["expected_tool"])
    actual_tools = out.get("name", "")
    has_tool = bool(actual_tools and actual_tools.strip())
    return float(expected == has_tool)


def tool_selection_f1(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """F1 по множеству имён вызванных тулзов (без учёта порядка)."""
    if not _is_metric_enabled(inp, "Tool selection"):
        return None

    ref_tools = set(t.strip() for t in inp["expected_tool"].split(","))
    actual_tools = set(out.get("name", "").split(","))
    actual_tools.discard("")

    if not ref_tools and not actual_tools:
        return 1.0

    true_positive = len(actual_tools & ref_tools)
    false_positive = len(actual_tools - ref_tools)
    false_negative = len(ref_tools - actual_tools)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0

    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def params_recall(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """Доля правильных параметров (None-поля пропускаются)."""
    if not _is_metric_enabled(inp, "Params"):
        return None

    ref_params = inp["expected_parameters"]
    args_str = out.get("arguments", "{}")
    try:
        actual_params = json.loads(args_str)
    except Exception:
        actual_params = {}

    if not ref_params:
        return 1.0

    total_count = 0
    correct_count = 0

    for key, value in ref_params.items():
        if value is None:
            continue
        total_count += 1
        actual_value = actual_params.get(key)
        correct_count += int(actual_value == value)

    return correct_count / total_count if total_count else 1.0


def result(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """Сводная: правильность tool + правильность параметров."""
    if not _is_metric_enabled(inp, "Result"):
        return None

    tool_correct = tool_selection_f1(inp, out)
    params_correct = params_recall(inp, out)

    # если одна из метрик отключена – не считаем сводную
    if tool_correct is None or params_correct is None:
        return None

    return (tool_correct + params_correct) / 2


def execution(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """Строгий порядок вызовов (chain)."""
    if not _is_metric_enabled(inp, "Execution"):
        return None

    ref_tools = [t.strip() for t in inp["expected_tool"].split(",")]
    actual_tools = out.get("name", "").split(",")
    actual_tools = [t for t in actual_tools if t]
    return float(actual_tools == ref_tools)


def noise(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    1 = вызваны ровно те тулзы и ровно те параметры, что в expected
    0 = появился лишний тулз или лишний параметр
    """
    if not _is_metric_enabled(inp, "Noise"):
        return None

    ref_tools = set(t.strip() for t in inp["expected_tool"].split(","))
    actual_tools = set(out.get("name", "").split(","))
    actual_tools.discard("")

    if actual_tools != ref_tools:
        return 0.0

    ref_params_keys = set(inp["expected_parameters"].keys())
    args_str = out.get("arguments", "{}")
    try:
        actual_params_keys = set(json.loads(args_str).keys())
    except Exception:
        actual_params_keys = set()

    if actual_params_keys != ref_params_keys:
        return 0.0

    return 1.0


def adaptability(inp: dict[str, Any], last_out: dict[str, Any]) -> float | None:
    """
    1 = последний вызов полностью совпал с expected (имена и параметры)
    0 = любое несовпадение
    """
    if not _is_metric_enabled(inp, "Adaptability"):
        return None

    ref_tools = set(t.strip() for t in inp["expected_tool"].split(","))
    last_tools = set(last_out.get("name", "").split(","))
    last_tools.discard("")

    if last_tools != ref_tools:
        return 0.0

    ref_params = {k: v for k, v in inp["expected_parameters"].items() if v is not None}
    args_str = last_out.get("arguments", "{}")
    try:
        last_params = json.loads(args_str)
    except Exception:
        last_params = {}

    if last_params != ref_params:
        return 0.0

    return 1.0


def error_handling(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Если агент НЕ вызвал ни одного тулза – 1.0 балл,
    иначе – 0.0 (метрика про «не вызвал»).
    """
    if not _is_metric_enabled(inp, "Error Handling"):
        return None

    return float(not out.get("name", "").strip())


def ambiguity(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Сценарий 1: requires_clarification = True
        1 = не вызвал тулз И не передал параметры (чисто уточнил)
        0 = вызвал хоть что-то
    Сценарий 2: requires_clarification = False
        1 = точно совпало (tool + params)
        0.5 = только tool или только params
        0 = ни tool, ни params не совпали
    """
    if not _is_metric_enabled(inp, "Ambiguity"):
        return None

    ref_tools = set(t.strip() for t in inp["expected_tool"].split(","))
    actual_tools = set(out.get("name", "").split(","))
    actual_tools.discard("")

    ref_params = {k: v for k, v in inp["expected_parameters"].items() if v is not None}
    args_str = out.get("arguments", "{}")
    try:
        actual_params = json.loads(args_str)
    except Exception:
        actual_params = {}

    if inp["requires_clarification"]:
        return float(not actual_tools and not actual_params)

    tool_correct = actual_tools == ref_tools
    param_correct = actual_params == ref_params

    if tool_correct and param_correct:
        return 1.0
    if tool_correct or param_correct:
        return 0.5
    return 0.0



# ---------- единая обёртка ----------
def evaluate_batch(inputs_for_logging: list[dict[str, Any]],
                outputs_for_logging: list[dict[str, Any]]) -> pd.DataFrame:
    """Принимает два списка одинаковой длины и возвращает DataFrame со скорами."""
    records = []
    for inp, out in zip(inputs_for_logging, outputs_for_logging):
        records.append({
            "id": inp["id"],
            "decision": decision(inp, out),
            "tool_selection": tool_selection_f1(inp, out),
            "params": params_recall(inp, out),
            "error_handling": error_handling(inp, out),
            "result": result(inp, out),
            "execution": execution(inp, out),
            "noise": noise(inp, out),
            "adaptability": adaptability(inp, out),
            "ambiguity": ambiguity(inp, out),
        })
    return pd.DataFrame(records)


#финальная оценка
def calculate_final_score(df: pd.DataFrame) -> float:
    """
    Для каждого запроса определяем, сколько у него метрик,
    и считаем скор по соответствующей схеме, затем усредняем.
    """

    weights_4 = {
        "decision": 0.30,
        "tool_selection": 0.30,
        "params": 0.22,
        "result": 0.18
    }

    weights_5_base = {
        "decision": 0.28,
        "tool_selection": 0.28,
        "params": 0.20,
        "result": 0.04
    }

    specific_metrics = ["Ambiguity", "Noise", "Adaptability", "Error Handling", "Execution"]
    specific_weight = 0.20

    scores = []

    for idx, row in df.iterrows():

        specific_found = None
        for metric in specific_metrics:
            if metric in row.index and pd.notna(row[metric]):
                specific_found = metric
                break

        if specific_found:

            score = 0.0
            for metric, weight in weights_5_base.items():
                if metric in row.index and pd.notna(row[metric]):
                    score += weight * row[metric]

            score += specific_weight * row[specific_found]
        else:
            score = 0.0
            for metric, weight in weights_4.items():
                if metric in row.index and pd.notna(row[metric]):
                    score += weight * row[metric]

        scores.append(score)

    if scores:
        final_score = sum(scores) / len(scores)
        return final_score * 100
    else:
        return 0.0

if __name__ == "__main__":
    # Проверяем совместимость ID
    validate_ids(inputs_for_logging, outputs_for_logging)

    # Вычисляем метрики для каждого примера
    df = evaluate_batch(inputs_for_logging, outputs_for_logging)

    # # Среднее по каждой метрике
    # print("Среднее по каждой метрике:")
    # print(df.drop(columns=['id']).mean().round(2))

    # Финальный скор модели
    final_score = calculate_final_score(df)
    print(f"\nФИНАЛЬНЫЙ СКОР МОДЕЛИ: {final_score:.2f}%")