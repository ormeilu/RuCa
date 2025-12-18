from typing import Any
import pandas as pd
import json
import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich import box
from json_parser import inputs_for_logging
from output_parser import outputs_for_logging
import argparse
console = Console()

# Хранилище для детальных ошибок
detailed_errors = []

def log_error(query_id: str, metric_name: str, reason: str, expected: Any = None, actual: Any = None):
    """Логирует детальную информацию об ошибке."""
    error_entry = {
        "id": query_id,
        "metric": metric_name,
        "reason": reason,
        "expected": expected,
        "actual": actual
    }
    detailed_errors.append(error_entry)


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
    
    result = float(expected == has_tool)
    
    if result == 0.0:
        if expected and not has_tool:
            log_error(
                inp["id"], 
                "Decision",
                "Ожидался вызов инструмента, но инструмент не был вызван",
                expected=f"Должен был вызвать: {inp['expected_tool']}",
                actual="Инструмент не вызван"
            )
        elif not expected and has_tool:
            log_error(
                inp["id"],
                "Decision", 
                "Не ожидался вызов инструмента, но инструмент был вызван",
                expected="Не должен был вызывать инструменты",
                actual=f"Вызван: {actual_tools}"
            )
    
    return result


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

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    if f1_score < 1.0:
        errors = []
        if false_positive > 0:
            extra = actual_tools - ref_tools
            errors.append(f"Лишние инструменты: {', '.join(extra)}")
        if false_negative > 0:
            missing = ref_tools - actual_tools
            errors.append(f"Не хватает инструментов: {', '.join(missing)}")
        
        log_error(
            inp["id"],
            "Tool Selection",
            " | ".join(errors),
            expected=f"Должны быть вызваны: {', '.join(ref_tools)}",
            actual=f"Вызваны: {', '.join(actual_tools) if actual_tools else 'Нет'}"
        )
    
    return f1_score


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
    wrong_params = []

    for key, value in ref_params.items():
        if value is None:
            continue
        total_count += 1
        actual_value = actual_params.get(key)
        
        if actual_value == value:
            correct_count += 1
        else:
            wrong_params.append(f"{key}: ожидалось '{value}', получено '{actual_value}'")

    recall = correct_count / total_count if total_count else 1.0
    
    if recall < 1.0 and wrong_params:
        log_error(
            inp["id"],
            "Params",
            f"Неправильные параметры: {' | '.join(wrong_params)}",
            expected=ref_params,
            actual=actual_params
        )
    
    return recall


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
    
    is_correct = actual_tools == ref_tools
    
    if not is_correct:
        log_error(
            inp["id"],
            "Execution",
            "Неправильный порядок вызова инструментов",
            expected=f"Порядок: {' -> '.join(ref_tools)}",
            actual=f"Порядок: {' -> '.join(actual_tools) if actual_tools else 'Нет'}"
        )
    
    return float(is_correct)


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
        extra = actual_tools - ref_tools
        missing = ref_tools - actual_tools
        errors = []
        if extra:
            errors.append(f"Лишние инструменты: {', '.join(extra)}")
        if missing:
            errors.append(f"Недостающие инструменты: {', '.join(missing)}")
        
        log_error(
            inp["id"],
            "Noise",
            " | ".join(errors),
            expected=', '.join(ref_tools),
            actual=', '.join(actual_tools) if actual_tools else 'Нет'
        )
        return 0.0

    ref_params_keys = set(inp["expected_parameters"].keys())
    args_str = out.get("arguments", "{}")
    try:
        actual_params_keys = set(json.loads(args_str).keys())
    except Exception:
        actual_params_keys = set()

    if actual_params_keys != ref_params_keys:
        extra = actual_params_keys - ref_params_keys
        missing = ref_params_keys - actual_params_keys
        errors = []
        if extra:
            errors.append(f"Лишние параметры: {', '.join(extra)}")
        if missing:
            errors.append(f"Недостающие параметры: {', '.join(missing)}")
        
        log_error(
            inp["id"],
            "Noise",
            " | ".join(errors),
            expected=', '.join(ref_params_keys),
            actual=', '.join(actual_params_keys) if actual_params_keys else 'Нет'
        )
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
        log_error(
            inp["id"],
            "Adaptability",
            "Инструменты в последнем вызове не совпадают",
            expected=', '.join(ref_tools),
            actual=', '.join(last_tools) if last_tools else 'Нет'
        )
        return 0.0

    ref_params = {k: v for k, v in inp["expected_parameters"].items() if v is not None}
    args_str = last_out.get("arguments", "{}")
    try:
        last_params = json.loads(args_str)
    except Exception:
        last_params = {}

    if last_params != ref_params:
        log_error(
            inp["id"],
            "Adaptability",
            "Параметры в последнем вызове не совпадают",
            expected=ref_params,
            actual=last_params
        )
        return 0.0

    return 1.0


def error_handling(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Если агент НЕ вызвал ни одного тулза – 1.0 балл,
    иначе – 0.0 (метрика про «не вызвал»).
    """
    if not _is_metric_enabled(inp, "Error Handling"):
        return None

    has_tool = bool(out.get("name", "").strip())
    
    if has_tool:
        log_error(
            inp["id"],
            "Error Handling",
            "Инструмент был вызван, хотя не должен был (ошибка обработки)",
            expected="Не должен вызывать инструменты",
            actual=f"Вызван: {out.get('name', '')}"
        )
    
    return float(not has_tool)


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
        result = float(not actual_tools and not actual_params)
        if result == 0.0:
            log_error(
                inp["id"],
                "Ambiguity",
                "Требовалось уточнение, но были вызваны инструменты или переданы параметры",
                expected="Не должен вызывать инструменты, нужно запросить уточнение",
                actual=f"Инструменты: {', '.join(actual_tools) if actual_tools else 'Нет'}, Параметры: {actual_params}"
            )
        return result

    tool_correct = actual_tools == ref_tools
    param_correct = actual_params == ref_params

    if tool_correct and param_correct:
        return 1.0
    elif tool_correct or param_correct:
        errors = []
        if not tool_correct:
            errors.append(f"Инструменты не совпадают (ожидалось: {', '.join(ref_tools)}, получено: {', '.join(actual_tools) if actual_tools else 'Нет'})")
        if not param_correct:
            errors.append(f"Параметры не совпадают")
        
        log_error(
            inp["id"],
            "Ambiguity",
            " | ".join(errors),
            expected={"tools": list(ref_tools), "params": ref_params},
            actual={"tools": list(actual_tools), "params": actual_params}
        )
        return 0.5
    else:
        log_error(
            inp["id"],
            "Ambiguity",
            "Не совпадают ни инструменты, ни параметры",
            expected={"tools": list(ref_tools), "params": ref_params},
            actual={"tools": list(actual_tools), "params": actual_params}
        )
        return 0.0



# ---------- единая обёртка ----------
def evaluate_batch(inputs_for_logging: list[dict[str, Any]],
                outputs_for_logging: list[dict[str, Any]]) -> pd.DataFrame:
    """Принимает два списка одинаковой длины и возвращает DataFrame со скорами."""
    global detailed_errors
    detailed_errors = []  # Очищаем перед новым запуском
    
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


def print_detailed_errors(threshold: float = 0.8):
    """Выводит детальную информацию об ошибках для запросов с низким скором."""
    if not detailed_errors:
        console.print("[green]✓ Ошибок не обнаружено![/green]\n")
        return
    
    console.print(Panel.fit(
        f"[bold red]ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК ({len(detailed_errors)} проблем)[/bold red]",
        border_style="red"
    ))
    console.print()
    
    # Группируем ошибки по ID
    errors_by_id = {}
    for error in detailed_errors:
        query_id = error["id"]
        if query_id not in errors_by_id:
            errors_by_id[query_id] = []
        errors_by_id[query_id].append(error)
    
    # Выводим ошибки для каждого запроса
    for query_id, errors in errors_by_id.items():
        error_table = Table(
            title=f"[bold yellow]ID запроса: {query_id}[/bold yellow]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold red"
        )
        
        error_table.add_column("Метрика", style="cyan", width=20)
        error_table.add_column("Причина ошибки", style="white", width=50)
        error_table.add_column("Ожидалось", style="green", width=30)
        error_table.add_column("Получено", style="red", width=30)
        
        for err in errors:
            expected_str = str(err["expected"])[:50] if err["expected"] else "N/A"
            actual_str = str(err["actual"])[:50] if err["actual"] else "N/A"
            
            error_table.add_row(
                err["metric"],
                err["reason"],
                expected_str,
                actual_str
            )
        
        console.print(error_table)
        console.print()


def print_benchmark_results(df: pd.DataFrame, inputs_for_logging: list[dict[str, Any]], benchmark_file: str = "benchmark_results.json") -> None:
    """Выводит таблицу с результатами бенчмарка."""
    
    metric_cols = [
        "decision", "tool_selection", "params", "result",
        "ambiguity", "noise", "adaptability", "error_handling", "execution"
    ]
    total_queries = len(df)
    
    # Считаем успешные запросы
    def is_successful(row):
        valid = row[metric_cols].dropna()
        if len(valid) == 0:
            return False
        return all(valid == 1.0)
    
    successful = df.apply(is_successful, axis=1).sum()
    failed = total_queries - successful
    success_rate = (successful / total_queries * 100) if total_queries > 0 else 0
    failed_rate = 100 - success_rate
    
    # Финальный скор
    final = calculate_final_score(df)
    
    # Средние по метрикам
    means = df[metric_cols].mean(skipna=True)

    # Формируем отображение
    metrics_display = {}
    for m in metric_cols:
        if m in means.index:
            metrics_display[m] = (means[m])
        else:
            metrics_display[m] = (float("nan"), 0)

    # ============ ВЫВОД ============
    console.print()
    
    # Заголовок
    console.print(Panel.fit(
        "[bold cyan]BENCHMARK RESULTS[/bold cyan]",
        border_style="cyan"
    ))
    
    # Основная статистика
    stats_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    stats_table.add_column(style="bold white", justify="left")
    stats_table.add_column(style="cyan", justify="right")
    
    stats_table.add_row("Всего запросов", f"{total_queries}")
    stats_table.add_row(
        "Успешно выполненные", 
        f"[green]{successful}[/green] ([green]{success_rate:.1f}%[/green])"
    )
    stats_table.add_row(
        "С ошибками", 
        f"[red]{failed}[/red] ([red]{failed_rate:.1f}%[/red])"
    )
    
    console.print(stats_table)
    console.print()
    
    # Финальный скор с цветом
    if final >= 80:
        score_color = "green"
    elif final >= 60:
        score_color = "yellow"
    else:
        score_color = "red"
    
    console.print(Panel(
        f"[bold {score_color}]{final:.1f}%[/bold {score_color}]",
        title="[bold white]ФИНАЛЬНАЯ ОЦЕНКА МОДЕЛИ[/bold white]",
        border_style=score_color,
        padding=(0, 4)
    ))
    console.print()
    
    # Статистика по метрикам
    metrics_table = Table(
        title="[bold white]МЕТРИКИ[/bold white]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    metrics_table.add_column("Метрика", style="white", justify="left", width=25)
    metrics_table.add_column("Значение", justify="center", width=10)
    metrics_table.add_column("Визуализация", justify="left", width=25)

    for metric_name, metric_value in metrics_display.items():

        if math.isnan(metric_value):
            value_str = "[white]N/A[/white]"
            bar = "░" * 20
            metrics_table.add_row(metric_name, value_str, bar)
            continue

        if metric_value >= 0.8:
            value_style = "green"
        elif metric_value >= 0.6:
            value_style = "yellow"
        else:
            value_style = "red"

        bar_length = int(metric_value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        metrics_table.add_row(
            metric_name,
            f"[{value_style}]{metric_value:.2f}[/{value_style}]",
            f"[{value_style}]{bar}[/{value_style}]"
        )
    
    console.print(metrics_table)
    console.print()
    
    # Выводим детальные ошибки
    # print_detailed_errors()

    try:
        output = {
            "config": {},
            "total_queries": total_queries,
            "successful": int(successful),
            "failed": int(failed),
            "success_rate": round(success_rate, 2),
            "final_score": round(final, 2),
            "metrics_mean": {m: (None if math.isnan(metrics_display[m]) else round(metrics_display[m], 4)) for m in metric_cols},
        }
        
       
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f: 
                bench_data = json.load(f)
                print(f"DEBUG evaluation.py: Found {benchmark_file}") 
                if "config" in bench_data:
                    output["config"] = bench_data["config"]
                    print(f"DEBUG: Config saved to evaluate_results.json: temperature={output['config'].get('temperature')}, top_p={output['config'].get('top_p')}, top_k={output['config'].get('top_k')}, seed={output['config'].get('seed')}")
                else:
                    print(f"DEBUG: No 'config' key in {benchmark_file}")  
        except Exception as e:
            print(f"DEBUG: Exception reading {benchmark_file}: {e}")  
        
        with open("evaluate_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        console.print("[green]Результаты сохранены в evaluate_results.json[/green]")
    except Exception as e:
        console.print(f"[red]Не удалось сохранить результаты в json: {e}[/red]")


def save_detailed_errors_json(
    filepath: str = "detailed_errors.json"
):
    """Сохраняет детальные ошибки в JSON-файл."""
    if not detailed_errors:
        return

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_results.json",
        help="Path to benchmark results JSON file"
    )
    
    args = parser.parse_args()
    # Проверяем совместимость ID
    validate_ids(inputs_for_logging, outputs_for_logging)

    # Вычисляем метрики для каждого примера
    df = evaluate_batch(inputs_for_logging, outputs_for_logging)

    # Вывод результатов
    print_benchmark_results(df, inputs_for_logging, benchmark_file=args.input)
    # Детальные ошибки в виде жисона
    # save_detailed_errors_json("detailed_errors.json")

