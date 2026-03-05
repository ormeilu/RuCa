# Standard library imports
import argparse
import json
import os
from typing import Any

# Third-party imports
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Process command-line arguments to set environment variable BEFORE importing output_parser
# (output_parser depends on this environment variable)
parser_temp: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
parser_temp.add_argument("--input", type=str, default="benchmark_results.json")
args_temp, _ = parser_temp.parse_known_args()
os.environ["BENCHMARK_FILE"] = args_temp.input

# Local imports - order matters due to environment variable dependency
from ruca.utils.json_parser import get_inputs_for_logging
from ruca.utils.metrics_enum import Metrics
from ruca.utils.output_parser import get_outputs_for_logging
from ruca.utils.final_score import calculate_final_score

console: Console = Console()

# Global storage for detailed error information
# Each entry contains: id, metric_name, reason, expected_value, actual_value
detailed_errors: list[dict[str, Any]] = []


def log_error(
    query_id: str,
    metric_name: str,
    reason: str,
    expected: Any = None,
    actual: Any = None
) -> None:
    """
    Log detailed error information for a failed metric evaluation.
    
    Args:
        query_id: Unique identifier for the query/request
        metric_name: Name of the metric that failed (e.g., 'Decision', 'Tool Selection')
        reason: Human-readable reason for the failure
        expected: Expected value according to the benchmark
        actual: Actual value produced by the model
    """
    error_entry: dict[str, Any] = {
        "id": query_id,
        "metric": metric_name,
        "reason": reason,
        "expected": expected,
        "actual": actual
    }
    detailed_errors.append(error_entry)


def validate_ids(
    inputs_for_logging: list[dict[str, Any]],
    outputs_for_logging: list[dict[str, Any]]
) -> bool:
    """
    Validate that query IDs in inputs and outputs match and are in the same order.
    
    Args:
        inputs_for_logging: List of input benchmark examples with 'id' field
        outputs_for_logging: List of model outputs with 'id' field
        
    Returns:
        True if all IDs match and are in order
        
    Raises:
        ValueError: If the number of inputs/outputs differ or IDs are misaligned
    """
    if len(inputs_for_logging) != len(outputs_for_logging):
        raise ValueError(
            f"Number of inputs ({len(inputs_for_logging)}) != outputs ({len(outputs_for_logging)})"
        )

    mismatches: list[str] = []
    for i, (inp, out) in enumerate(zip(inputs_for_logging, outputs_for_logging, strict=True)):
        inp_id: Any = inp.get("id")
        out_id: Any = out.get("id")

        if inp_id != out_id:
            mismatches.append(f"Position {i}: input_id='{inp_id}' != output_id='{out_id}'")

    if mismatches:
        error_msg: str = "ID Mismatch:\n" + "\n".join(mismatches)
        raise ValueError(error_msg)

    return True


def _is_metric_enabled(inp: dict[str, Any], metric: Metrics) -> bool:
    """
    Check if a specific metric is enabled for this query.
    
    A metric is enabled if it appears in the query's 'skills' list.
    Comparison is case-insensitive and ignores spaces/underscores.
    
    Args:
        inp: Input benchmark example containing 'skills' list
        metric: Metrics enum value to check
        
    Returns:
        True if metric is in the skills list, False otherwise
    """
    skills: list[str] = inp.get("skills", [])

    def normalize(s: str) -> str:
        """Normalize string for comparison: lowercase, remove spaces and underscores."""
        return s.lower().replace(" ", "").replace("_", "")

    skills_normalized: list[str] = [normalize(s) for s in skills]
    metric_normalized: str = normalize(metric.value)

    return metric_normalized in skills_normalized


# =====================================================================
# METRIC CALCULATION FUNCTIONS
# =====================================================================

def decision(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Decision metric: Binary check if tool call was expected and made, or vice versa.
    
    Scoring:
        1.0 if: tool was expected and called, OR tool was not expected and not called
        0.0 if: tool was expected but not called, OR tool was not expected but called
        None if metric is not enabled for this query
    """
    if not _is_metric_enabled(inp, Metrics.DECISION):
        return None

    expected: bool = bool(inp["expected_tool"])
    actual_tools: str = out.get("name", "")
    has_tool: bool = bool(actual_tools and actual_tools.strip())
    
    result: float = float(expected == has_tool)
    
    if result == 0.0:
        if expected and not has_tool:
            log_error(
                inp["id"], 
                "Decision",
                "Tool was expected but not called",
                expected=f"Should call: {inp['expected_tool']}",
                actual="No tool called"
            )
        elif not expected and has_tool:
            log_error(
                inp["id"],
                "Decision", 
                "Tool was not expected but was called",
                expected="Should not call tools",
                actual=f"Called: {actual_tools}"
            )
    
    return result


def tool_selection_f1(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Tool Selection F1 metric: F1-score based on set of called tools (order-independent).
    
    Compares the set of tools called against expected tools using precision/recall.
    Rewards correct tools and penalizes extra or missing tools.
    """
    if not _is_metric_enabled(inp, Metrics.TOOL_SELECTION):
        return None

    ref_tools: set[str] = set(t.strip() for t in inp["expected_tool"].split(","))
    actual_tools: set[str] = set(out.get("name", "").split(","))
    actual_tools.discard("")

    # Perfect match case: both empty
    if not ref_tools and not actual_tools:
        return 1.0

    true_positive: int = len(actual_tools & ref_tools)
    false_positive: int = len(actual_tools - ref_tools)
    false_negative: int = len(ref_tools - actual_tools)

    # Calculate precision and recall
    precision: float = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall: float = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0

    # Calculate F1 score
    f1_score: float = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    # Log errors for low scores
    if f1_score < 1.0:
        errors: list[str] = []
        if false_positive > 0:
            extra: set[str] = actual_tools - ref_tools
            errors.append(f"Unexpected tools: {', '.join(extra)}")
        if false_negative > 0:
            missing: set[str] = ref_tools - actual_tools
            errors.append(f"Missing tools: {', '.join(missing)}")
        
        log_error(
            inp["id"],
            "Tool Selection",
            " | ".join(errors),
            expected=f"Should call: {', '.join(ref_tools)}",
            actual=f"Called: {', '.join(actual_tools) if actual_tools else 'None'}"
        )
    
    return f1_score


def params_recall(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Parameters Recall metric: Proportion of expected parameters that were provided correctly.
    
    Only non-None expected parameters are counted. Scoring:
        1.0 if all expected parameters match
        0.0 if no parameters match
        Partial credit for some parameters matching
    """
    if not _is_metric_enabled(inp, Metrics.PARAMS):
        return None

    ref_params: dict[str, Any] = inp["expected_parameters"]
    args_str: str = out.get("arguments", "{}")
    try:
        actual_params: dict[str, Any] = json.loads(args_str)
    except Exception:
        actual_params: dict[str, Any] = {}

    # No expected parameters = perfect score
    if not ref_params:
        return 1.0

    total_count: int = 0
    correct_count: int = 0
    wrong_params: list[str] = []

    for key, value in ref_params.items():
        if value is None:
            # Skip None values - they indicate optional parameters
            continue
        total_count += 1
        actual_value: Any = actual_params.get(key)
        
        if actual_value == value:
            correct_count += 1
        else:
            wrong_params.append(f"{key}: expected '{value}', got '{actual_value}'")

    recall: float = correct_count / total_count if total_count else 1.0
    
    # Log errors for incorrect parameters
    if recall < 1.0 and wrong_params:
        log_error(
            inp["id"],
            "Params",
            f"Incorrect parameters: {' | '.join(wrong_params)}",
            expected=ref_params,
            actual=actual_params
        )
    
    return recall


def result(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Result metric: Combined score of tool selection and parameter correctness.
    
    Average of tool_selection_f1 and params_recall metrics.
    Returns None if either component metric is disabled.
    """
    if not _is_metric_enabled(inp, Metrics.RESULT):
        return None

    tool_correct: float | None = tool_selection_f1(inp, out)
    params_correct: float | None = params_recall(inp, out)

    # If either component metric is disabled, cannot compute combined score
    if tool_correct is None or params_correct is None:
        return None

    return (tool_correct + params_correct) / 2


def execution(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Execution metric: Strict tool call order (chain execution).
    
    Scoring:
        1.0 if tools were called in the exact expected order
        0.0 if order differs or tools are missing/extra
    """
    if not _is_metric_enabled(inp, Metrics.EXECUTION):
        return None

    ref_tools: list[str] = [t.strip() for t in inp["expected_tool"].split(",")]
    actual_tools: list[str] = out.get("name", "").split(",")
    actual_tools = [t for t in actual_tools if t]  # Filter empty strings
    
    is_correct: bool = actual_tools == ref_tools
    
    if not is_correct:
        log_error(
            inp["id"],
            "Execution",
            "Incorrect tool call order",
            expected=f"Order: {' -> '.join(ref_tools)}",
            actual=f"Order: {' -> '.join(actual_tools) if actual_tools else 'None'}"
        )
    
    return float(is_correct)


def noise(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Noise metric: Penalizes extraneous tool calls or parameters.
    
    Scoring:
        1.0 if exactly the expected tools and parameters were used
        0.0 if extra or missing tools/parameters were detected
    """
    if not _is_metric_enabled(inp, Metrics.NOISE):
        return None

    ref_tools: set[str] = set(t.strip() for t in inp["expected_tool"].split(","))
    actual_tools: set[str] = set(out.get("name", "").split(","))
    actual_tools.discard("")

    # Check tool set exactness
    if actual_tools != ref_tools:
        extra: set[str] = actual_tools - ref_tools
        missing: set[str] = ref_tools - actual_tools
        errors: list[str] = []
        if extra:
            errors.append(f"Unexpected tools: {', '.join(extra)}")
        if missing:
            errors.append(f"Missing tools: {', '.join(missing)}")
        
        log_error(
            inp["id"],
            "Noise",
            " | ".join(errors),
            expected=', '.join(ref_tools),
            actual=', '.join(actual_tools) if actual_tools else 'None'
        )
        return 0.0

    # Check parameter set exactness
    ref_params_keys: set[str] = set(inp["expected_parameters"].keys())
    args_str: str = out.get("arguments", "{}")
    try:
        actual_params_keys: set[str] = set(json.loads(args_str).keys())
    except Exception:
        actual_params_keys: set[str] = set()

    if actual_params_keys != ref_params_keys:
        extra: set[str] = actual_params_keys - ref_params_keys
        missing: set[str] = ref_params_keys - actual_params_keys
        errors: list[str] = []
        if extra:
            errors.append(f"Unexpected parameters: {', '.join(extra)}")
        if missing:
            errors.append(f"Missing parameters: {', '.join(missing)}")
        
        log_error(
            inp["id"],
            "Noise",
            " | ".join(errors),
            expected=', '.join(ref_params_keys),
            actual=', '.join(actual_params_keys) if actual_params_keys else 'None'
        )
        return 0.0

    return 1.0


def adaptability(inp: dict[str, Any], last_out: dict[str, Any]) -> float | None:
    """
    Adaptability metric: Check if the final tool call matches expected (tools + parameters).
    
    Scoring:
        1.0 if the last call exactly matches expected tools and parameters
        0.0 if there's any mismatch
    """
    if not _is_metric_enabled(inp, Metrics.ADAPTABILITY):
        return None

    ref_tools: set[str] = set(t.strip() for t in inp["expected_tool"].split(","))
    last_tools: set[str] = set(last_out.get("name", "").split(","))
    last_tools.discard("")

    # Check if last tool call matches
    if last_tools != ref_tools:
        log_error(
            inp["id"],
            "Adaptability",
            "Final tool call does not match expected",
            expected=', '.join(ref_tools),
            actual=', '.join(last_tools) if last_tools else 'None'
        )
        return 0.0

    # Check if final parameters match
    ref_params: dict[str, Any] = {k: v for k, v in inp["expected_parameters"].items() if v is not None}
    args_str: str = last_out.get("arguments", "{}")
    try:
        last_params: dict[str, Any] = json.loads(args_str)
    except Exception:
        last_params: dict[str, Any] = {}

    if last_params != ref_params:
        log_error(
            inp["id"],
            "Adaptability",
            "Final parameters do not match expected",
            expected=ref_params,
            actual=last_params
        )
        return 0.0

    return 1.0


def error_handling(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Error Handling metric: Rewards not calling any tools when none should be called.
    
    Scoring:
        1.0 if no tools were called (correct error handling case)
        0.0 if tools were called when they shouldn't be
    """
    if not _is_metric_enabled(inp, Metrics.ERROR_HANDLING):
        return None

    has_tool: bool = bool(out.get("name", "").strip())
    
    # Log if tool was called when it shouldn't be
    if has_tool:
        log_error(
            inp["id"],
            "Error Handling",
            "Tool called when error handling was required",
            expected="Should not call any tools",
            actual=f"Called: {out.get('name', '')}"
        )
    
    return float(not has_tool)


def ambiguity(inp: dict[str, Any], out: dict[str, Any]) -> float | None:
    """
    Ambiguity metric: Handle ambiguous queries that need clarification or precise matching.
    
    Scenario 1 (requires_clarification = True):
        1.0 if no tools/parameters provided (asked for clarification)
        0.0 if any tool or parameter was provided
        
    Scenario 2 (requires_clarification = False):
        1.0 if tools and parameters exactly match
        0.5 if only tools OR only parameters match
        0.0 if neither match
    """
    if not _is_metric_enabled(inp, Metrics.AMBIGUITY):
        return None

    ref_tools: set[str] = set(t.strip() for t in inp["expected_tool"].split(","))
    actual_tools: set[str] = set(out.get("name", "").split(","))
    actual_tools.discard("")

    ref_params: dict[str, Any] = {k: v for k, v in inp["expected_parameters"].items() if v is not None}
    args_str: str = out.get("arguments", "{}")
    try:
        actual_params: dict[str, Any] = json.loads(args_str)
    except Exception:
        actual_params: dict[str, Any] = {}

    # Handle case where clarification is required
    if inp["requires_clarification"]:
        result: float = float(not actual_tools and not actual_params)
        if result == 0.0:
            log_error(
                inp["id"],
                "Ambiguity",
                "Clarification was required but tools/parameters were provided",
                expected="Should not call tools, must ask for clarification",
                actual=f"Tools: {', '.join(actual_tools) if actual_tools else 'None'}, Parameters: {actual_params}"
            )
        return result

    # Check tool and parameter matches
    tool_correct: bool = actual_tools == ref_tools
    param_correct: bool = actual_params == ref_params

    if tool_correct and param_correct:
        return 1.0
    elif tool_correct or param_correct:
        # Partial credit for one matching component
        errors: list[str] = []
        if not tool_correct:
            errors.append(f"Tools mismatch (expected: {', '.join(ref_tools)}, got: {', '.join(actual_tools) if actual_tools else 'None'})")
        if not param_correct:
            errors.append(f"Parameters mismatch")
        
        log_error(
            inp["id"],
            "Ambiguity",
            " | ".join(errors),
            expected={"tools": list(ref_tools), "params": ref_params},
            actual={"tools": list(actual_tools), "params": actual_params}
        )
        return 0.5
    else:
        # No components match
        log_error(
            inp["id"],
            "Ambiguity",
            "Both tools and parameters do not match",
            expected={"tools": list(ref_tools), "params": ref_params},
            actual={"tools": list(actual_tools), "params": actual_params}
        )
        return 0.0


# =====================================================================
# BATCH EVALUATION AND RESULT PROCESSING
# =====================================================================

def evaluate_batch(
    inputs_for_logging: list[dict[str, Any]],
    outputs_for_logging: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Evaluate all metric scores for a batch of benchmark examples.
    
    Args:
        inputs_for_logging: List of input benchmark examples
        outputs_for_logging: List of corresponding model outputs
        
    Returns:
        DataFrame with one row per example, columns for each metric score
    """
    global detailed_errors
    # Clear error log before new evaluation run
    detailed_errors = []
    
    records: list[dict[str, Any]] = []
    for inp, out in zip(inputs_for_logging, outputs_for_logging, strict=True):
        records.append({
            "id": inp["id"],
            Metrics.DECISION.column_name: decision(inp, out),
            Metrics.TOOL_SELECTION.column_name: tool_selection_f1(inp, out),
            Metrics.PARAMS.column_name: params_recall(inp, out),
            Metrics.ERROR_HANDLING.column_name: error_handling(inp, out),
            Metrics.RESULT.column_name: result(inp, out),
            Metrics.EXECUTION.column_name: execution(inp, out),
            Metrics.NOISE.column_name: noise(inp, out),
            Metrics.ADAPTABILITY.column_name: adaptability(inp, out),
            Metrics.AMBIGUITY.column_name: ambiguity(inp, out),
        })
    return pd.DataFrame(records)


def print_detailed_errors(threshold: float = 0.8) -> None:
    """
    Print detailed error information grouped by query.
    
    Args:
        threshold: Score threshold for filtering (currently unused, kept for API compatibility)
    """
    if not detailed_errors:
        console.print("[green]✓ No errors found![/green]\n")
        return
    
    console.print(Panel.fit(
        f"[bold red]DETAILED ERROR ANALYSIS ({len(detailed_errors)} issues)[/bold red]",
        border_style="red"
    ))
    console.print()
    
    # Group errors by query ID
    errors_by_id: dict[str, list[dict[str, Any]]] = {}
    for error in detailed_errors:
        query_id: str = error["id"]
        if query_id not in errors_by_id:
            errors_by_id[query_id] = []
        errors_by_id[query_id].append(error)
    
    # Display errors grouped by query
    for query_id, errors in errors_by_id.items():
        error_table: Table = Table(
            title=f"[bold yellow]Query ID: {query_id}[/bold yellow]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold red"
        )
        
        error_table.add_column("Metric", style="cyan", width=20)
        error_table.add_column("Error Reason", style="white", width=50)
        error_table.add_column("Expected", style="green", width=30)
        error_table.add_column("Actual", style="red", width=30)
        
        for err in errors:
            expected_str: str = str(err["expected"])[:50] if err["expected"] else "N/A"
            actual_str: str = str(err["actual"])[:50] if err["actual"] else "N/A"
            
            error_table.add_row(
                err["metric"],
                err["reason"],
                expected_str,
                actual_str
            )
        
        console.print(error_table)
        console.print()


def print_benchmark_results(
    df: pd.DataFrame,
    inputs_for_logging: list[dict[str, Any]],
    benchmark_file: str = "benchmark_results.json"
) -> None:
    """
    Print and save benchmark evaluation results.
    
    Args:
        df: DataFrame with metric scores from evaluate_batch()
        inputs_for_logging: Input examples (for statistics)
        benchmark_file: Path to benchmark_results.json (contains config)
    """
    # Get all metric column names
    metric_cols: list[str] = [m.column_name for m in Metrics]
    
    total_queries: int = len(df)
    
    # Count successful queries (all metrics score 1.0)
    def is_successful(row: pd.Series) -> bool:
        """Check if all evaluated metrics are perfect (1.0)."""
        valid: pd.Series = row[metric_cols].dropna()
        if len(valid) == 0:
            return False
        return all(valid == 1.0)
    
    successful: int = int(df.apply(is_successful, axis=1).sum())
    failed: int = total_queries - successful
    success_rate: float = (successful / total_queries * 100) if total_queries > 0 else 0
    failed_rate: float = 100 - success_rate
    
    # Calculate final weighted score
    final: float = calculate_final_score(df)
    
    # Calculate mean scores per metric
    means: pd.Series = df[metric_cols].mean(skipna=True)

    # Prepare metrics display dictionary
    metrics_display: dict[str, float] = {}
    for m in metric_cols:
        if m in means.index:
            metrics_display[m] = means[m]
        else:
            metrics_display[m] = float("nan")

    # ============ OUTPUT DISPLAY ============
    console.print()
    
    # Title panel
    console.print(Panel.fit(
        "[bold cyan]BENCHMARK RESULTS[/bold cyan]",
        border_style="cyan"
    ))
    
    # Summary statistics table
    stats_table: Table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    stats_table.add_column(style="bold white", justify="left")
    stats_table.add_column(style="cyan", justify="right")
    
    stats_table.add_row("Total queries", f"{total_queries}")
    stats_table.add_row(
        "Successful",
        f"[green]{successful}[/green] ([green]{success_rate:.1f}%[/green])"
    )
    stats_table.add_row(
        "Failed",
        f"[red]{failed}[/red] ([red]{failed_rate:.1f}%[/red])"
    )
    
    console.print(stats_table)
    console.print()
    
    # Final score with color coding
    if final >= 80:
        score_color: str = "green"
    elif final >= 60:
        score_color: str = "yellow"
    else:
        score_color: str = "red"
    
    console.print(Panel(
        f"[bold {score_color}]{final:.1f}%[/bold {score_color}]",
        title="[bold white]FINAL MODEL SCORE[/bold white]",
        border_style=score_color,
        padding=(0, 4)
    ))
    console.print()
    
    # Metrics summary table
    metrics_table: Table = Table(
        title="[bold white]METRICS[/bold white]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    metrics_table.add_column("Metric", style="white", justify="left", width=25)
    metrics_table.add_column("Score", justify="center", width=10)
    metrics_table.add_column("Visualization", justify="left", width=25)

    # Display each metric with score and visualization
    for metric in Metrics:
        metric_value: float | None = means.get(metric.column_name)
        
        if pd.isna(metric_value):
            # Metric not evaluated for this query set
            value_str: str = "[white]N/A[/white]"
            bar: str = "░" * 20
            metrics_table.add_row(metric.value, value_str, bar)
            continue

        # Color code based on score threshold
        if metric_value >= 0.8:
            value_style: str = "green"
        elif metric_value >= 0.6:
            value_style: str = "yellow"
        else:
            value_style: str = "red"

        # Create visual bar representation
        bar_length: int = int(metric_value * 20)
        bar: str = "█" * bar_length + "░" * (20 - bar_length)

        metrics_table.add_row(
            metric.value,
            f"[{value_style}]{metric_value:.2f}[/{value_style}]",
            f"[{value_style}]{bar}[/{value_style}]"
        )
    
    console.print(metrics_table)
    console.print()
    
    # Display detailed error analysis
    # print_detailed_errors()
    
    # Save results to JSON file
    try:
        output: dict[str, Any] = {
            "config": {},
            "total_queries": total_queries,
            "successful": int(successful),
            "failed": int(failed),
            "success_rate": round(success_rate, 2),
            "final_score": round(final, 2),
            "metrics_mean": {
                m.column_name: (None if pd.isna(means.get(m.column_name)) else round(means.get(m.column_name), 4))
                for m in Metrics
            },
        }
        
        # Try to load benchmark config if available
        try:
            with open(benchmark_file, encoding="utf-8") as f:
                bench_data: dict[str, Any] = json.load(f)
                print(f"DEBUG evaluation.py: Found {benchmark_file}")
                if "config" in bench_data:
                    output["config"] = bench_data["config"]
                    print(
                        f"DEBUG: Config saved to evaluate_results.json: temperature={output['config'].get('temperature')}, "
                        f"top_p={output['config'].get('top_p')}, top_k={output['config'].get('top_k')}, "
                        f"seed={output['config'].get('seed')}"
                    )
                else:
                    print(f"DEBUG: No 'config' key in {benchmark_file}")
        except Exception as e:
            print(f"DEBUG: Exception reading {benchmark_file}: {e}")
        
        # Write results to file
        with open("evaluate_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        console.print("[green]Results saved to evaluate_results.json[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save results to JSON: {e}[/red]")


def save_detailed_errors_json(filepath: str = "detailed_errors.json") -> None:
    """
    Save detailed error logs to a JSON file.
    
    Args:
        filepath: Output file path for the error log
    """
    if not detailed_errors:
        return

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Parse command-line arguments for custom benchmark file path
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_results.json",
        help="Path to benchmark results JSON file"
    )

    args = parser.parse_args()
    # Update environment variable (already set above, but ensuring it's current for explicit script execution)
    os.environ["BENCHMARK_FILE"] = args.input

    # Get pre-loaded input and output data from modules
    inputs_for_logging = get_inputs_for_logging()
    outputs_for_logging = get_outputs_for_logging()

    # Validate that inputs and outputs match and are aligned
    validate_ids(inputs_for_logging, outputs_for_logging)

    # Evaluate all metrics for each benchmark example
    df: pd.DataFrame = evaluate_batch(inputs_for_logging, outputs_for_logging)

    # Display and save benchmark results
    print_benchmark_results(df, inputs_for_logging, benchmark_file=args.input)
    
    # Optionally save detailed error log
    # save_detailed_errors_json("detailed_errors.json")
