"""CLI entry point for running multi-model benchmarks and comparing results."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ruca.utils import get_all_models

console = Console()

RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_OUTPUT: Path = Path("evaluate_results.json")


def run_model_benchmark(model_name: str, model_config: dict[str, Any], concurrent: int, run_id: int) -> None:
    """Run the benchmark pipeline for a single model.

    Spawns ``four_more_agent`` as a subprocess, then runs evaluation and
    copies the result into :data:`RESULTS_DIR`.

    Args:
        model_name: Human-readable model identifier (config key).
        model_config: Resolved model parameters dict.
        concurrent: Maximum number of concurrent requests.
        run_id: Ordinal index of the current benchmark run.
    """
    print(f"\n{'=' * 70}")
    print(f"Запуск модели: {model_name}")
    print(f"Model: {model_config['model_name']}")
    print(f"Temperature: {model_config['temperature']}, Top-P: {model_config['top_p']}, Top-K: {model_config['top_k']}")
    print(f"{'=' * 70}\n")

    benchmark_output: str = f"benchmark_results_{model_name}_run{run_id}.json"

    # Build the CLI command for the benchmark agent
    cmd: list[str] = [
        "uv",
        "run",
        "-m",
        "ruca.agents.four_more_agent",
        "--model",
        model_config["model_name"],
        "--concurrent",
        str(concurrent),
        "--temperature",
        str(model_config["temperature"]),
        "--output",
        benchmark_output,
    ]

    # Append optional sampling parameters when provided
    if model_config.get("top_p") is not None:
        cmd.extend(["--top_p", str(model_config["top_p"])])

    if model_config.get("top_k") is not None:
        cmd.extend(["--top_k", str(model_config["top_k"])])

    if model_config.get("seed") is not None:
        cmd.extend(["--seed", str(model_config["seed"])])

    # Forward API credentials via environment variables if specified
    import os

    env: dict[str, str] = os.environ.copy()
    if model_config.get("api_key"):
        env["OPENAI_API_KEY"] = model_config["api_key"]
    if model_config.get("base_url"):
        env["OPENAI_BASE_URL"] = model_config["base_url"]

    subprocess.run(cmd, check=True, env=env)

    # Run evaluation on the freshly produced benchmark output
    subprocess.run(["uv", "run", "-m", "ruca.utils.evaluation", "--input", benchmark_output], check=True)

    # Persist the evaluation result under the model-specific run file
    target: Path = RESULTS_DIR / f"run_{model_name}_{run_id}.json"
    shutil.copy(EVAL_OUTPUT, target)

    print(f"✅ Модель {model_name} (запуск {run_id}) завершена → {target}\n")


def compute_model_averages(model_names: list[str], runs: int) -> dict[str, Any]:
    """Compute mean metric values for each model across all runs.

    Args:
        model_names: List of model identifier strings.
        runs: Total number of benchmark runs performed.

    Returns:
        Dict with ``averages`` and ``configs`` sub-dicts keyed by model name.
    """
    all_averages: dict[str, dict[str, float]] = {}
    all_configs: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        collected = {}
        config_data = {}

        for run_id in range(1, runs + 1):
            file = RESULTS_DIR / f"run_{model_name}_{run_id}.json"
            if not file.exists():
                continue

            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            if "config" in data and not config_data:
                config_data = data["config"]

            if "metrics_mean" in data and isinstance(data["metrics_mean"], dict):
                for metric_name, metric_value in data["metrics_mean"].items():
                    if metric_value is not None:
                        collected.setdefault(metric_name, []).append(metric_value)

            for key, value in data.items():
                if key not in ["metrics_mean", "config"] and isinstance(value, (int, float)):
                    collected.setdefault(key, []).append(value)

        averages = {}
        for metric_name, values in collected.items():
            if values:
                averages[metric_name] = round(mean(values), 4)

        all_averages[model_name] = averages
        all_configs[model_name] = config_data

    return {"averages": all_averages, "configs": all_configs}


def print_comparison_table(model_averages: dict[str, Any], configs: dict[str, Any]) -> None:
    """Print a rich-formatted comparison table of all model metrics."""
    console.print()
    console.print(Panel.fit("[bold cyan]СРАВНЕНИЕ МОДЕЛЕЙ[/bold cyan]", border_style="cyan"))
    console.print()

    metric_cols: list[str] = [
        "decision",
        "tool_selection",
        "params",
        "result",
        "ambiguity",
        "noise",
        "adaptability",
        "error_handling",
        "execution",
    ]

    comparison_table = Table(
        title="[bold white]МЕТРИКИ ПО МОДЕЛЯМ[/bold white]", box=box.ROUNDED, show_header=True, header_style="bold cyan"
    )

    comparison_table.add_column("Метрика", style="white", justify="left", width=25)

    for model_name in model_averages:
        comparison_table.add_column(model_name, justify="center", width=12)

    for metric_name in metric_cols:
        row = [metric_name]
        for _model_name, averages in model_averages.items():
            metric_value = averages.get(metric_name)

            if metric_value is None or (isinstance(metric_value, float) and math.isnan(metric_value)):
                row.append("[white]N/A[/white]")
            else:
                if metric_value >= 0.8:
                    color = "green"
                elif metric_value >= 0.6:
                    color = "yellow"
                else:
                    color = "red"
                row.append(f"[{color}]{metric_value:.2f}[/{color}]")

        comparison_table.add_row(*row)

    console.print(comparison_table)
    console.print()

    final_scores_table = Table(
        title="[bold white]ФИНАЛЬНЫЕ ОЦЕНКИ[/bold white]", box=box.ROUNDED, show_header=True, header_style="bold cyan"
    )

    final_scores_table.add_column("Модель", style="white", justify="left", width=25)
    final_scores_table.add_column("Финальный скор", justify="center", width=15)
    final_scores_table.add_column("Success Rate", justify="center", width=15)

    for model_name, averages in model_averages.items():
        final_score = averages.get("final_score", 0)
        success_rate = averages.get("success_rate", 0)

        if final_score >= 80:
            color = "green"
        elif final_score >= 60:
            color = "yellow"
        else:
            color = "red"

        final_scores_table.add_row(
            model_name, f"[{color}]{final_score:.1f}%[/{color}]", f"[cyan]{success_rate:.1f}%[/cyan]"
        )

    console.print(final_scores_table)
    console.print()

    config_table = Table(
        title="[bold white]КОНФИГУРАЦИЯ МОДЕЛЕЙ[/bold white]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    config_table.add_column("Модель", style="white", justify="left", width=20)
    config_table.add_column("Temperature", justify="center", width=12)
    config_table.add_column("Top-P", justify="center", width=10)
    config_table.add_column("Top-K", justify="center", width=10)
    config_table.add_column("Seed", justify="center", width=10)

    for model_name, config in configs.items():
        config_table.add_row(
            model_name,
            str(config.get("temperature", "N/A")),
            str(config.get("top_p", "N/A")),
            str(config.get("top_k", "N/A")),
            str(config.get("seed", "N/A")),
        )

    console.print(config_table)
    console.print()


def save_all_results_json(
    model_averages: dict[str, Any], configs: dict[str, Any], filename: str = "comparison_results.json"
) -> None:
    """Persist aggregated comparison results for all models to a JSON file."""
    output: dict[str, Any] = {
        "averages": model_averages,
        "configs": configs,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✓ Результаты сравнения сохранены в: {filename}")


def main() -> None:
    """CLI entry point: parse arguments, run benchmarks for every configured model, and display results."""
    parser = argparse.ArgumentParser(description="Run benchmark for multiple models from config.yaml")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    parser.add_argument("--concurrent", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--json", action="store_true", help="Save results to JSON")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    models: dict[str, dict[str, Any]] = get_all_models(args.config)

    if not models:
        console.print("[red]❌ Не найдены модели в config.yaml[/red]")
        return

    console.print(f"[cyan]Найдены модели: {', '.join(models.keys())}[/cyan]")
    console.print(f"[cyan]Количество запусков: {args.runs}[/cyan]\n")

    for run_id in range(1, args.runs + 1):
        for model_name, model_config in models.items():
            run_model_benchmark(model_name, model_config, args.concurrent, run_id)

    results: dict[str, Any] = compute_model_averages(list(models.keys()), args.runs)
    model_averages: dict[str, Any] = results["averages"]
    configs: dict[str, Any] = results["configs"]

    if args.json:
        save_all_results_json(model_averages, configs)
    else:
        print_comparison_table(model_averages, configs)


if __name__ == "__main__":
    main()
