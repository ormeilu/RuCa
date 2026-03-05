"""Parser for model outputs from benchmark results.

This module handles reading and extracting tool calls from benchmark result files,
preparing data for metric evaluation.
"""

import json
import os
from typing import Any

# Get benchmark file path from environment variable or use default
RESULTS_FILE: str = os.environ.get("BENCHMARK_FILE", "benchmark_results.json")


def read_benchmark_results(filepath: str) -> dict[str, Any]:
    """Read benchmark results from a JSON file.

    Handles both simple result dictionaries and structured formats
    with separate 'config' and 'results' keys.

    Args:
        filepath: Path to the benchmark results JSON file

    Returns:
        Dictionary of benchmark results (query_id -> result_data)
    """
    with open(filepath, encoding="utf-8") as f:
        data: dict[str, Any] | list[Any] = json.load(f)

    # Handle structured format with config and results sections
    if isinstance(data, dict) and "results" in data and "config" in data:
        return data["results"]

    return data if isinstance(data, dict) else {}


def extract_output(query_id: str, item: dict[str, Any]) -> dict[str, Any]:
    """Extract tool call information from model output.

    Handles both single and multiple tool calls, extracting:
    - Tool name(s)
    - Tool parameters

    Normalizes string values to lowercase for consistent comparison.

    Args:
        query_id: Unique identifier for the query
        item: Result item containing agent_response data

    Returns:
        Dictionary with id, tool name(s), and JSON-serialized arguments
    """
    agent_response: dict[str, Any] = item.get("agent_response", {})

    # Handle multiple tool calls (tool_calls list)
    tool_calls_list: list[dict[str, Any]] = agent_response.get("tool_calls", [])

    if tool_calls_list and isinstance(tool_calls_list, list):
        names: list[str] = []
        all_parameters: dict[str, Any] = {}

        for tool_call in tool_calls_list:
            name: str = tool_call.get("name", "").lower()
            parameters: dict[str, Any] = tool_call.get("parameters", {})

            names.append(name)

            # Normalize parameter values (lowercase strings)
            for key, value in parameters.items():
                if isinstance(value, str):
                    all_parameters[key] = value.lower()
                else:
                    all_parameters[key] = value

        name_str: str = ",".join(names)
        arguments: str = json.dumps(all_parameters, ensure_ascii=False)

        output: dict[str, Any] = {"id": query_id, "name": name_str, "arguments": arguments}
        return output

    # Handle single tool call
    tool_call: dict[str, Any] | None = agent_response.get("tool_call")

    if tool_call is None or not isinstance(tool_call, dict):
        output: dict[str, Any] = {"id": query_id, "name": "", "arguments": ""}
        return output

    name: str = tool_call.get("name", "").lower()
    parameters: dict[str, Any] = tool_call.get("parameters", {})

    # Normalize parameter values (lowercase strings)
    parameters_lower: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, str):
            parameters_lower[key] = value.lower()
        else:
            parameters_lower[key] = value

    arguments: str = json.dumps(parameters_lower, ensure_ascii=False)

    output: dict[str, Any] = {"id": query_id, "name": name, "arguments": arguments}

    return output


def process_benchmark_results(filepath: str | None = None) -> list[dict[str, Any]]:
    """Process benchmark results from a JSON file.

    Reads the benchmark results file and extracts tool call information
    for each query, preparing data for metric evaluation.

    Args:
        filepath: Path to the benchmark results file. If None, uses BENCHMARK_FILE
                 from environment variable or default value

    Returns:
        List of processed output dictionaries with extracted tool calls
    """
    if filepath is None:
        filepath = os.environ.get("BENCHMARK_FILE", "benchmark_results.json")

    data: dict[str, Any] = read_benchmark_results(filepath)

    outputs_for_logging: list[dict[str, Any]] = []

    for query_id, item in data.items():
        output_data: dict[str, Any] = extract_output(query_id, item)
        outputs_for_logging.append(output_data)

    return outputs_for_logging


def get_outputs_for_logging() -> list[dict[str, Any]]:
    """Lazily load and return model output data for metric evaluation.

    This function is called on demand to avoid loading data during import.
    Reads from the benchmark results file specified in the BENCHMARK_FILE
    environment variable.

    Returns:
        List of model outputs formatted for metric evaluation
    """
    return process_benchmark_results()
