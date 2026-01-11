import json
import os

# Получаем путь из переменной окружения или используем значение по умолчанию
RESULTS_FILE = os.environ.get("BENCHMARK_FILE", "benchmark_results.json")


def read_benchmark_results(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # Если data имеет структуру с config и results, извлекаем только results
    if isinstance(data, dict) and "results" in data and "config" in data:
        return data["results"]

    return data


# Извлекаем нужные данные из bencchmark_results.json
def extract_output(query_id, item):
    agent_response = item.get("agent_response", {})

    # Для запросов с несколькими tool_calls
    tool_calls_list = agent_response.get("tool_calls", [])

    if tool_calls_list and isinstance(tool_calls_list, list):
        names = []
        all_parameters = {}

        for tool_call in tool_calls_list:
            name = tool_call.get("name", "").lower()
            parameters = tool_call.get("parameters", {})

            names.append(name)

            for key, value in parameters.items():
                if isinstance(value, str):
                    all_parameters[key] = value.lower()
                else:
                    all_parameters[key] = value

        name_str = ",".join(names)
        arguments = json.dumps(all_parameters, ensure_ascii=False)

        output = {"id": query_id, "name": name_str, "arguments": arguments}
        return output

    # Для запросов с одинм tool_call
    tool_call = agent_response.get("tool_call")

    if tool_call is None or not isinstance(tool_call, dict):
        output = {"id": query_id, "name": "", "arguments": ""}
        return output

    name = tool_call.get("name", "").lower()
    parameters = tool_call.get("parameters", {})

    parameters_lower = {}
    for key, value in parameters.items():
        if isinstance(value, str):
            parameters_lower[key] = value.lower()
        else:
            parameters_lower[key] = value

    arguments = json.dumps(parameters_lower, ensure_ascii=False)

    output = {"id": query_id, "name": name, "arguments": arguments}

    return output


# прогоняем все и заносим в список
def process_benchmark_results(filepath=RESULTS_FILE):
    data = read_benchmark_results(filepath)

    outputs_for_logging = []

    for query_id, item in data.items():
        output_data = extract_output(query_id, item)
        outputs_for_logging.append(output_data)

    return outputs_for_logging


outputs_for_logging = process_benchmark_results(RESULTS_FILE)
