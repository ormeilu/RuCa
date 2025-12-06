import json

#вставить путь до файла с результатами
RESULTS_FILE = "/home/alena-kuriatnikova/ML модуль/benchmark/benchmark_results.json"

#читаем файл
def read_benchmark_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

#извлекаем нужные данные из json-файла
def extract_output(query_id, item):
    agent_response = item.get("agent_response", {})
    internal_data = agent_response.get("internal", {})
    raw_response = internal_data.get("raw_response", {})
    tool_calls_list = raw_response.get("tool_calls", [])
    
    name = ""
    arguments = ""
    if tool_calls_list and len(tool_calls_list) > 0:
        first_tool_call = tool_calls_list[0]
        name = first_tool_call.get("name", "")
        arguments = first_tool_call.get("arguments", "")
    
    output = {
        "id": query_id,
        "name": name,
        "arguments": arguments
    }
    
    return output

#прогоняем все и заносим в список
def process_benchmark_results(filepath=RESULTS_FILE):
    data = read_benchmark_results(filepath)
    
    outputs_for_logging = []
    
    for query_id, item in data.items():
        output_data = extract_output(query_id, item)
        outputs_for_logging.append(output_data)
    
    return outputs_for_logging

outputs_for_logging = process_benchmark_results(RESULTS_FILE)

