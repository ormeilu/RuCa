import json
from typing import Dict, Any, List

from onemoreagent import BenchmarkAgent
import time


from json_parser import process_all_queries 



def run_benchmark(
    system_prompt: str,
    inputs_for_llm: List[Dict[str, Any]],
    inputs_for_logging: List[Dict[str, Any]],
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Запускает все запросы через агента и собирает результаты в словарь.
    
    Args:
        system_prompt: Системный промпт для агента
        inputs_for_llm: Список запросов для модели
        inputs_for_logging: Список ground truth данных
        model: Название модели
        verbose: Выводить прогресс
    
    Returns:
        Словарь с результатами бенчмарка
    """
    
   
    if verbose:
        print(f"\n{'='*70}")
        print("🚀 RUSSIAN TOOL CALLING BENCHMARK")
        print(f"{'='*70}")
        print(f"Model: {model}")
        print(f"Total queries: {len(inputs_for_llm)}")
        print(f"{'='*70}\n")
    
    agent = BenchmarkAgent(
        model=model,
        use_retail=True,
        use_weather=True,
        use_translate=True,
        use_calculator=True,
        use_trash=False,  
        verbose=False  
    )
    
    if verbose:
        print(f"✅ Agent initialized with {len(agent.get_tools_info())} tools\n")
    
    results = {}
    start_time = time.time()
    
    # 2) Прогоняем через агента
    for idx, (request_data, ground_truth) in enumerate(zip(inputs_for_llm, inputs_for_logging), 1):
        query_id = request_data["id"]
        
        if verbose:
            print(f"[{idx}/{len(inputs_for_llm)}] Processing: {query_id}")
            print(f"   Query: {request_data['user_query'][:80]}...")
        
        try:
            # Вызываем агента
            agent_output = agent.run_single_query(
                user_query=request_data["user_query"],
                system_prompt=system_prompt,
                query_id=query_id
            )
            
            if verbose:
                if agent_output["tool_call"]:
                    print(f"    Tool: {agent_output['tool_call']['name']}")
                elif agent_output["clarification_question"]:
                    print(f"    Clarification needed")
                elif agent_output["user_message"]:
                    print(f"    Text response")
                else:
                    print(f"     No output")
        
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            
            
            agent_output = {
                "tool_call": None,
                "clarification_question": None,
                "user_message": None,
                "internal": {
                    "reasoning": None,
                    "raw_response": None,
                    "errors": f"RUNNER_ERROR: {e}"
                }
            }
        
        
        results[query_id] = {
            "id": query_id,
            "user_query": request_data["user_query"],
            "agent_response": agent_output,
            "ground_truth": {
                "expected_tool": ground_truth["expected_tool"],
                "expected_parameters": ground_truth["expected_parameters"],
                "requires_clarification": ground_truth["requires_clarification"],
                "skills": ground_truth["skills"]
            }
        }
        
        if verbose:
            print()  
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"{'='*70}")
        print(f"✅ Benchmark completed in {elapsed_time:.2f} seconds")
        print(f"   Average time per query: {elapsed_time/len(inputs_for_llm):.2f}s")
        print(f"{'='*70}\n")
    
    return results



def save_results(results: Dict[str, Any], filename: str = "benchmark_results.json"):
    """
    Сохранение результатов в JSON файл.
    
    Args:
        results: Словарь с результатами
        filename: Имя файла для сохранения
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {filename}")



def print_statistics(results: Dict[str, Any]):
    """
    Выводит базовую статистику по результатам.
    
    Args:
        results: Словарь с результатами бенчмарка
    """
    total = len(results)
    tool_calls = 0
    clarifications = 0
    text_responses = 0
    errors = 0
    
    for result in results.values():
        agent_response = result["agent_response"]
        
        if agent_response.get("tool_call"):
            tool_calls += 1
        elif agent_response.get("clarification_question"):
            clarifications += 1
        elif agent_response.get("user_message"):
            text_responses += 1
        
        if agent_response.get("internal", {}).get("errors"):
            errors += 1
    
    print(f"\n{'='*70}")
    print("📊 STATISTICS")
    print(f"{'='*70}")
    print(f"Total queries:        {total}")
    print(f"Tool calls:           {tool_calls} ({tool_calls/total*100:.1f}%)")
    print(f"Clarifications:       {clarifications} ({clarifications/total*100:.1f}%)")
    print(f"Text responses:       {text_responses} ({text_responses/total*100:.1f}%)")
    print(f"Errors:               {errors} ({errors/total*100:.1f}%)")
    print(f"{'='*70}\n")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Russian Tool Calling Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
        help="Model name to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output filename"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    
    args = parser.parse_args()
    
    
    from json_parser import process_all_queries, system_prompt
    inputs_for_llm, inputs_for_logging = process_all_queries(system_prompt=system_prompt)
    
    
    
    
    results = run_benchmark(
        system_prompt=system_prompt,
        inputs_for_llm=inputs_for_llm,
        inputs_for_logging=inputs_for_logging,
        model=args.model,
        verbose=not args.quiet
    )
    

    save_results(results, filename=args.output)
    
