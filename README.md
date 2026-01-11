# RuCa
RuCa Benchmark (pronounced "roo-ka") - Russian Tool Calling Benchmark for LLM

**Languages:** [English](README.md) | [Русский](README_ru.md)

To run the benchmark:

```bash
uv run ruca --runs 2 --concurrent 8 --json
```

## Table of Contents
1. [Introduction](#introduction)
2. [Running the Benchmark](#running-the-benchmark)
3. [Dataset Structure](#dataset-structure)
4. [Testing Scenarios](#testing-scenarios)
5. [Metrics System](#metrics-system)
6. [Leaderboard](#leaderboard)

## Introduction

RuCa Benchmark is a benchmark for evaluating the tool calling capabilities of large language models (LLMs). The project measures the accuracy of tool selection, correctness of parameter passing, and handling of various types of queries in Russian.

## Running the Benchmark

### Installation

```bash
# Clone the repository
git clone https://github.com/ormeilu/RuCa.git
cd RuCa

# Install dependencies
uv sync
```

### Environment Configuration

Create a `.env` file based on `env.template`.

Fill in the `.env` with your API_KEY and BASE_URL.

The central script (`main.py`) manages running the benchmark for all models described in `config.yaml`, and sequentially launches them with parameters specified in the config.

### Running

```bash
# Run with custom parameters
uv run main.py --runs 2 --concurrent 8 --json
```

**Parameters:**
- `--runs` — number of repeated dataset runs through the model for averaging results
- `--concurrent` — number of requests for asynchronous submission to the model
- `--json` — output results in JSON format

## Dataset Structure

The dataset consists of JSON files with a set of test queries, each containing expected data that the model should return.

### Record Format

```json
{
    "queries_basic": [
        {
            "id": "000000",
            "complexity": "easy",
            "category": "tool_basic",
            "type": "ordinary",
            "query": "Сколько сейчас времени в Токио?",
            "expected_tool": "get_time",
            "expected_parameters": {
                "format": "24h"
            },
            "requires_clarification": false,
            "skills": ["Decision", "Tool selection", "Params", "Result"]
        }
    ]
}
```

### Field Descriptions

| Field | Type | Description |
|------|-----|----------|
| `id` | string | Unique query identifier |
| `complexity` | enum | Complexity: `easy`, `medium`, `hard` |
| `category` | string | Domain for the query (e.g., `tool_basic`, `tool_airlines`, etc.) |
| `type` | string | Scenario type for the query: `ordinary`, `misprint`, `ambiguous`, etc. |
| `query` | string | User query text |
| `expected_tool` | string | Expected tool to call |
| `expected_parameters` | object | Expected call parameters |
| `requires_clarification` | boolean | Whether clarification from user is required |
| `skills` | array | List of metrics being tested |

---

## Testing Scenarios

The benchmark includes 8 scenarios that test different aspects of models' tool calling capabilities.

### 1. Explicit Queries

**What it tests:** The model's ability to execute direct instructions for calling a specific tool.

**Example:**
```
"query": "Вызови airbnb_search для Москвы на двоих взрослых"
```

---

### 2. Implicit Queries

**What it tests:** The model's ability to independently determine the needed tool from the query context.

**Example:**
```
"query": "Хочу забронировать билет из Москвы в Париж на 2025-06-10 эконом класс, меня зовут Иван Иванов"
```

---

### 3. Queries with Typos

**What it tests:** The model's robustness to errors and ability to understand the query despite typos.

**Example:**
```
"query": "Ищу жилье в Санкт-Питербурге на троих с 15 по 20 декабрья"
```

---

### 4. Queries with Noise

**What it tests:** The model's ability to filter irrelevant information and focus on the essence of the query.

**Example:**
```
"query": "Добавь в корзину кофеварку DeLonghi (ID 3125), кстати, мой e-mail — ivanov88@gmail.com"
```

---

### 5. Adaptive Queries

**What it tests:** The model's ability to adapt to changes in user instructions.

**Example:**
```
"query": "Переведи 100 долларов в евро... нет, лучше в японские иены"
```

---

### 6. Ambiguous Queries

**What it tests:** The model's ability to recognize lack of information and request clarification instead of guessing.

**Example:**
```
"query": "Какая погода там, где сейчас полдень?"
```

---

### 7. Sequential Calling Queries

**What it tests:** The model's ability to execute tools in strict order when the result of one call is required for the next.

**Example:**
```
"query": "Найди мне недорогие беспроводные наушники и сразу переведи их цену в доллары."
```

---

### 8. Error Handling Check Queries

**What it tests:** The model's ability to correctly handle situations when a non-existent tool is requested.

**Characteristic:** User asks to call a tool that is not in the list of available tools.

**Example:**
```
"query": "Используй ChekInn_Toooor чтобы сделать мне регистрацию на рейс LH900"
```

---

## Metrics System

The benchmark uses a metrics system for comprehensive evaluation of models' tool calling capabilities.

#### Decision — Checking if Tools Were Called

**What it evaluates:** Whether the agent decided to use tools.

**Values:** `1/0` (binary metric)
- `1` — model called a tool
- `0` — model did not call a tool

---

#### Tool Selection — Matching Expected and Called Tools

**What it evaluates:** Correctness of selecting a specific tool from available ones.

**Values:** `from 0 to 1`

---

#### Param — Matching Expected and Actually Used Parameters

**What it evaluates:** Correctness of parameter passing to tools.

**Values:** `from 0 to 1` (continuous metric)

---

#### Result — Comprehensive Metric

**What it evaluates:** Overall execution result, including evaluation of both Tool Selection and Param.

**Values:** `from 0 to 1`

---

#### Adaptability — Adaptation to Query Changes

**What it evaluates:** The model's ability to call only the last tool from those requested in a changing query.

**Values:** `1/0` (binary metric)

---

#### Ambiguity — Evaluation of Working with Ambiguous Queries

**What it evaluates:** Working with queries containing typos, or ambiguous queries lacking parameters for execution.

**Values:** `1/0.5/0`

**Evaluation Scenarios:**

**Scenario 1: `requires_clarification = True`**
- `1` — model **did not call** a tool AND **did not pass** parameters (requested clarification)
- `0` — model called anything (incorrect behavior)

**Scenario 2: `requires_clarification = False`**
- `1` — exact match (tool + params)
- `0.5` — only tool or only params matched
- `0` — neither tool nor params matched

---

#### Noise — Ignoring Noise

**What it evaluates:** The model's ability to ignore noise in the query: not pay attention to it, not include it in output, and not call extra tools.

**Values:** `1/0` (binary metric)

---

#### Error Handling — Handling Error Situations

**What it evaluates:** The agent's ability not to call tools if a non-existent tool is requested.

**Values:** `1/0` (binary metric)

---

#### Execution — Following Strict Order of Tool Calls

**What it evaluates:** The agent's ability to perform sequential operations in the correct order.

**Values:** `1/0` (binary metric)

---

### Final Score Formula

The final score is calculated differently depending on the number of applicable metrics for each query:

#### For Queries with 4 Basic Metrics

```
Score = 0.30×Decision + 0.30×ToolSelection + 0.22×Param + 0.18×Result
```

**Applied to:** Explicit and implicit queries without additional checks

#### For Queries with 5 Metrics (4 basic + 1 specialized)

```
Score = 0.28×Decision + 0.28×ToolSelection + 0.20×Param + 0.04×Result + 0.20×SpecificMetric
```

where `SpecificMetric` is one of: Ambiguity, Noise, Adaptability, ErrorHandling, Execution

**Applied to:** Queries testing specific skills

#### Model Final Score

```python
FinalScore = (Σ ScorePerQuery / TotalQueries) × 100
```

**Calculation Example:**

Model completed 360 queries:
- 270 queries with 4 metrics (average score: 0.65)
- 90 queries with 5 metrics (average score: 0.55)

```
FinalScore = ((270×0.65 + 90×0.55) / 360) × 100 = 62.5
```

### Results Interpretation

| Range | Level | Interpretation |
|----------|---------|---------------|
| 90-100 | Excellent | Model demonstrates high accuracy in all aspects of tool calling |
| 70-89 | Good | Model reliably works with most scenarios |
| 50-69 | Average | Model handles basic tasks but has problems in complex scenarios |
| 30-49 | Low | Model experiences significant difficulties with tool calling |
| 0-29 | Critical | Model is unable to correctly work with tools |

---

## Leaderboard

Current results of all tested models are available on the public leaderboard:
https://de.kotyan.com