"""Enumeration of benchmark evaluation metrics.

This module defines all metrics used for evaluating model performance on the benchmark.
Each metric has a corresponding column name for use in DataFrame operations.
"""

from enum import StrEnum


class Metrics(StrEnum):
    """Enumeration of all benchmark evaluation metrics.
    
    Each metric evaluates a different aspect of model performance:
    - DECISION: Whether tool call decision matches expectation (binary)
    - TOOL_SELECTION: F1-score on correct tool selection (set-based, order-independent)
    - PARAMS: Recall on correct parameter values
    - RESULT: Combined score of tool selection and parameter correctness
    - ERROR_HANDLING: Penalizes unwanted tool calls (binary)
    - EXECUTION: Strict order check for tool call sequences (binary)
    - NOISE: Penalizes extraneous tools or parameters (binary)
    - ADAPTABILITY: Exact match on final tool call (binary)
    - AMBIGUITY: Handling of ambiguous queries with partial credit (0/0.5/1.0)
    """

    DECISION = "Decision"
    TOOL_SELECTION = "Tool selection"
    PARAMS = "Params"
    RESULT = "Result"
    ERROR_HANDLING = "Error Handling"
    EXECUTION = "Execution"
    NOISE = "Noise"
    ADAPTABILITY = "Adaptability"
    AMBIGUITY = "Ambiguity"

    @property
    def column_name(self) -> str:
        """Get the DataFrame column name for this metric (lowercase, snake_case).
        
        Returns:
            Normalized metric name suitable for use as a DataFrame column
        """
        return self.value.lower().replace(" ", "_")
