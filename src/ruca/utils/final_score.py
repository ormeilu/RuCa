"""Calculate final weighted benchmark score with penalty system.

This module implements the final score calculation for benchmark evaluation,
combining individual metric scores with a sophisticated penalty system that
rewards consistency across metrics.
"""

import pandas as pd

# Penalty configuration constants
PENALTY_THRESHOLD: float = 0.4  # Threshold below which penalties apply
ALPHA_CORE: float = 1.5  # Power factor for core metric penalties (higher = steeper)
BETA_SPEC: float = 1.0  # Power factor for specific metric penalties

MIN_FLOOR_CORE: float = 0.4  # Minimum score floor for core metrics (won't go below 40%)
MIN_FLOOR_SPEC: float = 0.3  # Minimum score floor for specific metrics (won't go below 30%)

# Core metrics that measure tool decision and parameter accuracy
CORE_METRICS: list[str] = [
    "decision",
    "tool_selection",
    "params",
]

# Specific metrics that handle edge cases and special scenarios
SPECIFIC_METRICS: list[str] = [
    "ambiguity",
    "noise",
    "adaptability",
    "error_handling",
    "execution",
]


def _smooth_penalty(min_value: float, threshold: float, power: float, min_floor: float) -> float:
    """Calculate smooth penalty multiplier for scores below threshold.

    Implements a smooth penalty function that:
    - Returns 1.0 if score >= threshold (no penalty)
    - Returns a value >= min_floor if score < threshold (scaled by power)
    - Uses exponential scaling to provide smooth transitions

    Args:
        min_value: The minimum value in the metric group
        threshold: Score threshold above which no penalty applies
        power: Exponent for scaling (higher = steeper penalty curve)
        min_floor: Minimum possible penalty value (prevents score collapse)

    Returns:
        Penalty multiplier between min_floor and 1.0
    """
    if min_value >= threshold:
        return 1.0
    ratio: float = max(min_value / threshold, 0.0)
    return min_floor + (1 - min_floor) * (ratio**power)


def calculate_final_score(df: pd.DataFrame) -> float:
    """Calculate final weighted benchmark score with penalty system.

    Scoring system:
    1. Calculates linear score based on metric weights
    2. Applies smooth penalties if any core metrics are below threshold
    3. Applies smooth penalties if any specific metrics are below threshold
    4. Returns percentage (0-100) averaged across all examples

    Uses different weighting schemes:
    - 4-metric scheme: For basic tool selection and parameters (no special metrics)
    - 5-metric scheme: For cases with one specific metric enabled

    Args:
        df: DataFrame with one row per benchmark example and metric score columns

    Returns:
        Final weighted score as percentage (0-100)
    """
    # Weight definitions for different metric combinations
    weights_4: dict[str, float] = {
        "decision": 0.36,
        "tool_selection": 0.36,
        "params": 0.28,
    }

    weights_5_base: dict[str, float] = {
        "decision": 0.29,
        "tool_selection": 0.29,
        "params": 0.22,
    }

    # Specific metrics that add an additional evaluation dimension
    specific_metrics_weighted: list[str] = [
        "ambiguity",
        "noise",
        "adaptability",
        "error_handling",
        "execution",
    ]
    specific_weight: float = 0.20

    scores: list[float] = []

    # Process each row (query example)
    for _, row in df.iterrows():
        # Determine if any specific metric is enabled
        specific_found: str | None = next(
            (m for m in specific_metrics_weighted if m in row.index and pd.notna(row[m])),
            None,
        )

        # Calculate linear score based on metric scheme
        if specific_found:
            # 5-metric scheme: base weights + specific metric
            linear_score: float = sum(
                weight * row[metric]
                for metric, weight in weights_5_base.items()
                if metric in row.index and pd.notna(row[metric])
            )
            linear_score += specific_weight * row[specific_found]
        else:
            # 4-metric scheme: simple weighting
            linear_score: float = sum(
                weight * row[metric]
                for metric, weight in weights_4.items()
                if metric in row.index and pd.notna(row[metric])
            )

        # Extract metric values for penalty calculation
        core_values: list[float] = [row[m] for m in CORE_METRICS if m in row.index and pd.notna(row[m])]
        spec_values: list[float] = [row[m] for m in SPECIFIC_METRICS if m in row.index and pd.notna(row[m])]

        # Calculate penalty multipliers
        core_penalty: float = (
            _smooth_penalty(min(core_values), PENALTY_THRESHOLD, ALPHA_CORE, MIN_FLOOR_CORE) if core_values else 1.0
        )
        spec_penalty: float = (
            _smooth_penalty(min(spec_values), PENALTY_THRESHOLD, BETA_SPEC, MIN_FLOOR_SPEC) if spec_values else 1.0
        )

        # Apply penalties and convert to percentage
        final_score_value: float = linear_score * core_penalty * spec_penalty * 100
        scores.append(final_score_value)

    # Return average score across all examples
    return sum(scores) / len(scores) if scores else 0.0
