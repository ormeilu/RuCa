"""Configuration loader for model settings and parameters.

This module provides utilities to load model configurations from YAML files,
resolve parameters from config or environment variables, and manage API credentials.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file.

    Attempts to locate the configuration file in the following order (by priority):
    1. If an absolute path is provided - use it directly
    2. Relative to the current working directory
    3. Next to the ruca.utils module
    4. In the <project_root>/configs/config.yaml directory
    5. In the project root

    If the configuration file is not found, returns a default structure with empty models.
    
    Args:
        config_path: Path to the configuration file (default: "config.yaml")
        
    Returns:
        Dictionary with configuration data, or {"models": {}} if file not found
    """
    candidates: list[Path] = []
    provided: Path = Path(config_path)

    # If absolute path is provided, use it directly
    if provided.is_absolute():
        candidates.append(provided)
    else:
        # Search relative to current working directory
        candidates.append(Path.cwd() / provided)

        # Search next to the module
        module_dir: Path = Path(__file__).parent
        candidates.append(module_dir / provided)

        # Try to find in project root configs/ and root directories
        try:
            project_root: Path = Path(__file__).resolve().parents[3]
            candidates.append(project_root / "configs" / provided)
            candidates.append(project_root / provided)
        except IndexError:
            # Handle non-standard directory structures
            pass

    # Find the first existing candidate
    config_file: Path | None = None
    for cand in candidates:
        if cand.exists():
            config_file = cand
            break

    if config_file is None:
        return {"models": {}}

    with open(config_file, encoding="utf-8") as f:
        config: dict[str, Any] | None = yaml.safe_load(f)

    return config or {"models": {}}


def get_model_config(model_name: str, config_path: str = "config.yaml") -> dict[str, Any] | None:
    """Get configuration for a specific model by name.
    
    Args:
        model_name: Name of the model as defined in the config
        config_path: Path to the configuration file
        
    Returns:
        Model configuration dictionary, or None if model not found
    """
    config = load_config(config_path)
    models = config.get("models", {})
    return models.get(model_name)


def resolve_model_params(model_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve model parameters from configuration.

    Extracts model parameters from the config dictionary, falling back to
    environment variables for API credentials if not specified in config.
    
    Args:
        model_config: Model configuration dictionary from YAML
        
    Returns:
        Dictionary with resolved parameters including:
            - model_name: Name of the model
            - temperature: Sampling temperature
            - top_p: Top-p sampling parameter
            - top_k: Top-k sampling parameter
            - seed: Random seed for reproducibility
            - reasoning: Whether to enable extended reasoning
            - api_key: API key (from config or OPENAI_API_KEY env var)
            - base_url: API base URL (from config or OPENAI_BASE_URL env var)
    """
    params: dict[str, Any] = {
        "model_name": model_config.get("model_name", "unknown"),
        "temperature": model_config.get("temperature", 0.5),
        "top_p": model_config.get("top_p"),
        "top_k": model_config.get("top_k"),
        "seed": model_config.get("seed"),
        "reasoning": model_config.get("reasoning", False),
    }

    # Resolve API Key from config or environment
    api_key: str = model_config.get("api_key", "").strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or ""
    params["api_key"] = api_key

    # Resolve Base URL from config or environment
    base_url: str = model_config.get("base_url", "").strip()
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL") or ""
    params["base_url"] = base_url

    return params


def get_all_models(config_path: str = "config.yaml") -> dict[str, dict[str, Any]]:
    """Get configuration for all models from config.yaml.
    
    Loads the configuration file and resolves parameters for all defined models.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary mapping model names to their resolved parameters
    """
    config = load_config(config_path)
    models = config.get("models", {})

    resolved = {}
    for model_name, model_config in models.items():
        resolved[model_name] = resolve_model_params(model_config)

    return resolved
