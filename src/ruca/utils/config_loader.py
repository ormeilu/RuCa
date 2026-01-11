import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Загружает конфигурацию из YAML файла."""
    # Если это относительный путь по умолчанию, ищем рядом с модулем
    if config_path == "config.yaml":
        module_dir = Path(__file__).parent
        config_path = str(module_dir / config_path)

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or {"models": {}}
    except FileNotFoundError:
        return {"models": {}}


def get_model_config(model_name: str, config_path: str = "config.yaml") -> dict[str, Any] | None:
    """Получает конфигурацию конкретной модели."""
    config = load_config(config_path)
    models = config.get("models", {})
    return models.get(model_name)


def resolve_model_params(model_config: dict[str, Any]) -> dict[str, Any]:
    """
    Разрешает параметры модели из конфига.
    Если api_key или base_url пусты, ищет в окружении.
    """
    params = {
        "model_name": model_config.get("model_name", "unknown"),
        "temperature": model_config.get("temperature", 0.5),
        "top_p": model_config.get("top_p"),
        "top_k": model_config.get("top_k"),
        "seed": model_config.get("seed"),
        "reasoning": model_config.get("reasoning", False),
    }

    # API Key
    api_key = model_config.get("api_key", "").strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    params["api_key"] = api_key

    # Base URL
    base_url = model_config.get("base_url", "").strip()
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL")
    params["base_url"] = base_url

    return params


def get_all_models(config_path: str = "config.yaml") -> dict[str, dict[str, Any]]:
    """Получает конфигурацию всех моделей из config.yaml."""
    config = load_config(config_path)
    models = config.get("models", {})

    resolved = {}
    for model_name, model_config in models.items():
        resolved[model_name] = resolve_model_params(model_config)

    return resolved
