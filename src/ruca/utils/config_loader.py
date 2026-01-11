import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Загружает конфигурацию из YAML файла.

    Пытается найти файл в нескольких местах (в порядке приоритета):
    1. если указан абсолютный путь — используем его;
    2. относительный к текущей рабочей директории;
    3. рядом с модулем `ruca.utils` (как раньше);
    4. в каталоге `<project_root>/configs/config.yaml`;
    5. в корне проекта.

    Если файл не найден — возвращаем пустую структуру с ключом `models`.
    """
    candidates: list[Path] = []
    provided = Path(config_path)

    # Если указан абсолютный путь, используем как есть
    if provided.is_absolute():
        candidates.append(provided)
    else:
        # Как указано относительно текущей директории
        candidates.append(Path.cwd() / provided)

        # Рядом с модулем
        module_dir = Path(__file__).parent
        candidates.append(module_dir / provided)

        # Попробуем найти в корне проекта в папке configs/ и в корне
        try:
            project_root = Path(__file__).resolve().parents[3]
            candidates.append(project_root / "configs" / provided)
            candidates.append(project_root / provided)
        except IndexError:
            # на всякий случай, если структура директорий нестандартная
            pass

    # Найдём первый существующий путь
    config_file: Path | None = None
    for cand in candidates:
        if cand.exists():
            config_file = cand
            break

    if config_file is None:
        return {"models": {}}

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {"models": {}}


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
