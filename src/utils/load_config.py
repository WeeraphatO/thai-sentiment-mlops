from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml_config(filename: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / filename

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data_config() -> dict[str, Any]:
    return load_yaml_config("data.yaml")


def load_training_config() -> dict[str, Any]:
    return load_yaml_config("train.yaml")