from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_data_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "data.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_training_config() -> dict[str, Any]:

    config_path = (
        PROJECT_ROOT
        / "config"
        / "train.yaml"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)