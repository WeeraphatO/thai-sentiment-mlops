from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlflow.mlflow_registry import ModelRegistry
from src.utils.load_config import load_training_config


def main(auto: bool, version: str | None, config_path: str) -> None:
    config = load_training_config(config_path)

    # Server URI stays in the environment — it's infrastructure, not model config
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Both the registry name and experiment name come from train.yaml
    model_name      = config["mlflow"]["model_name"]
    experiment_name = config["mlflow"]["experiment_name"]

    registry = ModelRegistry(
        tracking_uri=tracking_uri,
        model_name=model_name,
    )

    if auto:
        new_version = registry.register_best_run(experiment_name)
        registry.transition_to_staging(new_version)

        print("Validation gate: checking val F1 > 0.70...")
        registry.promote_to_production(new_version)

    elif version:
        registry.promote_to_production(version)

    else:
        print("Specify --auto or --version <N>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register and promote the best MLflow run to Production."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Register best run and auto-promote to Production.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Manually promote a specific model version number.",
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to training config YAML  (default: configs/train.yaml)",
    )
    args = parser.parse_args()
    main(args.auto, args.version, args.config)