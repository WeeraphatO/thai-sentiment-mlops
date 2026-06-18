from pathlib import Path
import os
import sys

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlflow.mlflow_registry import ModelRegistry
from src.utils.load_config import load_training_config


# Configuration
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

config = load_training_config()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()


def get_best_run(experiment_name: str) -> tuple[str, str]:
    experiment = client.get_experiment_by_name(experiment_name)

    print("Experiment:")
    print(experiment)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_f1_macro DESC"],
    )

    print("\nTop runs:")
    print(runs[["run_id"]].head())

    best_run_id = runs.iloc[0]["run_id"]

    print(f"\nBest run: {best_run_id}")

    return experiment.experiment_id, best_run_id


def show_artifacts(run_id: str) -> None:
    print("\nArtifacts:")

    for artifact in client.list_artifacts(run_id):
        print("-", artifact.path)


def show_logged_model(experiment_id: str, run_id: str) -> None:
    print("\nSearching logged model for best run...")

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        output_format="list",
    )

    model = next(
        (
            m
            for m in logged_models
            if getattr(m, "source_run_id", None) == run_id
        ),
        None,
    )

    if model is None:
        print("No logged model found for best run")
        return

    print("\nBest Run Model:")
    print(model)

    print("\nModel Attributes:")
    try:
        for key, value in vars(model).items():
            print(f"{key}: {value}")
    except Exception:
        pass


def register_and_promote() -> int:
    registry = ModelRegistry(
        tracking_uri=tracking_uri,
        model_name=config["mlflow"]["model_name"],
    )

    version = registry.register_best_run(
        experiment_name=config["mlflow"]["experiment_name"],
    )

    registry.transition_to_staging(version)
    registry.promote_to_production(version)

    return version


def main() -> None:
    experiment_id, best_run_id = get_best_run(
        config["mlflow"]["experiment_name"]
    )

    show_artifacts(best_run_id)
    show_logged_model(experiment_id, best_run_id)

    version = register_and_promote()
    print(f"\nRegistered version: {version}")


if __name__ == "__main__":
    main()