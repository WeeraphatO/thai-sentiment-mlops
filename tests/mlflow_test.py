from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.load_config import load_training_config
from src.mlflow.mlflow_registry import ModelRegistry

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

config = load_training_config()

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

experiment = client.get_experiment_by_name(
    config["mlflow"]["experiment_name"]
)

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

print("\nArtifacts:")
artifacts = client.list_artifacts(best_run_id)

for artifact in artifacts:
    print("-", artifact.path)

print("\nSearching logged model for best run...")

logged_models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    output_format="list",
)

best_model = None

for model in logged_models:
    source_run_id = getattr(model, "source_run_id", None)

    if source_run_id == best_run_id:
        best_model = model
        break

if best_model is None:
    print("No logged model found for best run")
else:
    print("\nBest Run Model:")
    print(best_model)

    print("\nModel Attributes:")
    try:
        for k, v in vars(best_model).items():
            print(f"{k}: {v}")
    except Exception:
        pass

registry = ModelRegistry(
    tracking_uri=tracking_uri,
    model_name=config["mlflow"]["model_name"],
)

version = registry.register_best_run(
    experiment_name=config["mlflow"]["experiment_name"],
)

print(f"\nRegistered version: {version}")

registry.transition_to_staging(version)
registry.promote_to_production(version)