import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, tracking_uri: str, model_name: str):
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.model_name = model_name
        mlflow.set_tracking_uri(tracking_uri)

    def register_best_run(self, experiment_name: str, metric: str = "f1_macro") -> str:
        """Find the best run in an experiment and register it."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        if not runs:
            raise ValueError(f"No runs found in experiment: {experiment_name}")

        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"

        result = mlflow.register_model(model_uri=model_uri, name=self.model_name)
        print(f"Registered model: {self.model_name} v{result.version} "
              f"(run_id: {best_run.info.run_id}, {metric}: "
              f"{best_run.data.metrics[metric]:.4f})")
        return result.version

    def transition_to_staging(self, version: str):
        self.client.transition_model_version_stage(
            name=self.model_name, version=version, stage="Staging"
        )
        print(f"Model v{version} → Staging")

    def promote_to_production(self, version: str):
        """Promote version to Production, archive the previous Production model."""
        # Archive existing Production version
        prod_versions = self.client.get_latest_versions(
            self.model_name, stages=["Production"]
        )
        for v in prod_versions:
            self.client.transition_model_version_stage(
                name=self.model_name, version=v.version, stage="Archived"
            )
            print(f"Archived previous Production model v{v.version}")

        self.client.transition_model_version_stage(
            name=self.model_name, version=version, stage="Production"
        )
        print(f"Model v{version} → Production ✓")

    def get_production_model_uri(self) -> str:
        return f"models:/{self.model_name}/Production"

    def get_latest_production_version(self) -> str:
        versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
        if not versions:
            raise ValueError(f"No Production model found for: {self.model_name}")
        return versions[0].version