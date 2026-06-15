from __future__ import annotations

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


class ModelPredictor:
    def __init__(self, model_name: str, tracking_uri: str):
        self.model_name = model_name
        self.model_version: str | None = None
        self.is_ready = False
        self._model = None

        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def _find_production_version(self) -> str:
        versions = self._client.search_model_versions(
            f"name='{self.model_name}'"
        )

        for version in versions:
            if version.current_stage == "Production" or version.tags.get("stage") == "production":
                return version.version

        raise RuntimeError(
            f"No production version found for '{self.model_name}'"
        )

    def load(self) -> None:
        version = self._find_production_version()

        model_uri = f"models:/{self.model_name}/{version}"

        self._model = mlflow.pyfunc.load_model(model_uri)

        self.model_version = version
        self.is_ready = True

        print("Model URI:", model_uri)
        print("Metadata:", self._model.metadata)

        print(f"Loaded {self.model_name} v{version} (production)")

    def predict(self, text: str) -> dict:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")

        df = pd.DataFrame({"text": [text]})

        result = self._model.predict(df)

        # MLflow transformers usually returns list[dict]
        if isinstance(result, list):
            result = result[0]

        return {
            "label": result["label"],
            "confidence": float(result.get("score", result.get("confidence", 0.0))),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")

        df = pd.DataFrame({"text": texts})

        results = self._model.predict(df)

        outputs = []
        for r in results:
            outputs.append({
                "label": r["label"],
                "confidence": float(r.get("score", r.get("confidence", 0.0))),
            })

        return outputs