from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.preprocessor import clean_text

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("artifacts") / "serving_cache"


class SentimentPredictor:
    def __init__(self) -> None:
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.id2label: dict[int, str] = {}
        self.model_name: str = ""
        self.model_version: str = ""
        self.max_length: int = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_ready: bool = False

    def load_from_registry(
        self,
        model_name: str,
        stage: str = "Production",
        tracking_uri: str | None = None,
    ) -> None:
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        client = MlflowClient(tracking_uri=uri)

        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise RuntimeError(
                f"No model found in stage '{stage}' for '{model_name}'. "
                "Run scripts/promote_model.py --auto first."
            )

        version = versions[0]
        run_id = version.run_id

        self.model_name = model_name
        self.model_version = version.version

        logger.info(
            "Loading %s v%s (run_id=%s) from %s",
            model_name,
            self.model_version,
            run_id,
            stage,
        )

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        model_local = client.download_artifacts(
            run_id,
            "model",
            str(_CACHE_DIR),
        )

        tokenizer_local = client.download_artifacts(
            run_id,
            "tokenizer",
            str(_CACHE_DIR),
        )

        run = client.get_run(run_id)
        self.max_length = int(run.data.params.get("max_length", 256))

        self.model = AutoModelForSequenceClassification.from_pretrained(model_local)
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_local)

        self.id2label = {
            int(k): v
            for k, v in self.model.config.id2label.items()
        }

        self._is_ready = True

        logger.info(
            "Model ready  version=%s  device=%s  max_length=%d",
            self.model_version,
            self.device,
            self.max_length,
        )

    def predict(self, text: str) -> dict:
        self._check_ready()
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        self._check_ready()

        cleaned = [clean_text(t) for t in texts]

        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoding = {
            k: v.to(self.device)
            for k, v in encoding.items()
        }

        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []

        for i, original_text in enumerate(texts):
            label_id = int(probs[i].argmax())

            results.append({
                "text": original_text,
                "label": self.id2label[label_id],
                "label_id": label_id,
                "confidence": float(probs[i][label_id]),
                "probabilities": {
                    self.id2label[j]: float(p)
                    for j, p in enumerate(probs[i])
                },
                "model_version": self.model_version,
            })

        return results

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _check_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError(
                "Predictor is not ready. Call load_from_registry() first."
            )