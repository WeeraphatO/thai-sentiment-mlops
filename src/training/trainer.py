from __future__ import annotations

import logging
import os

import mlflow
import torch
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.training.evaluator import (
    compute_metrics,
    evaluate_on_test,
    log_artifacts_to_mlflow,
)

logger = logging.getLogger(__name__)


class SentimentTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._validate_config()

    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset,
        val_dataset,
        test_dataset,
        test_df,
    ) -> tuple[str, dict[str, float]]:
        from src.models.classifier import LABEL_NAMES

        self._setup_mlflow()
        training_args = self._build_training_args()

        hf_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config["early_stopping"]["patience"]
                )
            ],
        )

        with mlflow.start_run(run_name="phayathaibert-finetune") as run:
            mlflow.log_params(
                self._build_mlflow_params(
                    n_train=len(train_dataset),
                    n_val=len(val_dataset),
                    n_test=len(test_dataset),
                )
            )
            mlflow.set_tags({
                "method":   "PhayaThaiBERT fine-tune",
                "language": "thai",
                "dataset":  "wisesight_sentiment",
            })

            logger.info("Starting training...")
            hf_trainer.train()

            self._log_step_metrics(hf_trainer)

            val_results = hf_trainer.evaluate(val_dataset)
            mlflow.log_metrics({
                "val_accuracy":    val_results["eval_accuracy"],
                "val_f1_weighted": val_results["eval_f1_weighted"],
                "val_f1_macro":    val_results["eval_f1_macro"],
                "val_loss":        val_results["eval_loss"],
            })
            logger.info(
                "Val  accuracy=%.4f  f1_weighted=%.4f  f1_macro=%.4f",
                val_results["eval_accuracy"],
                val_results["eval_f1_weighted"],
                val_results["eval_f1_macro"],
            )

            test_labels = test_df["label"].tolist()
            test_metrics, test_preds = evaluate_on_test(
                hf_trainer, test_dataset, test_labels, LABEL_NAMES
            )
            mlflow.log_metrics(test_metrics)
            logger.info(
                "Test accuracy=%.4f  f1_weighted=%.4f  f1_macro=%.4f",
                test_metrics["test_accuracy"],
                test_metrics["test_f1_weighted"],
                test_metrics["test_f1_macro"],
            )

            log_artifacts_to_mlflow(
                trainer=hf_trainer,
                tokenizer=tokenizer,
                test_labels=test_labels,
                test_preds=test_preds,
                label_names=LABEL_NAMES,
                best_model_dir=self.config["paths"]["best_model_dir"],
                output_dir=self.config["paths"]["output_dir"],
            )

            run_id = run.info.run_id
            logger.info("MLflow run complete: %s", run_id)

        return run_id, test_metrics

    def _build_training_args(self) -> TrainingArguments:
        t = self.config["training"]
        p = self.config["paths"]
        m = self.config["metrics"]
        s = self.config["seed"]

        return TrainingArguments(
            output_dir=p["output_dir"],

            num_train_epochs=t["epochs"],
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=t.get("eval_batch_size", t["batch_size"] * 2),

            learning_rate=t["learning_rate"],
            warmup_ratio=t["warmup_ratio"],
            weight_decay=t["weight_decay"],

            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model=m["best_metric"],
            greater_is_better=m["greater_is_better"],

            logging_strategy="steps",
            logging_steps=50,
            report_to="none",        

            fp16=torch.cuda.is_available(),

            seed=s["random_state"],
        )

    def _setup_mlflow(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    def _build_mlflow_params(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> dict:
        t = self.config["training"]
        mo = self.config["model"]
        s = self.config["seed"]
        return {
            "model_name":    mo["name"],
            "max_length":    mo["max_length"],
            "learning_rate": t["learning_rate"],
            "batch_size":    t["batch_size"],
            "epochs":        t["epochs"],
            "warmup_ratio":  t["warmup_ratio"],
            "weight_decay":  t["weight_decay"],
            "train_samples": n_train,
            "val_samples":   n_val,
            "test_samples":  n_test,
            "fp16":          torch.cuda.is_available(),
            "seed":          s["random_state"],
        }

    @staticmethod
    def _log_step_metrics(hf_trainer: Trainer) -> None:

        loggable = {
            "eval_accuracy",
            "eval_f1_weighted",
            "eval_f1_macro",
            "eval_loss",
            "loss",
        }
        for entry in hf_trainer.state.log_history:
            if entry.get("epoch") is None:
                continue
            to_log = {k: v for k, v in entry.items() if k in loggable}
            if to_log:
                mlflow.log_metrics(to_log, step=int(entry.get("step", 0)))

    def _validate_config(self) -> None:
        required: dict[str, list[str]] = {
            "model":          ["name", "num_labels", "max_length"],
            "training":       ["batch_size", "learning_rate", "epochs",
                               "warmup_ratio", "weight_decay"],
            "early_stopping": ["patience"],
            "paths":          ["output_dir", "best_model_dir"],
            "mlflow":         ["experiment_name"],
            "metrics":        ["best_metric", "greater_is_better"],
            "seed":           ["random_state"],
        }
        for section, keys in required.items():
            if section not in self.config:
                raise KeyError(
                    f"Missing config section: '{section}'. "
                    f"Check configs/train.yaml."
                )
            for key in keys:
                if key not in self.config[section]:
                    raise KeyError(
                        f"Missing config key: '{section}.{key}'. "
                        f"Check configs/train.yaml."
                    )