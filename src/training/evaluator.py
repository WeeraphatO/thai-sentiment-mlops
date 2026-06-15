from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import PreTrainedTokenizerBase, Trainer, pipeline

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, preds, average="macro", zero_division=0)
        ),
    }


def evaluate_on_test(
    trainer: Trainer,
    test_dataset,
    test_labels: list[int],
    label_names: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    logger.info("Running test-set inference (%d samples)...", len(test_labels))

    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=-1)

    metrics: dict[str, float] = {
        "test_accuracy": float(accuracy_score(test_labels, preds)),
        "test_f1_weighted": float(
            f1_score(test_labels, preds, average="weighted", zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(test_labels, preds, average="macro", zero_division=0)
        ),
    }

    report = classification_report(
        test_labels,
        preds,
        labels=[0, 1, 2, 3],
        target_names=label_names,
        zero_division=0,
    )
    logger.info("\n%s", report)

    return metrics, preds


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    plt.tight_layout()

    return fig


def log_artifacts_to_mlflow(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    test_labels: list[int],
    test_preds: np.ndarray,
    label_names: list[str],
    output_dir: str,
) -> None:

    fig = plot_confusion_matrix(test_labels, test_preds, label_names)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    report = classification_report(
        test_labels,
        test_preds,
        target_names=label_names,
        zero_division=0,
    )

    mlflow.log_text(report, "classification_report.txt")

    clf_pipeline = pipeline(
        task="text-classification",
        model=trainer.model,
        tokenizer=tokenizer,
    )

    mlflow.transformers.log_model(
        transformers_model=clf_pipeline,
        name="model",
    )

    logger.info("Logged MLflow model + artifacts")