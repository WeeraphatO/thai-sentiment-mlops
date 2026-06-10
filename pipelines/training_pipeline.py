from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd
import torch
from prefect import flow, get_run_logger, task

from src.data.dataset import WisesightDataset
from src.models.classifier import load_model, load_tokenizer
from src.mlflow.mlflow_registry import ModelRegistry
from src.training.trainer import SentimentTrainer
from src.utils.load_config import load_training_config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(
    name="validate-processed-data",
    retries=1,
)
def validate_processed_data(data_dir: str) -> None:
    """
    Confirm that train.csv, val.csv, and test.csv exist and contain
    the required columns (text_clean, label).

    Raises FileNotFoundError if any split is missing, so the flow fails
    early with a clear message before wasting time loading the model.
    """
    logger = get_run_logger()
    required_cols = {"text_clean", "label"}

    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. "
                "Run pipelines/data_pipeline.py first."
            )

        df = pd.read_csv(path, nrows=5)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        n_rows = sum(1 for _ in open(path)) - 1   # fast row count
        logger.info("%-5s  %d rows  %s", split, n_rows, path)


@task(
    name="run-training",
    retries=1,
    retry_delay_seconds=30,
)
def run_training(
    config_path: str,
    data_dir: str,
) -> tuple[str, dict[str, float]]:
    """
    Load data + model, run SentimentTrainer.train(), return MLflow run_id
    and test metrics.
    """
    logger = get_run_logger()

    config = load_training_config()
    _set_seed(config["seed"]["random_state"])

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    val_df   = pd.read_csv(Path(data_dir) / "val.csv")
    test_df  = pd.read_csv(Path(data_dir) / "test.csv")

    logger.info(
        "Data loaded  train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    tokenizer  = load_tokenizer()
    max_length = config["model"]["max_length"]

    train_dataset = WisesightDataset(train_df, tokenizer, max_length)
    val_dataset   = WisesightDataset(val_df,   tokenizer, max_length)
    test_dataset  = WisesightDataset(test_df,  tokenizer, max_length)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = load_model()
    trainer = SentimentTrainer(config)

    logger.info("Starting training...")
    run_id, results = trainer.train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        test_df=test_df,
    )

    logger.info("Training complete  run_id=%s", run_id)
    for metric, value in results.items():
        logger.info("  %-24s %.4f", metric, value)

    return run_id, results


@task(
    name="register-and-promote",
    retries=2,
    retry_delay_seconds=10,
)
def register_and_promote(config_path: str) -> str:
    """
    Register the best run from the experiment and promote it to Production.
    Returns the new Production model version number.
    """
    logger = get_run_logger()

    config = load_training_config()

    registry = ModelRegistry(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        model_name=config["mlflow"]["model_name"],
    )

    version = registry.register_best_run(
        experiment_name=config["mlflow"]["experiment_name"],
    )
    registry.transition_to_staging(version)
    registry.promote_to_production(version)

    logger.info(
        "Model '%s' v%s is now in Production.",
        config["mlflow"]["model_name"],
        version,
    )
    return version


# ── Flow ───────────────────────────────────────────────────────────────────────


@flow(
    name="thai-sentiment-training-pipeline",
    description=(
        "Fine-tune PhayaThaiBERT on Wisesight Sentiment, log to MLflow, "
        "and promote the best model to Production."
    ),
    log_prints=True,
)
def training_pipeline(
    config_path: str = "configs/train.yaml",
    data_dir: str = "data/processed",
) -> None:
    validate_processed_data(data_dir)

    run_id, results = run_training(config_path, data_dir)

    version = register_and_promote(config_path)

    print()
    print("=" * 52)
    print("Pipeline complete")
    print(f"  MLflow run ID    : {run_id}")
    print(f"  Registry version : {version}  (Production)")
    print("-" * 52)
    for metric, value in results.items():
        print(f"  {metric:<24} {value:.4f}")
    print("=" * 52)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Thai sentiment training pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to training config YAML  (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory with train/val/test CSVs  (default: data/processed)",
    )
    args = parser.parse_args()
    training_pipeline(config_path=args.config, data_dir=args.data_dir)