from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import DatasetDict
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

from src.data.loader import load_wisesight_dataset, save_data
from src.data.preprocessor import save_processed_splits


# ── Tasks ──────────────────────────────────────────────────────────────────────


@task(
    name="download-wisesight-dataset",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=15,
)
def download_dataset() -> DatasetDict:
    logger = get_run_logger()
    logger.info("Downloading pythainlp/wisesight_sentiment from HuggingFace...")

    ds = load_wisesight_dataset()

    logger.info(
        "Download complete  train=%d  val=%d  test=%d",
        len(ds["train"]),
        len(ds["validation"]),
        len(ds["test"]),
    )
    return ds


@task(
    name="save-raw-splits",
    retries=1,
)
def save_raw_splits(ds: DatasetDict) -> None:
    logger = get_run_logger()
    logger.info("Saving raw splits to data/raw/...")
    save_data(ds)
    logger.info("Raw splits saved.")


@task(
    name="preprocess-splits",
    retries=1,
)
def preprocess_splits(
    ds: DatasetDict,
    output_dir: str = "data/processed",
) -> None:
    logger = get_run_logger()
    logger.info("Preprocessing splits → %s", output_dir)
    save_processed_splits(ds, output_dir=output_dir)
    logger.info("Preprocessing complete.")


# ── Flow ───────────────────────────────────────────────────────────────────────


@flow(
    name="thai-sentiment-data-pipeline",
    description="Download Wisesight dataset and produce preprocessed CSV splits.",
    log_prints=True,
)
def data_pipeline(
    output_dir: str = "data/processed",
    save_raw: bool = True,
) -> None:
    ds = download_dataset()

    if save_raw:
        save_raw_splits(ds)

    preprocess_splits(ds, output_dir=output_dir)


if __name__ == "__main__":
    data_pipeline()