from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import WisesightDataset
from src.models.classifier import LABEL_NAMES, load_model, load_tokenizer
from src.training.trainer import SentimentTrainer
from src.utils.load_config import load_training_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)


def load_splits(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}

    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing data file: {path}\n"
                f"Run src/data/preprocessor.py first to generate the splits."
            )
        df = pd.read_csv(path)

        for col in ("text_clean", "label"):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {path}. "
                    f"Available: {list(df.columns)}"
                )

        splits[split] = df
        logger.info("Loaded %-5s  %d rows", split, len(df))

    return splits["train"], splits["val"], splits["test"]

def main(config_path: str, data_dir: str) -> None:
    load_dotenv(".env")

    config = load_training_config(config_path)
    set_seed(config["seed"]["random_state"])

    train_df, val_df, test_df = load_splits(data_dir)

    tokenizer = load_tokenizer()
    max_length = config["model"]["max_length"]

    train_dataset = WisesightDataset(train_df, tokenizer, max_length)
    val_dataset   = WisesightDataset(val_df,   tokenizer, max_length)
    test_dataset  = WisesightDataset(test_df,  tokenizer, max_length)

    logger.info(
        "Datasets  train=%d  val=%d  test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    model = load_model()

    trainer = SentimentTrainer(config)
    run_id, results = trainer.train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        test_df=test_df,
    )

    print()
    print("=" * 52)
    print("Training complete")
    print(f"MLflow run ID : {run_id}")
    print("-" * 52)
    for metric, value in results.items():
        print(f"  {metric:<24} {value:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune PhayaThaiBERT for Thai sentiment classification."
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to training config YAML  (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory with train.csv / val.csv / test.csv  "
             "(default: data/processed)",
    )
    args = parser.parse_args()
    main(config_path=args.config, data_dir=args.data_dir)