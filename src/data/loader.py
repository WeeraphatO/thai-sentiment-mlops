from datasets import load_dataset, DatasetDict
import pandas as pd
from pathlib import Path
import yaml
from src.utils.load_config import load_data_config

def load_wisesight_dataset() -> DatasetDict:
    return load_dataset("pythainlp/wisesight_sentiment")

def save_data(
    ds: DatasetDict,
) -> None:

    config = load_data_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    split_mapping = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }
    for split_name, file_name in split_mapping.items():

        df = ds[split_name].to_pandas()

        df[
            [
                "texts",
                "category"
            ]
        ].to_csv(
            raw_dir / file_name,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Saved Raw {file_name} ({len(df):,} rows)")