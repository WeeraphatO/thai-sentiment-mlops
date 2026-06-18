from pathlib import Path
import re

import pandas as pd
from datasets import DatasetDict
from pythainlp.util import normalize as thai_normalize

from src.data.loader import load_wisesight_dataset

LABEL_MAP = {
    0: "pos",
    1: "neu",
    2: "neg",
    3: "q",
}


def clean_text(text: str) -> str:
    """
    Thai text cleaning pipeline.

    Steps
    -----
    1. Normalize Thai text
    2. Remove URLs
    3. Remove @mentions
    4. Remove masked phone numbers
    5. Collapse whitespace
    """
    text = str(text)

    # normalize Thai characters
    text = thai_normalize(text)

    # remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # remove mentions
    text = re.sub(r"@\S+", " ", text)

    # remove phone numbers
    text = re.sub(
        r"\d{3}-\d{3}-\d{4}|\d{2}-\d{4}-\d{4}|\d{1}-\d{4}-\d{4}",
        " ",
        text,
    )

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning and remove empty rows.
    """
    df = df.copy()
    df = df.rename(
        columns={
            "texts": "text",
            "category": "label",
        }
    )
    df["label_name"] = df["label"].map(LABEL_MAP)
    df["text_clean"] = df["text"].apply(clean_text)

    before = len(df)
    df = df.drop_duplicates(subset="text_clean")
    df = df.dropna(subset=["text", "label"])
    df = (
        df[df["text_clean"].str.strip().str.len() > 0]
        .reset_index(drop=True)
    )
    dropped = before - len(df)

    print(
        f"Removed {dropped:,} rows "
        f"({dropped / before * 100:.2f}%)"
    )

    return df


def save_processed_splits(
    ds: DatasetDict,
    output_dir: str = "data/processed",
) -> None:
    """
    Save processed data into csv files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_mapping = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }

    for split_name, file_name in split_mapping.items():
        df = ds[split_name].to_pandas()
        df = preprocess_df(df)
        df[
            [
                "text_clean",
                "label",
                "label_name",
            ]
        ].to_csv(
            output_path / file_name,
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"Saved {file_name} "
            f"({len(df):,} rows)"
        )


def main():
    ds = load_wisesight_dataset()
    save_processed_splits(ds)


if __name__ == "__main__":
    main()