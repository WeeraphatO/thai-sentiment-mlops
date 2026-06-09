from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from src.utils.load_config import load_training_config


config = load_training_config()

MODEL_NAME = config["model"]["name"]
NUM_LABELS: int = config["model"]["num_labels"]

LABEL_NAMES: list[str] = ["neg", "neu", "pos", "q"]
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL: dict[int, str] = {i: l for i, l in enumerate(LABEL_NAMES)}


def load_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )


def load_model() -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )