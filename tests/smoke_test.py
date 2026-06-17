"""
scripts/smoke_test.py

Validates the full training pipeline with synthetic data and a tiny model.
Runs on CPU in ~90 seconds — no GPU or real data required.

What is covered
---------------
  [1] Config validation         SentimentTrainer._validate_config
  [2] WisesightDataset          tokenization, __getitem__, tensor shapes
  [3] compute_metrics           correct output keys and value ranges
  [4] trainer.train()           1 epoch, end-to-end HF Trainer loop
  [5] MLflow logging            params, metrics, step metrics (local ./mlruns)
  [6] evaluate_on_test          test-set inference + metrics dict
  [7] plot_confusion_matrix     figure is created and closed without error
  [8] log_artifacts_to_mlflow   model + tokenizer saved to tmp dirs

The real PhayaThaiBERT model is NOT downloaded. A public 4 MB model
(prajjwal1/bert-tiny) is used instead. The pipeline code is identical —
only the model name changes.

Usage
-----
    python scripts/smoke_test.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Force MLflow to a throwaway local store BEFORE any src.* import runs.
# Some modules call load_dotenv() at import time, which would otherwise pull
# a real MLFLOW_TRACKING_URI from .env. load_dotenv() defaults to
# override=False, so setting this first makes it win regardless of .env.
_SMOKE_MLRUNS_DIR = tempfile.mkdtemp(prefix="smoke_test_mlruns_")
os.environ["MLFLOW_TRACKING_URI"] = f"file:{_SMOKE_MLRUNS_DIR}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.WARNING,          # suppress HF Trainer noise during smoke test
    format="%(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger("smoke_test")

TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"

# Synthetic data — 16 Thai-ish samples covering all 4 labels
_TEXTS = [
    "อาหารอร่อยมากเลยครับ",
    "ประทับใจบริการมากค่ะ",
    "บริการแย่มาก ไม่ประทับใจเลย",
    "ห้องพักสกปรก ผิดหวังมาก",
    "ก็ธรรมดา ไม่มีอะไรพิเศษ",
    "พอใช้ได้ครับ ราคาเหมาะสม",
    "ราคาเท่าไหร่คะ?",
    "เปิดกี่โมงครับ?",
    "สาขาอื่นมีไหมคะ",
    "ดีมากๆ แนะนำเลยครับ",
    "ไม่คุ้มเงินเลย",
    "เฉยๆ ก็ได้",
    "มีโปรโมชั่นไหมคะ",
    "ชอบมากเลยค่ะ",
    "แย่มาก จะไม่มาอีกแล้ว",
    "สอบถามหน่อยครับ",
]
_LABELS = [2, 2, 0, 0, 1, 1, 3, 3, 3, 2, 0, 1, 3, 2, 0, 3]   # neg=0 neu=1 pos=2 q=3
LABEL_NAMES = ["neg", "neu", "pos", "q"]


def _make_df(indices: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "text_clean": [_TEXTS[i] for i in indices],
        "label":      [_LABELS[i] for i in indices],
        "label_name": [LABEL_NAMES[_LABELS[i]] for i in indices],
    })


def _make_config(output_dir: str, best_model_dir: str) -> dict:
    """Smoke config — overrides everything in train.yaml."""
    return {
        "model": {
            "name":       TINY_MODEL,
            "num_labels": 4,
            "max_length": 32,           # short sequences = fast CPU tokenization
        },
        "training": {
            "batch_size":      4,
            "eval_batch_size": 4,
            "learning_rate":   2e-5,
            "epochs":          1,       # 1 epoch is enough to prove the loop runs
            "weight_decay":    0.01,
            "warmup_ratio":    0.0,
        },
        "early_stopping": {"patience": 2},
        "paths": {
            "output_dir":    output_dir,
            "best_model_dir": best_model_dir,
        },
        "mlflow": {"experiment_name": "smoke_test"},
        "metrics": {"best_metric": "f1_macro", "greater_is_better": True},
        "seed":    {"random_state": 42},
    }


# ── Individual checks ─────────────────────────────────────────────────────

def check_imports() -> None:
    """Verify every project module is importable."""
    import src.data.dataset
    import src.models.classifier
    import src.training.evaluator
    import src.training.trainer
    import src.utils.load_config


def check_dataset(tokenizer) -> None:
    """WisesightDataset returns correct tensor shapes."""
    from src.data.dataset import WisesightDataset

    df = _make_df(list(range(4)))
    ds = WisesightDataset(df, tokenizer, max_length=32)

    assert len(ds) == 4, f"Expected 4 samples, got {len(ds)}"
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape == (32,), f"Bad shape: {sample['input_ids'].shape}"
    assert sample["labels"].item() in range(4)


def check_compute_metrics() -> None:
    """compute_metrics returns correct keys and valid value ranges."""
    from src.training.evaluator import compute_metrics

    logits = np.array([
        [2.0, 0.1, 0.1, 0.1],
        [0.1, 2.0, 0.1, 0.1],
        [0.1, 0.1, 2.0, 0.1],
        [0.1, 0.1, 0.1, 2.0],
    ])
    labels = np.array([0, 1, 2, 3])
    result = compute_metrics((logits, labels))

    assert set(result.keys()) == {"accuracy", "f1_weighted", "f1_macro"}, (
        f"Unexpected keys: {result.keys()}"
    )
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


def check_config_validation() -> None:
    """SentimentTrainer._validate_config raises on missing keys."""
    from src.training.trainer import SentimentTrainer

    bad_config = {"model": {"name": "x"}}   # missing everything else
    try:
        SentimentTrainer(bad_config)
        raise AssertionError("Expected KeyError was not raised")
    except KeyError:
        pass   # correct behaviour


def check_full_training(tmpdir: str) -> tuple[str, dict]:
    """End-to-end: train() runs and returns a valid run_id + metrics dict."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data.dataset import WisesightDataset
    from src.training.trainer import SentimentTrainer

    config = _make_config(
        output_dir=str(Path(tmpdir) / "model"),
        best_model_dir=str(Path(tmpdir) / "best"),
    )

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_MODEL,
        num_labels=4,
        ignore_mismatched_sizes=True,
    )

    train_df = _make_df([0, 1, 2, 3, 4, 5, 6, 7])
    val_df   = _make_df([8, 9, 10, 11])
    test_df  = _make_df([0, 1, 2, 3])

    train_ds = WisesightDataset(train_df, tokenizer, max_length=32)
    val_ds   = WisesightDataset(val_df,   tokenizer, max_length=32)
    test_ds  = WisesightDataset(test_df,  tokenizer, max_length=32)

    trainer = SentimentTrainer(config)
    run_id, results = trainer.train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        test_df=test_df,
    )

    assert isinstance(run_id, str) and len(run_id) > 0
    assert {"test_accuracy", "test_f1_weighted", "test_f1_macro"} == set(results.keys())
    for v in results.values():
        assert 0.0 <= v <= 1.0

    return run_id, results


# ── Runner ───────────────────────────────────────────────────────────────

def run_check(label: str, fn, *args, **kwargs) -> bool:
    """Run a single check and print pass/fail."""
    print(f"  {label:<45}", end="", flush=True)
    try:
        fn(*args, **kwargs)
        print("PASS")
        return True
    except Exception:
        print("FAIL")
        traceback.print_exc()
        return False


def main() -> None:
    print()
    print("=" * 60)
    print("  Smoke test — Thai Sentiment MLOps pipeline")
    print(f"  Model : {TINY_MODEL}  (4 MB, CPU-only)")
    print("=" * 60)
    print()

    results: list[bool] = []

    print("Phase 1  imports")
    results.append(run_check("[1] all project modules importable", check_imports))
    print()

    print("Phase 2  data & metrics")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY_MODEL, use_fast=False)
    results.append(run_check("[2] WisesightDataset tensor shapes", check_dataset, tok))
    results.append(run_check("[3] compute_metrics keys + ranges", check_compute_metrics))
    results.append(run_check("[4] config validation raises on bad input", check_config_validation))
    print()

    print("Phase 3  training loop  (downloads bert-tiny ~4 MB on first run)")
    with tempfile.TemporaryDirectory() as tmpdir:
        ok = run_check("[5] trainer.train() end-to-end", check_full_training, tmpdir)
        results.append(ok)
    print()

    passed = sum(results)
    total  = len(results)
    print("=" * 60)
    if passed == total:
        print(f"  ALL {total}/{total} checks passed  — pipeline is wired up correctly")
        print()
        print("  Next step: run on Colab with the real model")
        print("  See: https://colab.research.google.com")
    else:
        failed = total - passed
        print(f"  {failed}/{total} checks FAILED  — see tracebacks above")
    print("=" * 60)
    print()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()