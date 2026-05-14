import os
from typing import Tuple

import pandas as pd
from datasets import Dataset

from .config import MAX_SEQ_LENGTH, TRAIN_DATA_PATH, TEST_DATA_PATH


def _read_csv_stable(path: str) -> pd.DataFrame:
    """Read CSV with a Windows-stable parser mode."""
    return pd.read_csv(path, engine="python")


def _validate_dataframe(frame: pd.DataFrame, source_name: str) -> None:
    """Validate required schema and normalize dtypes."""
    required = {"url", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {sorted(missing)}")

    frame["url"] = frame["url"].astype(str)
    frame["label"] = frame["label"].astype(int)


def load_training_data(train_path: str, test_path: str) -> Tuple[Dataset, Dataset]:
    """Load and validate training data from local CSV files."""
    print(f"\n[DATA] Loading training data from:")
    print(f"  - Train: {train_path}")
    print(f"  - Test: {test_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    train_df = _read_csv_stable(train_path)
    test_df = _read_csv_stable(test_path)
    _validate_dataframe(train_df, "train dataset")
    _validate_dataframe(test_df, "test dataset")

    train_dataset = Dataset.from_pandas(train_df[["url", "label"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["url", "label"]], preserve_index=False)

    print(f"\n[DATA] Dataset loaded:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")

    label_counts = {}
    for i in range(4):
        label_counts[i] = sum(1 for d in train_dataset if d["label"] == i)
    print(f"  - Label distribution: {label_counts}")

    return train_dataset, test_dataset


def tokenize_function(examples, tokenizer):
    """Tokenize URLs for model input"""
    return tokenizer(examples["url"], truncation=True, max_length=MAX_SEQ_LENGTH)
