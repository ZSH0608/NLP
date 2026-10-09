"""NYT 数据加载与 Train/Val/Test 划分。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src import config


def load_raw_nyt(path: Path | None = None) -> pd.DataFrame:
    path = path or config.NYT_CSV
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    return df


def load_raw_ag(path: Path | None = None) -> pd.DataFrame:
    path = path or config.AG_CSV
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    return df


def split_nyt(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    val = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = shuffled.iloc[n_train + n_val :].reset_index(drop=True)
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path | None = None,
) -> None:
    out_dir = out_dir or config.DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)


def load_splits(processed_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    processed_dir = processed_dir or config.DATA_PROCESSED_DIR
    return {
        "train": pd.read_csv(processed_dir / "train.csv"),
        "val": pd.read_csv(processed_dir / "val.csv"),
        "test": pd.read_csv(processed_dir / "test.csv"),
    }
