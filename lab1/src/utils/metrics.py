"""评价指标：Accuracy、Macro-F1。"""

from __future__ import annotations

from typing import Dict, Iterable

from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(y_true: Iterable, y_pred: Iterable) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true: Iterable, y_pred: Iterable) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def evaluate_predictions(y_true: Iterable, y_pred: Iterable) -> Dict[str, float]:
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_f1": compute_macro_f1(y_true, y_pred),
    }
