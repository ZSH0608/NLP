"""Task 1 流程：BoW 特征 + Logistic Regression → Test 指标。"""

from __future__ import annotations

from typing import Dict

from src import config
from src.data.split import load_splits
from src.features.bow import BinaryBoWVectorizer, CountBoWVectorizer
from src.models.classifiers import build_logreg, predict_logreg, train_logreg
from src.utils.io import save_metrics
from src.utils.metrics import evaluate_predictions


def _run_bow(name: str, vectorizer) -> Dict[str, float]:
    splits = load_splits()
    train, test = splits["train"], splits["test"]

    X_train = vectorizer.fit_transform(train["text"].tolist())
    X_test = vectorizer.transform(test["text"].tolist())
    y_train = train["label"].tolist()
    y_test = test["label"].tolist()

    model = build_logreg()
    train_logreg(model, X_train, y_train)
    y_pred = predict_logreg(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["method"] = name
    save_metrics(metrics, config.RESULTS_DIR / f"task1_{name}.json")
    return metrics


def run_binary_bow() -> Dict[str, float]:
    return _run_bow("binary", BinaryBoWVectorizer())


def run_word_frequency() -> Dict[str, float]:
    return _run_bow("count", CountBoWVectorizer())


def run_all_task1() -> Dict[str, Dict[str, float]]:
    return {
        "binary": run_binary_bow(),
        "count": run_word_frequency(),
    }
