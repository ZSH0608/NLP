"""Task 3 流程：BERT Fine-tuning → Test 指标。"""

from __future__ import annotations

import time
from typing import Dict

from src import config
from src.data.split import load_splits
from src.models.bert_classifier import predict_bert, train_bert
from src.utils.io import save_metrics
from src.utils.metrics import evaluate_predictions


def run_bert() -> Dict[str, float]:
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    t0 = time.time()
    model, tokenizer, maps = train_bert(
        train_texts=train["text"].tolist(),
        train_labels=train["label"].tolist(),
        val_texts=val["text"].tolist(),
        val_labels=val["label"].tolist(),
    )
    elapsed = time.time() - t0

    pred_ids = predict_bert(model, tokenizer, test["text"].tolist())
    id2label = maps["id2label"]
    y_pred = [id2label[int(i)] for i in pred_ids]
    y_true = test["label"].tolist()
    metrics = evaluate_predictions(y_true, y_pred)
    metrics["method"] = "bert"
    metrics["device"] = maps.get("device", "unknown")
    metrics["train_seconds"] = round(elapsed, 2)
    save_metrics(metrics, config.RESULTS_DIR / "task3_bert.json")
    return metrics
