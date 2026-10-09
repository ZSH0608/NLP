"""Task 2 流程：词向量平均文档表示 + Logistic Regression。"""

from __future__ import annotations

from typing import Dict

from src import config
from src.data.split import load_raw_ag, load_splits
from src.features.embeddings import docs_to_avg_vectors, load_glove, train_word2vec
from src.models.classifiers import build_logreg, predict_logreg, train_logreg
from src.utils.io import save_metrics
from src.utils.metrics import evaluate_predictions
from src.utils.tokenize import tokenize


def _train_eval_with_lookup(name: str, embedding_lookup) -> Dict[str, float]:
    splits = load_splits()
    train, test = splits["train"], splits["test"]

    X_train = docs_to_avg_vectors(train["text"].tolist(), embedding_lookup)
    X_test = docs_to_avg_vectors(test["text"].tolist(), embedding_lookup)
    y_train = train["label"].tolist()
    y_test = test["label"].tolist()

    model = build_logreg()
    train_logreg(model, X_train, y_train)
    y_pred = predict_logreg(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["method"] = name
    save_metrics(metrics, config.RESULTS_DIR / f"task2_{name}.json")
    return metrics


def run_glove() -> Dict[str, float]:
    glove = load_glove()
    return _train_eval_with_lookup("glove", glove)


def run_w2v_ag() -> Dict[str, float]:
    ag = load_raw_ag()
    sentences = [tokenize(t) for t in ag["text"].tolist()]
    w2v = train_word2vec(sentences)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    w2v.save(str(config.MODELS_DIR / "w2v_ag.model"))
    return _train_eval_with_lookup("w2v_ag", w2v)


def run_w2v_nyt() -> Dict[str, float]:
    splits = load_splits()
    sentences = [tokenize(t) for t in splits["train"]["text"].tolist()]
    w2v = train_word2vec(sentences)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    w2v.save(str(config.MODELS_DIR / "w2v_nyt.model"))
    return _train_eval_with_lookup("w2v_nyt", w2v)


def run_all_task2() -> Dict[str, Dict[str, float]]:
    return {
        "glove": run_glove(),
        "w2v_ag": run_w2v_ag(),
        "w2v_nyt": run_w2v_nyt(),
    }
