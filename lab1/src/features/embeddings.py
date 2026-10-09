"""Task 2：基于词向量平均的文档表示。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
from gensim.models import Word2Vec

from src import config
from src.utils.tokenize import tokenize

EmbeddingLookup = Union[Dict[str, np.ndarray], object]


def load_glove(path: Path | None = None, dim: int = config.EMBEDDING_DIM) -> EmbeddingLookup:
    """
    优先加载本地 glove.6B.100d.txt；
    若不存在则通过 gensim 下载 glove-wiki-gigaword-100（同为 GloVe 6B 100d）。
    """
    path = path or config.GLOVE_PATH
    if path.exists():
        embeddings: Dict[str, np.ndarray] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != dim + 1:
                    continue
                embeddings[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        return embeddings

    import gensim.downloader as api

    print("Local GloVe file not found; downloading glove-wiki-gigaword-100 via gensim ...")
    return api.load("glove-wiki-gigaword-100")


def train_word2vec(
    sentences: List[List[str]],
    vector_size: int = config.EMBEDDING_DIM,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    epochs: int = 5,
    seed: int = config.RANDOM_SEED,
    **kwargs,
) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=4,
        **kwargs,
    )
    return model


def _get_vector(embedding_lookup: EmbeddingLookup, token: str, dim: int) -> np.ndarray | None:
    if isinstance(embedding_lookup, dict):
        return embedding_lookup.get(token)
    # gensim KeyedVectors / Word2Vec.wv
    wv = getattr(embedding_lookup, "wv", embedding_lookup)
    if token in wv:
        return np.asarray(wv[token], dtype=np.float32)
    return None


def docs_to_avg_vectors(
    texts: Iterable[str],
    embedding_lookup: EmbeddingLookup,
    dim: int = config.EMBEDDING_DIM,
) -> np.ndarray:
    vectors = []
    for text in texts:
        tokens = tokenize(text)
        acc = np.zeros(dim, dtype=np.float32)
        count = 0
        for tok in tokens:
            vec = _get_vector(embedding_lookup, tok, dim)
            if vec is None:
                continue
            acc += vec
            count += 1
        if count > 0:
            acc /= count
        vectors.append(acc)
    return np.vstack(vectors)
