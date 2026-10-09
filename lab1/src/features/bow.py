"""Task 1：Binary BoW 与 Word Frequency。"""

from __future__ import annotations

from typing import Iterable, List

from sklearn.feature_extraction.text import CountVectorizer

from src.utils.tokenize import tokenize


def _analyzer(text: str) -> List[str]:
    return tokenize(text)


class BinaryBoWVectorizer:
    """Binary Bag of Words：词出现则为 1，否则 0。"""

    def __init__(self) -> None:
        self._vec = CountVectorizer(analyzer=_analyzer, binary=True)

    def fit(self, texts: Iterable[str]) -> "BinaryBoWVectorizer":
        self._vec.fit(list(texts))
        return self

    def transform(self, texts: Iterable[str]):
        # 保持稀疏矩阵，避免高维稠密矩阵撑爆内存
        return self._vec.transform(list(texts))

    def fit_transform(self, texts: Iterable[str]):
        return self.fit(texts).transform(texts)


class CountBoWVectorizer:
    """Word Frequency Bag of Words。"""

    def __init__(self) -> None:
        self._vec = CountVectorizer(analyzer=_analyzer, binary=False)

    def fit(self, texts: Iterable[str]) -> "CountBoWVectorizer":
        self._vec.fit(list(texts))
        return self

    def transform(self, texts: Iterable[str]):
        return self._vec.transform(list(texts))

    def fit_transform(self, texts: Iterable[str]):
        return self.fit(texts).transform(texts)
