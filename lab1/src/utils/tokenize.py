"""文本分词工具。"""

from __future__ import annotations

from typing import List

from nltk import word_tokenize


def tokenize(text: str) -> List[str]:
    """小写 + nltk.word_tokenize。"""
    if not isinstance(text, str) or not text.strip():
        return []
    return word_tokenize(text.lower())
