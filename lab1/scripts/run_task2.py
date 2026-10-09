#!/usr/bin/env python3
"""运行 Task 2：GloVe / Word2Vec(AG) / Word2Vec(NYT) + Logistic Regression。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.task2_word2vec import run_all_task2


def main() -> None:
    # 前置：prepare_data.py；GloVe 文件放到 embeddings/glove.6B.100d.txt
    results = run_all_task2()
    print("=== Task 2 Results (NYT Test) ===")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
