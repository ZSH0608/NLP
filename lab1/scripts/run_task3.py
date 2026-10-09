#!/usr/bin/env python3
"""运行 Task 3：BERT Fine-tuning（max_length=64, epochs=3）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.task3_bert import run_bert


def main() -> None:
    # 前置：prepare_data.py；首次运行需下载 Hugging Face 模型
    metrics = run_bert()
    print("=== Task 3 Results (NYT Test) ===")
    print(metrics)


if __name__ == "__main__":
    main()
