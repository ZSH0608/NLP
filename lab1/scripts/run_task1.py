#!/usr/bin/env python3
"""运行 Task 1：Bag of Words + Logistic Regression。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.task1_bow import run_all_task1


def main() -> None:
    # 请先运行: python scripts/prepare_data.py
    results = run_all_task1()
    print("=== Task 1 Results (NYT Test) ===")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
