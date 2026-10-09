#!/usr/bin/env python3
"""将 NYT 随机打乱并按 80/10/10 划分，写入 data/processed/。"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证可从项目根目录以 `python scripts/xxx.py` 运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split import load_raw_nyt, save_splits, split_nyt


def main() -> None:
    print("Loading NYT ...")
    df = load_raw_nyt()
    print(f"Total samples: {len(df)}")

    train, val, test = split_nyt(df)
    print(f"Train/Val/Test: {len(train)} / {len(val)} / {len(test)}")

    save_splits(train, val, test)
    print("Saved to data/processed/")


if __name__ == "__main__":
    main()
