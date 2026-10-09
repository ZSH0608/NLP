"""结果与模型的简单读写。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
