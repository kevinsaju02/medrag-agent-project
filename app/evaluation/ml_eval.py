from __future__ import annotations

import json
from pathlib import Path


def load_ml_metrics(metrics_path: str | Path) -> dict[str, object]:
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))
