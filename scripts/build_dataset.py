from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data_processing.dataset_builder import save_dataset_artifacts


def main() -> None:
    artifact_paths = save_dataset_artifacts(output_dir=ROOT_DIR / "data", record_count=200, seed=42)

    for name, path in artifact_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
