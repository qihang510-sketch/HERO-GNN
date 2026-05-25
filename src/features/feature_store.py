from __future__ import annotations

from pathlib import Path

import numpy as np


def save_features(path: str | Path, features: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)


def load_features(path: str | Path) -> np.ndarray:
    return np.load(Path(path))

