from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class NumericEncoder:
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(values)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.transform(values)


def numeric_feature_columns(columns: list[str]) -> list[str]:
    return sorted(
        [column for column in columns if column.startswith("feat_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )
