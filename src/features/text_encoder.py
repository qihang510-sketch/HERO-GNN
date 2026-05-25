from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfTextEncoder:
    def __init__(self, max_features: int = 128) -> None:
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        values = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        return self._pad(values)

    def transform(self, texts: list[str]) -> np.ndarray:
        values = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return self._pad(values)

    def feature_names(self) -> list[str]:
        return [f"text_{name}" for name in self.vectorizer.get_feature_names_out()]

    def _pad(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] == self.max_features:
            return values
        padded = np.zeros((values.shape[0], self.max_features), dtype=np.float32)
        padded[:, : values.shape[1]] = values
        return padded
