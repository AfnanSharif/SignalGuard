from __future__ import annotations

from pathlib import Path

from .data import load_csv
from .model import LinearAutoencoder, Prediction


class DetectionService:
    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def train_csv(cls, path: str | Path, *, features: list[str] | None = None, label_column: str | None = "label", latent_dim: int = 2, contamination: float = 0.05) -> "DetectionService":
        rows, selected = load_csv(path, features, label_column)
        return cls(LinearAutoencoder(latent_dim, contamination).fit(rows, selected))

    def predict(self, payload: dict[str, float]) -> Prediction:
        return self.model.predict(payload)
