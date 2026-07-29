from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of no values")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@dataclass(frozen=True)
class Prediction:
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
    risk_score: float
    feature_errors: dict[str, float]


class RobustScaler:
    def __init__(self) -> None:
        self.centers: list[float] = []
        self.scales: list[float] = []

    def fit(self, matrix: list[list[float]]) -> "RobustScaler":
        columns = list(zip(*matrix))
        self.centers, self.scales = [], []
        for column in columns:
            center = median(column)
            mad = median(abs(value - center) for value in column) * 1.4826
            if mad < 1e-9:
                mean = sum(column) / len(column)
                mad = math.sqrt(sum((value - mean) ** 2 for value in column) / max(1, len(column) - 1))
            self.centers.append(float(center))
            self.scales.append(float(mad if mad >= 1e-9 else 1.0))
        return self

    def transform_one(self, row: list[float]) -> list[float]:
        if len(row) != len(self.centers):
            raise ValueError("feature count does not match the fitted scaler")
        return [(value - center) / scale for value, center, scale in zip(row, self.centers, self.scales)]


class LinearAutoencoder:
    """Portable linear encoder/decoder using robust scaling and PCA-style weights.

    It is an offline baseline with the same reconstruction-error contract as the
    optional deep Keras implementation, useful for smoke runs and constrained hosts.
    """

    def __init__(self, latent_dim: int = 2, contamination: float = 0.05) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        self.latent_dim = latent_dim
        self.contamination = contamination
        self.features: list[str] = []
        self.scaler = RobustScaler()
        self.data_mean: list[float] = []
        self.components: list[list[float]] = []
        self.threshold = 0.0
        self.training_size = 0

    @property
    def fitted(self) -> bool:
        return bool(self.features and self.components and self.threshold >= 0)

    def fit(self, rows: list[dict[str, float]], features: list[str] | None = None) -> "LinearAutoencoder":
        if not rows:
            raise ValueError("training data cannot be empty")
        self.features = features or list(rows[0])
        if len(self.features) < 2:
            raise ValueError("at least two features are required")
        if len(rows) < len(self.features) + 1:
            raise ValueError("training rows must exceed the number of features")
        matrix = [self._values(row) for row in rows]
        self.scaler.fit(matrix)
        scaled = [self.scaler.transform_one(row) for row in matrix]
        self.data_mean = [sum(row[index] for row in scaled) / len(scaled) for index in range(len(self.features))]
        covariance = self._covariance(scaled)
        self.components = self._principal_components(covariance, min(self.latent_dim, len(self.features) - 1))
        errors = [self._error(vector)[0] for vector in scaled]
        self.threshold = _quantile(errors, 1 - self.contamination)
        self.training_size = len(rows)
        return self

    def predict(self, row: dict[str, float]) -> Prediction:
        if not self.fitted:
            raise RuntimeError("model has not been fitted")
        scaled = self.scaler.transform_one(self._values(row))
        error, per_feature = self._error(scaled)
        ratio = error / max(self.threshold, 1e-12)
        risk = min(100.0, 100 * (1 - math.exp(-ratio)))
        return Prediction(error > self.threshold, error, self.threshold, round(risk, 2), dict(zip(self.features, per_feature)))

    def predict_many(self, rows: list[dict[str, float]]) -> list[Prediction]:
        return [self.predict(row) for row in rows]

    def _values(self, row: dict[str, float]) -> list[float]:
        missing = [name for name in self.features if name not in row]
        if missing:
            raise ValueError(f"missing features: {', '.join(missing)}")
        values = [float(row[name]) for name in self.features]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("features must be finite numbers")
        return values

    @staticmethod
    def _covariance(matrix: list[list[float]]) -> list[list[float]]:
        n, width = len(matrix), len(matrix[0])
        means = [sum(row[col] for row in matrix) / n for col in range(width)]
        return [
            [sum((row[i] - means[i]) * (row[j] - means[j]) for row in matrix) / max(1, n - 1) for j in range(width)]
            for i in range(width)
        ]

    @staticmethod
    def _principal_components(covariance: list[list[float]], count: int) -> list[list[float]]:
        components: list[list[float]] = []
        width = len(covariance)
        for component_index in range(count):
            vector = [1 / (index + component_index + 1) for index in range(width)]
            for _ in range(120):
                candidate = _matvec(covariance, vector)
                for prior in components:
                    projection = _dot(candidate, prior)
                    candidate = [value - projection * basis for value, basis in zip(candidate, prior)]
                length = _norm(candidate)
                if length < 1e-12:
                    candidate = [1.0 if index == component_index % width else 0.0 for index in range(width)]
                    for prior in components:
                        projection = _dot(candidate, prior)
                        candidate = [value - projection * basis for value, basis in zip(candidate, prior)]
                    length = _norm(candidate)
                candidate = [value / max(length, 1e-12) for value in candidate]
                if _norm([a - b for a, b in zip(candidate, vector)]) < 1e-10 or _norm([a + b for a, b in zip(candidate, vector)]) < 1e-10:
                    vector = candidate
                    break
                vector = candidate
            components.append(vector)
        return components

    def _error(self, scaled: list[float]) -> tuple[float, list[float]]:
        centered = [value - center for value, center in zip(scaled, self.data_mean)]
        reconstruction = [0.0] * len(scaled)
        for component in self.components:
            encoded = _dot(centered, component)
            reconstruction = [current + encoded * weight for current, weight in zip(reconstruction, component)]
        reconstruction = [value + center for value, center in zip(reconstruction, self.data_mean)]
        squared = [(actual - rebuilt) ** 2 for actual, rebuilt in zip(scaled, reconstruction)]
        return sum(squared) / len(squared), squared

    def to_dict(self) -> dict[str, object]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted model")
        return {
            "format_version": 1,
            "model_type": "robust-linear-autoencoder",
            "latent_dim": self.latent_dim,
            "contamination": self.contamination,
            "features": self.features,
            "centers": self.scaler.centers,
            "scales": self.scaler.scales,
            "data_mean": self.data_mean,
            "components": self.components,
            "threshold": self.threshold,
            "training_size": self.training_size,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearAutoencoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("model_type") != "robust-linear-autoencoder":
            raise ValueError("unsupported model format")
        model = cls(int(payload["latent_dim"]), float(payload["contamination"]))
        model.features = [str(name) for name in payload["features"]]
        model.scaler.centers = [float(value) for value in payload["centers"]]
        model.scaler.scales = [float(value) for value in payload["scales"]]
        model.data_mean = [float(value) for value in payload.get("data_mean", [0.0] * len(model.features))]
        model.components = [[float(value) for value in row] for row in payload["components"]]
        model.threshold = float(payload["threshold"])
        model.training_size = int(payload["training_size"])
        return model
