from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..model import Prediction


class KerasAutoencoder:
    """Lazy TensorFlow deep autoencoder with robust training-data thresholding."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 16, 8),
        contamination: float = 0.05,
        seed: int = 42,
        learning_rate: float = 1e-3,
        validation_split: float = 0.15,
        patience: int = 7,
    ) -> None:
        if not hidden or any(units < 1 for units in hidden):
            raise ValueError("hidden layers must contain positive widths")
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        if learning_rate <= 0 or not 0 < validation_split < 0.5 or patience < 1:
            raise ValueError("learning rate, validation split, and patience are invalid")
        self.hidden, self.contamination, self.seed = hidden, contamination, seed
        self.learning_rate, self.validation_split, self.patience = learning_rate, validation_split, patience
        self.model: Any = None
        self.scaler: Any = None
        self.threshold: float | None = None
        self.features: list[str] = []
        self.training_size = 0

    def fit(self, rows: list[dict[str, float]], features: list[str], epochs: int = 50, batch_size: int = 64) -> dict[str, list[float]]:
        if len(rows) < max(10, len(features) + 2):
            raise ValueError("deep training needs at least ten normal rows")
        if not features or epochs < 1 or batch_size < 1:
            raise ValueError("features, epochs, and batch size must be valid")
        try:
            import numpy as np
            import tensorflow as tf
            from sklearn.preprocessing import RobustScaler
        except ImportError as exc:
            raise RuntimeError("Install tensorflow, numpy, and scikit-learn for the deep model") from exc
        tf.keras.utils.set_random_seed(self.seed)
        self.features = features
        self.training_size = len(rows)
        matrix = np.asarray([[row[name] for name in features] for row in rows], dtype="float32")
        self.scaler = RobustScaler().fit(matrix)
        scaled = self.scaler.transform(matrix)
        inputs = tf.keras.Input(shape=(len(features),))
        value = inputs
        for units in self.hidden:
            value = tf.keras.layers.Dense(units, activation="relu")(value)
            value = tf.keras.layers.BatchNormalization()(value)
        for units in reversed(self.hidden[:-1]):
            value = tf.keras.layers.Dense(units, activation="relu")(value)
        outputs = tf.keras.layers.Dense(len(features))(value)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss="mse")
        history = self.model.fit(
            scaled,
            scaled,
            validation_split=self.validation_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)],
        )
        reconstructed = self.model.predict(scaled, verbose=0)
        errors = np.mean(np.square(scaled - reconstructed), axis=1)
        self.threshold = float(np.quantile(errors, 1 - self.contamination))
        return {key: [float(item) for item in values] for key, values in history.history.items()}

    def predict(self, row: dict[str, float]) -> Prediction:
        if self.model is None or self.scaler is None or self.threshold is None:
            raise RuntimeError("model has not been fitted")
        import numpy as np
        matrix = self.scaler.transform([[row[name] for name in self.features]])
        rebuilt = self.model.predict(matrix, verbose=0)
        squared = np.square(matrix - rebuilt)[0]
        error = float(np.mean(squared))
        ratio = error / max(self.threshold, 1e-12)
        risk = min(100.0, 100 * (1 - math.exp(-ratio)))
        return Prediction(error > self.threshold, error, self.threshold, round(risk, 2), {name: float(value) for name, value in zip(self.features, squared)})

    def save(self, directory: str | Path) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("model has not been fitted")
        import joblib
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "autoencoder.keras")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        (directory / "metadata.json").write_text(json.dumps({"features": self.features, "threshold": self.threshold, "contamination": self.contamination, "hidden": self.hidden, "training_size": self.training_size, "learning_rate": self.learning_rate, "validation_split": self.validation_split, "patience": self.patience}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "KerasAutoencoder":
        try:
            import joblib
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("Install tensorflow and joblib to load the deep model") from exc
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(
            tuple(int(value) for value in metadata.get("hidden", (32, 16, 8))),
            float(metadata["contamination"]),
            learning_rate=float(metadata.get("learning_rate", 1e-3)),
            validation_split=float(metadata.get("validation_split", 0.15)),
            patience=int(metadata.get("patience", 7)),
        )
        instance.features = [str(name) for name in metadata["features"]]
        instance.threshold = float(metadata["threshold"])
        instance.training_size = int(metadata.get("training_size", 0))
        instance.model = tf.keras.models.load_model(directory / "autoencoder.keras")
        instance.scaler = joblib.load(directory / "scaler.joblib")
        return instance
