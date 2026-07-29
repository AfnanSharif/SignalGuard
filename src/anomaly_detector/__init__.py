"""Trainable anomaly detection with a portable local baseline and optional Keras AE."""

from .model import LinearAutoencoder, Prediction

__all__ = ["LinearAutoencoder", "Prediction"]
__version__ = "1.0.0"
