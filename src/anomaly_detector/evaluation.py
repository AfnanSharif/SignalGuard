from __future__ import annotations

from statistics import mean
from typing import Protocol

from .model import Prediction


class Detector(Protocol):
    def predict(self, row: dict[str, float]) -> Prediction: ...


def evaluate_model(model: Detector, rows: list[dict[str, float]], labels: list[bool]) -> dict[str, float | int]:
    if not rows or len(rows) != len(labels):
        raise ValueError("rows and labels must be non-empty and have equal length")
    predictions = [model.predict(row) for row in rows]
    predicted = [item.is_anomaly for item in predictions]
    tp = sum(actual and guess for actual, guess in zip(labels, predicted))
    fp = sum(not actual and guess for actual, guess in zip(labels, predicted))
    tn = sum(not actual and not guess for actual, guess in zip(labels, predicted))
    fn = sum(actual and not guess for actual, guess in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "records": len(rows),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_reconstruction_error": mean(item.reconstruction_error for item in predictions),
    }
