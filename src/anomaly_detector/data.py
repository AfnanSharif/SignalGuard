from __future__ import annotations

import csv
import math
from pathlib import Path


def load_labeled_csv(
    path: str | Path,
    features: list[str] | None = None,
    label_column: str | None = "label",
    normal_label: str = "0",
) -> tuple[list[dict[str, float]], list[str], list[bool]]:
    """Load every validated record plus a binary anomaly label for evaluation."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV is missing a header")
        selected = features or [name for name in reader.fieldnames if name != label_column]
        if not selected:
            raise ValueError("at least one feature column is required")
        unknown = set(selected) - set(reader.fieldnames)
        if unknown:
            raise ValueError(f"unknown feature columns: {', '.join(sorted(unknown))}")
        rows: list[dict[str, float]] = []
        labels: list[bool] = []
        for line, raw in enumerate(reader, 2):
            row: dict[str, float] = {}
            for name in selected:
                try:
                    value = float(raw[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"line {line}: {name} must be numeric") from exc
                if not math.isfinite(value):
                    raise ValueError(f"line {line}: {name} must be finite")
                row[name] = value
            rows.append(row)
            label = raw.get(label_column, normal_label) if label_column else normal_label
            labels.append(label not in {"", normal_label})
    if not rows:
        raise ValueError("dataset has no records")
    return rows, selected, labels


def load_csv(path: str | Path, features: list[str] | None = None, label_column: str | None = "label", normal_label: str = "0") -> tuple[list[dict[str, float]], list[str]]:
    all_rows, selected, labels = load_labeled_csv(path, features, label_column, normal_label)
    rows = [row for row, is_anomaly in zip(all_rows, labels) if not is_anomaly]
    if len(rows) < max(5, len(selected) + 1):
        raise ValueError("not enough normal training rows")
    return rows, selected


def profile(rows: list[dict[str, float]], features: list[str]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("cannot profile an empty dataset")
    report = {}
    for feature in features:
        values = sorted(float(row[feature]) for row in rows)
        n = len(values)
        mean = sum(values) / n
        variance = sum((value - mean) ** 2 for value in values) / max(1, n - 1)
        report[feature] = {
            "count": n,
            "min": values[0],
            "mean": mean,
            "median": values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2,
            "max": values[-1],
            "std": math.sqrt(variance),
        }
    return report
