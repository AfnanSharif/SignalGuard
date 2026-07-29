from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(ROOT / ".env")

from flask import Flask, jsonify, render_template, request

from anomaly_detector.model import LinearAutoencoder
from anomaly_detector.service import DetectionService
from anomaly_detector.data import load_csv, profile

SAMPLE = ROOT / "data" / "transactions.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.json"))


def load_service() -> DetectionService:
    if os.getenv("MODEL_BACKEND", "linear").lower() == "keras":
        from anomaly_detector.providers import KerasAutoencoder
        return DetectionService(KerasAutoencoder.load(MODEL_PATH))
    if MODEL_PATH.exists():
        return DetectionService(LinearAutoencoder.load(MODEL_PATH))
    return DetectionService.train_csv(
        SAMPLE,
        latent_dim=int(os.getenv("LATENT_DIM", "2")),
        contamination=float(os.getenv("CONTAMINATION", ".05")),
    )


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
service = load_service()
sample_rows, sample_features = load_csv(SAMPLE)
sample_profile = profile(sample_rows, sample_features)
for values in sample_profile.values():
    span = max(values["max"] - values["min"], 1e-9)
    values["mean_position"] = max(0.0, min(100.0, 100 * (values["mean"] - values["min"]) / span))


@app.get("/")
def index():
    return render_template(
        "index.html",
        features=service.model.features,
        threshold=service.model.threshold,
        training_size=getattr(service.model, "training_size", 0),
        backend=type(service.model).__name__,
        profile=sample_profile,
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": type(service.model).__name__, "features": service.model.features})


@app.get("/api/v1/profile")
def data_profile():
    return jsonify({"rows": len(sample_rows), "features": sample_profile})


def parse_payload(payload) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    unknown = set(payload) - set(service.model.features)
    if unknown:
        raise ValueError(f"unknown features: {', '.join(sorted(unknown))}")
    return {name: float(payload[name]) for name in service.model.features}


@app.post("/api/v1/predict")
def predict():
    try:
        result = service.predict(parse_payload(request.get_json(silent=True)))
        return jsonify(asdict(result))
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/v1/batch")
def predict_batch():
    payload = request.get_json(silent=True)
    if not isinstance(payload, list) or len(payload) > 500:
        return jsonify({"error": "body must be a list of at most 500 objects"}), 400
    try:
        return jsonify({"predictions": [asdict(service.predict(parse_payload(row))) for row in payload]})
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
