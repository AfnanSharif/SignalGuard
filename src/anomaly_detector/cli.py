from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .data import load_csv, load_labeled_csv, profile
from .evaluation import evaluate_model
from .model import LinearAutoencoder


def _hidden(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden layers must be comma-separated integers") from exc
    if not layers or any(item < 1 for item in layers):
        raise argparse.ArgumentTypeError("hidden layer widths must be positive")
    return layers


def _load_payload(raw: str) -> dict[str, float]:
    source = None if raw.lstrip().startswith("{") else Path(raw)
    payload = json.loads(source.read_text(encoding="utf-8")) if source and source.exists() else json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("values must be a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, tune, compare, or run reconstruction-error anomaly detectors")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit the portable PCA baseline or deep Keras autoencoder")
    train.add_argument("csv", type=Path)
    train.add_argument("--backend", choices=("linear", "keras"), default="linear")
    train.add_argument("--output", type=Path)
    train.add_argument("--features", help="comma-separated columns; defaults to all but label")
    train.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "2")))
    train.add_argument("--hidden", type=_hidden, default=_hidden(os.getenv("KERAS_HIDDEN", "32,16,8")))
    train.add_argument("--epochs", type=int, default=int(os.getenv("KERAS_EPOCHS", "50")))
    train.add_argument("--batch-size", type=int, default=int(os.getenv("KERAS_BATCH_SIZE", "64")))
    train.add_argument("--contamination", type=float, default=float(os.getenv("CONTAMINATION", ".05")))
    predict = commands.add_parser("predict", help="score one JSON record")
    predict.add_argument("model", type=Path)
    predict.add_argument("values", help="JSON object or path to a JSON file")
    predict.add_argument("--backend", choices=("linear", "keras"), default="linear")
    describe = commands.add_parser("profile", help="print descriptive EDA for normal rows")
    describe.add_argument("csv", type=Path)
    benchmark = commands.add_parser("benchmark", help="compare PCA against tuned Keras architectures on labeled data")
    benchmark.add_argument("csv", type=Path)
    benchmark.add_argument("--architectures", default="32,16,8;64,32,8;32,8")
    benchmark.add_argument("--epochs", type=int, default=int(os.getenv("KERAS_EPOCHS", "50")))
    benchmark.add_argument("--batch-size", type=int, default=int(os.getenv("KERAS_BATCH_SIZE", "64")))
    benchmark.add_argument("--contamination", type=float, default=float(os.getenv("CONTAMINATION", ".05")))
    benchmark.add_argument("--output", type=Path, default=Path("models/deep"))
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "train":
        selected = args.features.split(",") if args.features else None
        rows, features = load_csv(args.csv, selected)
        if args.backend == "keras":
            from .providers import KerasAutoencoder

            model = KerasAutoencoder(args.hidden, args.contamination)
            history = model.fit(rows, features, args.epochs, args.batch_size)
            output = args.output or Path("models/deep")
            model.save(output)
            payload = {"saved": str(output), "backend": "keras", "features": features, "threshold": model.threshold, "training_rows": model.training_size, "epochs_run": len(history.get("loss", []))}
        else:
            model = LinearAutoencoder(args.latent_dim, args.contamination).fit(rows, features)
            output = args.output or Path("models/model.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            model.save(output)
            payload = {"saved": str(output), "backend": "linear", "features": features, "threshold": model.threshold, "training_rows": model.training_size}
        print(json.dumps(payload, indent=2))
    elif args.command == "predict":
        if args.backend == "keras":
            from .providers import KerasAutoencoder

            model = KerasAutoencoder.load(args.model)
        else:
            model = LinearAutoencoder.load(args.model)
        print(json.dumps(asdict(model.predict(_load_payload(args.values))), indent=2))
    elif args.command == "profile":
        rows, features = load_csv(args.csv)
        print(json.dumps(profile(rows, features), indent=2))
    else:
        from .providers import KerasAutoencoder

        all_rows, features, labels = load_labeled_csv(args.csv)
        normal_rows = [row for row, label in zip(all_rows, labels) if not label]
        baseline = LinearAutoencoder(min(2, len(features) - 1), args.contamination).fit(normal_rows, features)
        reports: list[dict[str, object]] = [{"backend": "linear", **evaluate_model(baseline, all_rows, labels)}]
        candidates: list[tuple[float, object, dict[str, object]]] = []
        for raw in args.architectures.split(";"):
            architecture = _hidden(raw)
            model = KerasAutoencoder(architecture, args.contamination)
            history = model.fit(normal_rows, features, args.epochs, args.batch_size)
            report = {"backend": "keras", "hidden": architecture, "epochs_run": len(history.get("loss", [])), **evaluate_model(model, all_rows, labels)}
            reports.append(report)
            candidates.append((float(report["f1"]), model, report))
        _, best_model, best_report = max(candidates, key=lambda item: (item[0], -float(item[2]["mean_reconstruction_error"])))
        best_model.save(args.output)
        print(json.dumps({"reports": reports, "selected": best_report, "saved": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
