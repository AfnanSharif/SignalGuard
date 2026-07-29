import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detector.data import load_csv, load_labeled_csv, profile
from anomaly_detector.evaluation import evaluate_model
from anomaly_detector.model import LinearAutoencoder
from anomaly_detector.providers import KerasAutoencoder
from anomaly_detector.cli import main


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
        cls.rows, cls.features = load_csv(cls.path)
        cls.model = LinearAutoencoder(latent_dim=2, contamination=.08).fit(cls.rows, cls.features)

    def test_training_and_normal_prediction(self):
        prediction = self.model.predict(self.rows[10])
        self.assertGreaterEqual(prediction.risk_score, 0)
        self.assertEqual(set(prediction.feature_errors), set(self.features))
        json.dumps(asdict(prediction))

    def test_obvious_outlier_is_flagged(self):
        prediction = self.model.predict({"amount": 1800, "velocity_1h": 22, "distance_from_home_km": 900, "device_age_days": 1, "failed_attempts": 9, "hour": 3})
        self.assertTrue(prediction.is_anomaly)
        self.assertGreater(prediction.reconstruction_error, prediction.threshold)

    def test_model_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            self.model.save(path)
            restored = LinearAutoencoder.load(path)
            before = self.model.predict(self.rows[0]).reconstruction_error
            after = restored.predict(self.rows[0]).reconstruction_error
            self.assertAlmostEqual(before, after)

    def test_profile_and_label_filter(self):
        report = profile(self.rows, self.features)
        self.assertEqual(report["amount"]["count"], 40)
        self.assertLess(report["amount"]["max"], 100)

    def test_missing_feature_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict({"amount": 10})

    def test_labeled_evaluation_covers_held_out_anomalies(self):
        rows, features, labels = load_labeled_csv(self.path)
        self.assertEqual(features, self.features)
        self.assertEqual(sum(labels), 3)
        report = evaluate_model(self.model, rows, labels)
        self.assertEqual(report["records"], 43)
        self.assertGreaterEqual(report["f1"], 0)
        self.assertLessEqual(report["f1"], 1)

    def test_keras_configuration_validates_without_tensorflow(self):
        with self.assertRaises(ValueError):
            KerasAutoencoder(hidden=())
        with self.assertRaises(ValueError):
            KerasAutoencoder(contamination=.8)

    def test_backend_selectable_cli_trains_portable_model(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "baseline.json"
            with redirect_stdout(io.StringIO()):
                status = main(["train", str(self.path), "--backend", "linear", "--output", str(destination)])
            self.assertEqual(status, 0)
            self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
