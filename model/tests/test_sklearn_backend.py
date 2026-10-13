"""Tests for the optional sklearn report backend."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from model.config import ModelConfig
from model.generator import LabReportGenerator
from model.inference import SklearnModelBackend, build_backend
from model.schemas import SectionName
from model.validator import validate_report


SKLEARN_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("sklearn", "joblib", "numpy", "pandas")
)
FEATURE_NAMES = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
)


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn dependencies are not installed")
class SklearnModelBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.preprocessing import StandardScaler

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        temporary_path = Path(self.temporary_directory.name)
        self.model_path = temporary_path / "model.joblib"
        self.scaler_path = temporary_path / "scaler.joblib"

        raw_features = pd.DataFrame(
            [
                [float(row_index + column_index) for column_index in range(len(FEATURE_NAMES))]
                for row_index in range(4)
            ],
            columns=FEATURE_NAMES,
        )
        scaler = StandardScaler().fit(raw_features)
        scaled_features = pd.DataFrame(
            scaler.transform(raw_features),
            columns=FEATURE_NAMES,
        )
        classifier = DummyClassifier(strategy="constant", constant=1)
        classifier.fit(scaled_features, [0, 1, 0, 1])
        joblib.dump(classifier, self.model_path)
        joblib.dump(scaler, self.scaler_path)

        self.config = ModelConfig(
            backend="sklearn",
            model_id="test-sklearn-model",
            sklearn_model_path=str(self.model_path),
            sklearn_scaler_path=str(self.scaler_path),
        )

    def test_backend_is_registered(self) -> None:
        backend = build_backend(self.config)
        self.assertIsInstance(backend, SklearnModelBackend)

    def test_missing_measurements_are_reported_without_crashing(self) -> None:
        report = LabReportGenerator(config=self.config).generate(
            {
                "title": "Incomplete wine sample",
                "objective": "classify wine quality",
                "measurements": [
                    {"name": "alcohol", "value": 10.0, "unit": ""}
                ],
            }
        )

        conclusion = report.get_section(SectionName.CONCLUSION)
        self.assertIsNotNone(conclusion)
        assert conclusion is not None
        self.assertIn("Prediction unavailable", conclusion.content)
        self.assertIn("fixed acidity", conclusion.content)
        self.assertIn("volatile acidity", conclusion.content)

    def test_valid_reordered_measurements_produce_prediction(self) -> None:
        measurements = [
            {
                "name": name.upper().replace(" ", "   "),
                "value": float(index + 1),
                "unit": "",
            }
            for index, name in reversed(list(enumerate(FEATURE_NAMES)))
        ]
        report = LabReportGenerator(config=self.config).generate(
            {
                "title": "Red wine sample",
                "objective": "classify wine quality",
                "materials": ["red wine sample"],
                "procedure": ["Measure the listed chemical properties"],
                "measurements": measurements,
            }
        )

        conclusion = report.get_section(SectionName.CONCLUSION)
        self.assertIsNotNone(conclusion)
        assert conclusion is not None
        self.assertIn("Predicted quality: Good", conclusion.content)
        self.assertIn("confidence 1.00", conclusion.content)
        self.assertTrue(validate_report(report).is_valid)


if __name__ == "__main__":
    unittest.main()