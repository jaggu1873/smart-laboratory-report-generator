"""Generate and validate a sample Wine Quality laboratory report."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import LabReportGenerator, ModelConfig, validate_report


def main() -> int:
    """Print a sample report with the trained sklearn backend."""
    raw_experiment = {
        "title": "Red Wine Quality Sample",
        "objective": "assess whether the red wine meets the good-quality threshold",
        "materials": ["red wine sample", "laboratory analysis equipment"],
        "procedure": [
            "Measure the physicochemical properties of the wine sample",
            "Compare the measurements with the trained quality classifier",
        ],
        "measurements": [
            {"name": "fixed acidity", "value": 7.4, "unit": ""},
            {"name": "volatile acidity", "value": 0.70, "unit": ""},
            {"name": "citric acid", "value": 0.00, "unit": ""},
            {"name": "residual sugar", "value": 1.9, "unit": ""},
            {"name": "chlorides", "value": 0.076, "unit": ""},
            {"name": "free sulfur dioxide", "value": 11.0, "unit": ""},
            {"name": "total sulfur dioxide", "value": 34.0, "unit": ""},
            {"name": "density", "value": 0.9978, "unit": ""},
            {"name": "pH", "value": 3.51, "unit": ""},
            {"name": "sulphates", "value": 0.56, "unit": ""},
            {"name": "alcohol", "value": 9.4, "unit": ""},
        ],
    }
    generator = LabReportGenerator(
        config=ModelConfig(
            backend="sklearn",
            model_id="sklearn-winequality-baseline",
        )
    )
    report = generator.generate(raw_experiment)
    validation = validate_report(report)

    print(report.title)
    print(f"Status: {report.status.value}")
    for section in report.sections:
        print(f"\n## {section.name.value.title()}\n{section.content}")
    if not validation.is_valid:
        print("\nValidation: FAILED")
        for issue in validation.issues:
            print(f"- {issue.code}: {issue.message}")
        return 1

    print("\nValidation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())