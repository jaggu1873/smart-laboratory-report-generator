"""Evaluate the selected baseline model and track final test metrics."""

import json
from pathlib import Path

import joblib
import matplotlib
import mlflow
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")


def main() -> None:
    """Write final test metrics and log them to a final-evaluation run."""
    project_root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load((project_root / "params.yaml").read_text(encoding="utf-8"))
    model = joblib.load(project_root / "models" / "best_model.joblib")
    test_data = pd.read_csv(project_root / "data" / "processed" / "test.csv")
    test_features = test_data.drop(columns="good")
    test_labels = test_data["good"].astype(int)

    predictions = model.predict(test_features)
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(test_features)[:, positive_class_index]
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(precision_score(test_labels, predictions, zero_division=0)),
        "recall": float(recall_score(test_labels, predictions, zero_division=0)),
        "f1": float(f1_score(test_labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_labels, probabilities)),
    }

    metrics_dir = project_root / "metrics"
    figures_dir = project_root / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    confusion_path = figures_dir / "confusion_matrix.png"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save a reproducible, labeled confusion matrix for the held-out test set.
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        test_labels,
        predictions,
        display_labels=["not good", "good"],
        values_format="d",
        cmap="Blues",
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="final-evaluation"):
        mlflow.set_tag("stage", "final_evaluation")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {confusion_path}")


if __name__ == "__main__":
    main()