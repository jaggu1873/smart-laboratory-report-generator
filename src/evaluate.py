"""Evaluate the selected legacy Wine baseline and write test artifacts."""

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
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
    """Write Wine test metrics and figures without MLflow tracking."""
    project_root = Path(__file__).resolve().parents[1]
    model = joblib.load(project_root / "models" / "best_model.joblib")
    train_data = pd.read_csv(project_root / "data" / "processed" / "train.csv")
    feature_names = train_data.drop(columns="good").columns
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
    feature_importance_path = metrics_dir / "feature_importance.csv"
    feature_importance_figure_path = figures_dir / "feature_importance.png"
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

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            raise ValueError(
                "Model feature importance count does not match the training "
                f"feature count ({len(importances)} != {len(feature_names)})"
            )
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=True, kind="stable")
        feature_importance.to_csv(feature_importance_path, index=False)

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.barh(
            feature_importance["feature"],
            feature_importance["importance"],
            color="steelblue",
        )
        axis.set_xlabel("Feature importance")
        axis.set_title("Random Forest Feature Importance")
        figure.tight_layout()
        figure.savefig(feature_importance_figure_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {confusion_path}")
    if hasattr(model, "feature_importances_"):
        print(f"Saved feature importance data to {feature_importance_path}")
        print(f"Saved feature importance plot to {feature_importance_figure_path}")


if __name__ == "__main__":
    main()