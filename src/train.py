"""Train the retained legacy Wine baseline and save its CV-selected model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def build_classifier(run_config: dict, random_state: int):
    """Construct one configured classifier with a reproducible random state."""
    model_type = run_config["model_type"]
    hyperparameters = run_config["hyperparameters"]
    if model_type == "dummy":
        return DummyClassifier(**hyperparameters)
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=random_state, **hyperparameters)
    if model_type == "random_forest":
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **hyperparameters,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def calculate_metrics(classifier, features: pd.DataFrame, labels: pd.Series) -> dict:
    """Calculate binary classification metrics for a fitted classifier."""
    predictions = classifier.predict(features)
    positive_class_index = list(classifier.classes_).index(1)
    probabilities = classifier.predict_proba(features)[:, positive_class_index]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def main() -> None:
    """Run configured Wine CV experiments and save the best CV-F1 model."""
    project_root = Path(__file__).resolve().parents[1]
    params = yaml.safe_load((project_root / "params.yaml").read_text(encoding="utf-8"))
    train_params = params["train"]
    split_params = params["split"]
    data_params = params["data"]
    train_path = project_root / data_params["processed_dir"] / "train.csv"
    test_path = project_root / data_params["processed_dir"] / "test.csv"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_features = train_data.drop(columns="good")
    train_labels = train_data["good"].astype(int)
    test_features = test_data.drop(columns="good")
    test_labels = test_data["good"].astype(int)

    cross_validation = StratifiedKFold(
        n_splits=train_params["cv_folds"],
        shuffle=True,
        random_state=split_params["random_state"],
    )
    candidates = []

    for run_config in train_params["runs"]:
        classifier = build_classifier(run_config, split_params["random_state"])
        run_params = {
            "model_type": run_config["model_type"],
            "test_size": split_params["test_size"],
            "label_threshold": data_params["label_threshold"],
            "dataset_path": "data/processed/train.csv",
            "test_dataset_path": "data/processed/test.csv",
            "cv_folds": train_params["cv_folds"],
            "random_state": split_params["random_state"],
            **run_config["hyperparameters"],
        }

        # Legacy Wine training remains available, but hydraulic MLflow tracking
        # is the only active experiment-tracking path in this migrated project.
        cv_scores = cross_val_score(
            classifier,
            train_features,
            train_labels,
            cv=cross_validation,
            scoring="f1",
            n_jobs=1,
        )
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        classifier.fit(train_features, train_labels)
        test_metrics = calculate_metrics(classifier, test_features, test_labels)
        candidates.append(
            {
                "run_name": run_config["name"],
                "run_params": run_params,
                "cv_f1_mean": cv_mean,
                "classifier": classifier,
                "test_f1": test_metrics["f1"],
            }
        )
        print(
            f"{run_config['name']}: CV F1={cv_mean:.4f} "
            f"(+/- {cv_std:.4f}), test F1={test_metrics['f1']:.4f}"
        )

    # Select by cross-validation only; held-out scores never choose the model.
    best_candidate = max(candidates, key=lambda candidate: candidate["cv_f1_mean"])
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_candidate["classifier"], models_dir / "best_model.joblib")
    best_run = {
        "run_name": best_candidate["run_name"],
        "params": best_candidate["run_params"],
        "cv_f1_mean": best_candidate["cv_f1_mean"],
    }
    (models_dir / "best_run.json").write_text(
        json.dumps(best_run, indent=2),
        encoding="utf-8",
    )
    print(f"Best run by CV F1: {best_candidate['run_name']}")


if __name__ == "__main__":
    main()