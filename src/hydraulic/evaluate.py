"""Re-evaluate an existing hydraulic run without fitting on held-out data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .artifacts import run_paths, validate_run_id, validate_split_membership_file
from .data import (
    FEATURE_RECIPE_VERSION,
    PHYSICAL_SENSOR_NAMES,
    load_hydraulic_dataset,
)
from .modeling import build_candidate_models, calculate_binary_metrics
from .splitting import (
    CycleSplits,
    chronological_splits,
    validate_no_cross_split_duplicates,
)


def select_evaluation_partition(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    cycle_ids: np.ndarray,
    splits: CycleSplits,
    split_name: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """Return one fixed partition after checking row and label identity."""
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"unknown evaluation split: {split_name}")
    ids = np.asarray(cycle_ids)
    if len(features) != len(labels) or len(features) != len(ids):
        raise ValueError("features, labels, and cycle IDs must have equal row counts")
    if not np.array_equal(features.index.to_numpy(), ids):
        raise ValueError("feature index does not match source cycle IDs")
    if not labels.index.equals(features.index):
        raise ValueError("feature and label cycle indexes do not match")
    indices = np.asarray(getattr(splits, split_name), dtype=np.int64)
    selected_features = features.iloc[indices]
    selected_labels = labels.iloc[indices]
    if not selected_features.index.equals(selected_labels.index):
        raise ValueError(f"{split_name} labels are not aligned with feature cycles")
    return indices, selected_features, selected_labels


def evaluate_run(
    *,
    run_id: str,
    data_dir: Path,
    project_root: Path,
    audit_id: str | None = None,
) -> dict[str, Any]:
    """Verify saved provenance and compute metrics for all fixed partitions."""
    paths = run_paths(project_root, run_id)
    model_path = paths["model_dir"] / "pipeline.joblib"
    metadata_path = paths["model_dir"] / "training_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"no complete hydraulic model run found for {run_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = load_hydraulic_dataset(data_dir)
    if metadata["archive_sha256"] != dataset.archive_sha256:
        raise ValueError("current source archive differs from the model's recorded source")
    if metadata["feature_recipe_version"] != FEATURE_RECIPE_VERSION:
        raise ValueError("saved model feature recipe does not match this implementation")
    model = joblib.load(model_path)
    expected_names = [
        f"{sensor}__{statistic}"
        for sensor in PHYSICAL_SENSOR_NAMES
        for statistic in ("mean", "std", "min", "max", "rms", "p25", "median", "p75")
    ]
    if metadata["model_feature_names"] != expected_names:
        raise ValueError("saved model metadata has unexpected hydraulic feature names")
    model_features = dataset.model_features()
    splits = chronological_splits(len(dataset.cycle_ids))
    membership_path = paths["processed_dir"] / "split_membership.csv"
    validate_split_membership_file(membership_path, dataset.cycle_ids, splits)
    manifest_path = paths["model_dir"] / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ranges = {
        "train_cycle_id_inclusive": [
            int(dataset.cycle_ids[splits.train[0]]),
            int(dataset.cycle_ids[splits.train[-1]]),
        ],
        "validation_cycle_id_inclusive": [
            int(dataset.cycle_ids[splits.validation[0]]),
            int(dataset.cycle_ids[splits.validation[-1]]),
        ],
        "test_cycle_id_inclusive": [
            int(dataset.cycle_ids[splits.test[0]]),
            int(dataset.cycle_ids[splits.test[-1]]),
        ],
    }
    for key, expected in expected_ranges.items():
        if manifest.get(key) != expected:
            raise ValueError(f"saved split manifest has an unexpected {key}")
    if manifest.get("purged_cycle_count") != len(splits.purged):
        raise ValueError("saved split manifest has an unexpected purge count")
    validate_no_cross_split_duplicates(
        model_features.to_numpy(), splits
    )
    if list(model.feature_names_in_) != expected_names:
        raise ValueError("saved estimator feature names/order do not match the source")
    fitted_scaler = getattr(model, "named_steps", {}).get("scale")
    if fitted_scaler is not None and hasattr(fitted_scaler, "mean_"):
        training_feature_means = model_features.iloc[splits.train].mean().to_numpy()
        if not np.allclose(
            fitted_scaler.mean_,
            training_feature_means,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("saved preprocessing statistics do not match training cycles")
    labels = dataset.pump_leakage_binary
    dummy = build_candidate_models(int(metadata["random_state"]))["majority_dummy"]
    dummy.fit(model_features.iloc[splits.train], labels.iloc[splits.train])
    results: dict[str, Any] = {
        "run_id": run_id,
        "archive_sha256": dataset.archive_sha256,
        "selected_model": metadata["estimator"],
        "positive_class_definition": (
            "label 1 = pump leakage present (original pump grades 1 or 2)"
        ),
        "decision_threshold": float(metadata["validation_threshold"]),
        "test_used_for_model_or_threshold_selection": False,
        "splits": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        indices, split_features, split_labels = select_evaluation_partition(
            features=model_features,
            labels=labels,
            cycle_ids=dataset.cycle_ids,
            splits=splits,
            split_name=split_name,
        )
        threshold = float(metadata["validation_threshold"])
        results["splits"][split_name] = {
            "cycle_id_start": int(dataset.cycle_ids[indices[0]]),
            "cycle_id_end": int(dataset.cycle_ids[indices[-1]]),
            "selected_model": calculate_binary_metrics(
                model,
                split_features,
                split_labels,
                threshold=threshold,
            ),
            "majority_dummy_baseline": calculate_binary_metrics(
                dummy, split_features, split_labels
            ),
        }
        probabilities: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            classes = np.asarray(model.classes_)
            positive_indexes = np.flatnonzero(classes == 1)
            if len(positive_indexes) == 1:
                probabilities = np.asarray(model.predict_proba(split_features))[
                    :, int(positive_indexes[0])
                ]
        predictions = (
            (probabilities >= threshold).astype(np.int64)
            if probabilities is not None
            else np.asarray(model.predict(split_features), dtype=np.int64)
        )
        if predictions.shape != (len(indices),):
            raise ValueError(f"{split_name} predictions do not align with cycle IDs")
        if audit_id is not None:
            for offset, row_index in enumerate(indices):
                prediction_rows.append(
                    {
                        "split": split_name,
                        "row_position_zero_based": int(row_index),
                        "cycle_id_one_based": int(dataset.cycle_ids[row_index]),
                        "true_pump_leakage_binary": int(split_labels.iloc[offset]),
                        "positive_class_probability": (
                            float(probabilities[offset])
                            if probabilities is not None
                            else None
                        ),
                        "predicted_pump_leakage_binary": int(predictions[offset]),
                        "decision_threshold": threshold,
                    }
                )

    saved_metrics_path = paths["metrics_dir"] / "metrics.json"
    saved_metrics = json.loads(saved_metrics_path.read_text(encoding="utf-8"))
    compatibility_notes: list[str] = []
    for split_name, split_result in results["splits"].items():
        for model_name in ("selected_model", "majority_dummy_baseline"):
            saved = saved_metrics["split_metrics"][split_name][model_name]
            actual = split_result[model_name]
            for key, actual_value in actual.items():
                if key == "decision_threshold" and key not in saved:
                    expected_threshold = (
                        float(metadata["validation_threshold"])
                        if model_name == "selected_model"
                        else 0.5
                    )
                    if not math.isclose(
                        float(actual_value), expected_threshold, abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"saved {split_name} {model_name} threshold metadata "
                            "does not match independent recomputation"
                        )
                    compatibility_notes.append(
                        f"{split_name}.{model_name}.decision_threshold is absent "
                        "from the historical metrics file; verified against the "
                        "run metadata/default baseline threshold"
                    )
                    continue
                saved_value = saved.get(key)
                if isinstance(actual_value, (float, int)) and not isinstance(
                    actual_value, bool
                ):
                    if not isinstance(saved_value, (float, int)) or not math.isclose(
                        float(saved_value),
                        float(actual_value),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"saved {split_name} {model_name} metric {key} "
                            "does not match independent recomputation"
                        )
                elif saved_value != actual_value:
                    raise ValueError(
                        f"saved {split_name} {model_name} metric {key} "
                        "does not match independent recomputation"
                    )
    results["saved_metrics_match_recomputation"] = True
    results["saved_metrics_compatibility_notes"] = sorted(set(compatibility_notes))
    if audit_id is not None:
        safe_audit_id = validate_run_id(audit_id)
        audit_dir = (
            project_root / "metrics" / "hydraulic" / "audits" / safe_audit_id
        )
        audit_dir.mkdir(parents=True, exist_ok=False)
        results["audit_id"] = safe_audit_id
        results["prediction_count_saved"] = len(prediction_rows)
        (audit_dir / "evaluation_audit.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pd.DataFrame(prediction_rows).to_csv(
            audit_dir / "cycle_predictions.csv", index=False
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--audit-id",
        help="write a separate, collision-safe evaluation and prediction audit",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[2]
    data_dir = args.data_dir or project_root / "data" / "inspection" / (
        "uci-hydraulic-condition-monitoring"
    ) / "extracted"
    print(
        json.dumps(
            evaluate_run(
                run_id=args.run_id,
                data_dir=data_dir,
                project_root=project_root,
                audit_id=args.audit_id,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
