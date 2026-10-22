"""Unit tests for the isolated hydraulic data, model, split, and report code."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.schemas import SectionName
from model.validator import validate_report
from src.hydraulic.artifacts import (
    create_run_directories,
    split_membership_frame,
    validate_split_membership_file,
)
from src.hydraulic.comparison import (
    BLOCKED_FOLDS,
    blocked_temporal_folds,
    comparison_pipeline,
    select_comparison_candidate,
)
from src.hydraulic.data import (
    FEATURE_STATISTICS,
    PHYSICAL_SENSOR_NAMES,
    PROFILE_COLUMNS,
    PROFILE_VALUE_COUNTS,
    SENSOR_SPECS,
    DatasetIntegrityError,
    HydraulicDataset,
    extract_sensor_features,
    feature_name,
    load_sensor_matrix,
    make_cycle_ids,
    map_pump_leakage_grade,
    validate_profile,
)
from src.hydraulic.evaluate import select_evaluation_partition
from src.hydraulic.modeling import (
    build_candidate_models,
    calculate_binary_metrics,
    select_probability_threshold,
)
from src.hydraulic.report import (
    build_cycle_report,
    comparison_cycle_split,
    serialize_report,
)
from src.hydraulic.splitting import (
    chronological_splits,
    validate_no_cross_split_duplicates,
)


class HydraulicDataTests(unittest.TestCase):
    def test_sensor_file_loading_validates_dimensions_and_finiteness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sensor.txt"
            path.write_text("1\t2\t3\n4\t5\t6\n", encoding="utf-8")
            values = load_sensor_matrix(path, expected_rows=2, expected_columns=3)
            np.testing.assert_array_equal(values, [[1, 2, 3], [4, 5, 6]])
            with self.assertRaises(DatasetIntegrityError):
                load_sensor_matrix(path, expected_rows=3, expected_columns=3)
            path.write_text("1\t2\n4\t5\n", encoding="utf-8")
            with self.assertRaises(DatasetIntegrityError):
                load_sensor_matrix(path, expected_rows=2, expected_columns=3)
            path.write_text("1\tNaN\n", encoding="utf-8")
            with self.assertRaises(DatasetIntegrityError):
                load_sensor_matrix(path, expected_rows=1, expected_columns=2)

    def test_cycle_alignment_uses_equal_positional_row_counts(self) -> None:
        cycle_ids = make_cycle_ids({"PS1": 3, "FS1": 3}, profile_rows=3)
        np.testing.assert_array_equal(cycle_ids, [1, 2, 3])
        with self.assertRaises(DatasetIntegrityError):
            make_cycle_ids({"PS1": 3, "FS1": 2}, profile_rows=3)

    def test_binary_target_mapping_preserves_grade_semantics(self) -> None:
        grades = np.array([0, 1, 2, 0, 2])
        np.testing.assert_array_equal(map_pump_leakage_grade(grades), [0, 1, 1, 0, 1])
        np.testing.assert_array_equal(grades, [0, 1, 2, 0, 2])
        with self.assertRaises(ValueError):
            map_pump_leakage_grade(np.array([0, 3]))

    def test_feature_extraction_has_stable_dimension_and_finite_values(self) -> None:
        matrix = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = extract_sensor_features(matrix)
        self.assertEqual(tuple(result), FEATURE_STATISTICS)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(values.shape == (2,) for values in result.values()))
        self.assertTrue(all(np.isfinite(values).all() for values in result.values()))
        self.assertAlmostEqual(float(result["std"][0]), np.std(matrix[0], ddof=0))
        with self.assertRaises(ValueError):
            extract_sensor_features(np.array([[1.0, np.inf]]))

    def test_profile_distribution_and_schema_are_strict(self) -> None:
        columns = [
            np.repeat(list(counts), list(counts.values()))
            for counts in PROFILE_VALUE_COUNTS
        ]
        profile = np.column_stack(columns)
        validate_profile(profile)
        invalid = profile.copy()
        invalid[0, 2] = 3
        with self.assertRaises(DatasetIntegrityError):
            validate_profile(invalid)
        with self.assertRaises(DatasetIntegrityError):
            validate_profile(profile[:, :4])

    def test_model_feature_view_excludes_profile_targets_and_virtual_channels(self) -> None:
        all_feature_names = [
            feature_name(sensor_name, statistic)
            for sensor_name in SENSOR_SPECS
            for statistic in FEATURE_STATISTICS
        ]
        features = pd.DataFrame(
            np.ones((2, len(all_feature_names))),
            index=pd.Index([1, 2], name="cycle_id"),
            columns=all_feature_names,
        )
        profile = pd.DataFrame(
            np.zeros((2, len(PROFILE_COLUMNS))),
            index=features.index,
            columns=PROFILE_COLUMNS,
        )
        dataset = HydraulicDataset(
            cycle_ids=np.array([1, 2]),
            features=features,
            profile_targets=profile,
            pump_leakage_grade=pd.Series([0, 1], index=features.index),
            pump_leakage_binary=pd.Series([0, 1], index=features.index),
            archive_sha256="fixture",
        )
        model_features = dataset.model_features()
        self.assertEqual(len(model_features.columns), len(PHYSICAL_SENSOR_NAMES) * 8)
        self.assertTrue(
            all(name.startswith(PHYSICAL_SENSOR_NAMES) for name in model_features.columns)
        )
        self.assertFalse(set(PROFILE_COLUMNS) & set(model_features.columns))
        self.assertFalse(any(name.startswith(("CE__", "CP__", "SE__")) for name in model_features))


class HydraulicModelTests(unittest.TestCase):
    def test_chronological_split_is_reproducible_and_purges_boundaries(self) -> None:
        first = chronological_splits(2205)
        second = chronological_splits(2205)
        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.validation, second.validation)
        np.testing.assert_array_equal(first.test, second.test)
        self.assertEqual((first.train[0], first.train[-1]), (0, 1482))
        self.assertEqual((first.validation[0], first.validation[-1]), (1603, 1813))
        self.assertEqual((first.test[0], first.test[-1]), (1934, 2204))
        self.assertEqual(len(first.purged), 240)
        self.assertLess(first.train[-1], first.validation[0])
        self.assertLess(first.validation[-1], first.test[0])

    def test_saved_split_membership_is_checked_against_cycle_ids(self) -> None:
        splits = chronological_splits(20, embargo_cycles=0)
        cycle_ids = np.arange(1, 21, dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "split_membership.csv"
            expected = split_membership_frame(cycle_ids, splits)
            expected.to_csv(path, index=False)
            validate_split_membership_file(path, cycle_ids, splits)
            corrupted = expected.copy()
            corrupted.loc[
                corrupted["split"] == "test", "cycle_id_one_based"
            ] += 1
            corrupted.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_split_membership_file(path, cycle_ids, splits)

    def test_evaluation_selects_labels_and_features_from_the_test_partition(self) -> None:
        splits = chronological_splits(20, embargo_cycles=0)
        cycle_ids = np.arange(1, 21, dtype=np.int64)
        features = pd.DataFrame(
            {"sensor__mean": cycle_ids.astype(float)},
            index=pd.Index(cycle_ids, name="cycle_id"),
        )
        labels = pd.Series(
            (cycle_ids >= 18).astype(int),
            index=features.index,
            name="pump_leakage_binary",
        )
        indices, test_features, test_labels = select_evaluation_partition(
            features=features,
            labels=labels,
            cycle_ids=cycle_ids,
            splits=splits,
            split_name="test",
        )
        np.testing.assert_array_equal(cycle_ids[indices], [18, 19, 20])
        np.testing.assert_array_equal(test_features.index.to_numpy(), [18, 19, 20])
        np.testing.assert_array_equal(test_labels.to_numpy(), [1, 1, 1])

    def test_exact_duplicate_cycles_cannot_cross_partitions(self) -> None:
        splits = chronological_splits(2205)
        values = np.arange(2205 * 2, dtype=float).reshape(2205, 2)
        validate_no_cross_split_duplicates(values, splits)
        values[1603] = values[0]
        with self.assertRaisesRegex(ValueError, "cross the train/validation"):
            validate_no_cross_split_duplicates(values, splits)

    def test_preprocessing_is_fitted_on_training_partition_only(self) -> None:
        splits = chronological_splits(2205)
        features = pd.DataFrame(
            {"sensor__mean": np.arange(2205, dtype=float),
             "sensor__std": np.arange(2205, dtype=float) * 2}
        )
        labels = (np.arange(2205) % 2).astype(int)
        estimator = build_candidate_models()["logistic_regression"]
        estimator.fit(features.iloc[splits.train], labels[splits.train])
        scaler_mean = estimator.named_steps["scale"].mean_
        np.testing.assert_allclose(scaler_mean, features.iloc[splits.train].mean().values)
        self.assertFalse(
            np.allclose(scaler_mean, features.mean().values),
            "the held-out cycles must not influence fitted preprocessing",
        )

    def test_model_results_are_reproducible_for_fixed_seed(self) -> None:
        rng = np.random.default_rng(7)
        features = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
        labels = np.tile([0, 1], 40)
        first = build_candidate_models(random_state=42)["logistic_regression"]
        second = build_candidate_models(random_state=42)["logistic_regression"]
        first.fit(features, labels)
        second.fit(features, labels)
        np.testing.assert_array_equal(first.predict(features), second.predict(features))
        np.testing.assert_allclose(first.predict_proba(features), second.predict_proba(features))

    def test_blocked_comparison_folds_are_forward_and_embargoed(self) -> None:
        folds = blocked_temporal_folds(np.arange(1483))
        self.assertEqual(len(folds), len(BLOCKED_FOLDS))
        for (train_indices, validation_indices), (start, stop) in zip(
            folds, BLOCKED_FOLDS, strict=True
        ):
            self.assertEqual(int(validation_indices[0]), start)
            self.assertEqual(int(validation_indices[-1]) + 1, stop)
            self.assertEqual(int(train_indices[-1]), start - 61)
            self.assertLess(int(train_indices[-1]), int(validation_indices[0]))

    def test_comparison_estimators_are_reproducible_and_pipeline_wrapped(self) -> None:
        rng = np.random.default_rng(19)
        features = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        labels = np.tile([0, 1], 30)
        configurations = {
            "logistic_regression": {"C": 0.1},
            "random_forest": {"max_depth": 8, "min_samples_leaf": 5},
            "hist_gradient_boosting": {
                "max_iter": 20,
                "learning_rate": 0.1,
                "max_leaf_nodes": 7,
            },
        }
        for family, parameters in configurations.items():
            first = comparison_pipeline(family, parameters, random_state=73)
            second = comparison_pipeline(family, parameters, random_state=73)
            self.assertIsInstance(first, Pipeline)
            first.fit(features, labels)
            second.fit(features, labels)
            np.testing.assert_array_equal(first.predict(features), second.predict(features))

    def test_comparison_selection_is_deterministic_and_validation_based(self) -> None:
        rows = [
            {
                "family": "logistic_regression",
                "blocked_cv": {
                    "mean_balanced_accuracy": 0.7,
                    "std_balanced_accuracy": 0.1,
                },
                "validation_balanced_accuracy_threshold_0_5": 0.5,
                "validation_roc_auc": 0.9,
            },
            {
                "family": "random_forest",
                "blocked_cv": {
                    "mean_balanced_accuracy": 0.9,
                    "std_balanced_accuracy": 0.15,
                },
                "validation_balanced_accuracy_threshold_0_5": 0.8,
                "validation_roc_auc": 0.99,
            },
        ]
        self.assertEqual(select_comparison_candidate(rows), "random_forest")
        self.assertEqual(select_comparison_candidate(rows), "random_forest")

    def test_comparison_cycle_membership_uses_saved_ranges(self) -> None:
        distributions = {
            "train": {"cycle_id_range": [1, 1483]},
            "validation": {"cycle_id_range": [1604, 1814]},
            "test": {"cycle_id_range": [1935, 2205]},
        }
        self.assertEqual(comparison_cycle_split(2005, distributions), "test")
        self.assertEqual(comparison_cycle_split(1500, distributions), "purged")
        with self.assertRaisesRegex(ValueError, "positive"):
            comparison_cycle_split(0, distributions)

    def test_metric_calculation_includes_valid_roc_auc(self) -> None:
        class FixedEstimator:
            classes_ = np.array([0, 1])

            def predict(self, _features: pd.DataFrame) -> np.ndarray:
                return np.array([0, 1, 1, 0])[: len(_features)]

            def predict_proba(self, _features: pd.DataFrame) -> np.ndarray:
                return np.array(
                    [[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3]]
                )[: len(_features)]

        metrics = calculate_binary_metrics(
            FixedEstimator(),
            pd.DataFrame({"x": [1, 2, 3, 4]}),
            np.array([0, 0, 1, 1]),
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["class_counts"], {"0": 2, "1": 2})
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)

    def test_probability_threshold_is_selected_on_validation_data(self) -> None:
        class ProbabilityEstimator:
            classes_ = np.array([0, 1])

            def predict_proba(self, _features: pd.DataFrame) -> np.ndarray:
                return np.array(
                    [[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]]
                )[: len(_features)]

        threshold = select_probability_threshold(
            ProbabilityEstimator(),
            pd.DataFrame({"x": [1, 2, 3, 4]}),
            np.array([0, 0, 1, 1]),
        )
        self.assertAlmostEqual(threshold, 0.5)

    def test_extreme_threshold_selection_uses_aligned_validation_probabilities(self) -> None:
        class ExtremeProbabilityEstimator:
            classes_ = np.array([0, 1])

            def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
                return np.array(
                    [
                        [1 - 1e-53, 1e-53],
                        [1 - 1e-51, 1e-51],
                        [1 - 2e-50, 2e-50],
                        [1 - 4e-50, 4e-50],
                    ]
                )[: len(features)]

        features = pd.DataFrame(
            {"x": [1, 2, 3, 4]},
            index=pd.Index([11, 12, 13, 14], name="cycle_id"),
        )
        labels = pd.Series([0, 0, 1, 1], index=features.index)
        threshold = select_probability_threshold(
            ExtremeProbabilityEstimator(), features, labels
        )
        self.assertAlmostEqual(threshold, 1.05e-50, delta=1e-64)
        metrics = calculate_binary_metrics(
            ExtremeProbabilityEstimator(),
            features,
            labels,
            threshold=threshold,
        )
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(metrics["decision_threshold"], threshold)

    def test_evaluation_rejects_misaligned_feature_and_label_cycle_ids(self) -> None:
        class FixedEstimator:
            classes_ = np.array([0, 1])

            def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
                return np.tile([[0.5, 0.5]], (len(features), 1))

        features = pd.DataFrame(
            {"x": [1, 2]}, index=pd.Index([1, 2], name="cycle_id")
        )
        misaligned_labels = pd.Series(
            [0, 1], index=pd.Index([2, 1], name="cycle_id")
        )
        with self.assertRaisesRegex(ValueError, "cycle indexes do not match"):
            calculate_binary_metrics(FixedEstimator(), features, misaligned_labels)

    def test_run_directories_refuse_to_overwrite_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {
                "model_dir": root / "model" / "run-a",
                "metrics_dir": root / "metrics" / "run-a",
            }
            create_run_directories(paths)
            sentinel = paths["model_dir"] / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_run_directories(paths)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


class HydraulicReportTests(unittest.TestCase):
    def test_report_contains_provenance_metrics_and_gated_ground_truth(self) -> None:
        rng = np.random.default_rng(11)
        names = [
            f"{sensor}__{statistic}"
            for sensor in PHYSICAL_SENSOR_NAMES
            for statistic in FEATURE_STATISTICS
        ]
        model_features = pd.DataFrame(rng.normal(size=(12, len(names))), columns=names)
        labels = np.tile([0, 1], 6)
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        ).fit(model_features, labels)
        all_names = [
            feature_name(sensor, statistic)
            for sensor in SENSOR_SPECS
            for statistic in FEATURE_STATISTICS
        ]
        features = pd.DataFrame(
            rng.normal(size=(2, len(all_names))),
            index=pd.Index([1, 2], name="cycle_id"),
            columns=all_names,
        )
        targets = pd.DataFrame(
            np.zeros((2, len(PROFILE_COLUMNS)), dtype=int),
            index=features.index,
            columns=PROFILE_COLUMNS,
        )
        dataset = HydraulicDataset(
            cycle_ids=np.array([1, 2]),
            features=features,
            profile_targets=targets,
            pump_leakage_grade=pd.Series([0, 2], index=features.index),
            pump_leakage_binary=pd.Series([0, 1], index=features.index),
            archive_sha256="verified-fixture-sha256",
        )
        evaluation = {
            "splits": {
                "test": {
                    "cycle_id_start": 1935,
                    "cycle_id_end": 2205,
                    "selected_model": {
                        "accuracy": 0.7,
                        "precision": 0.75,
                        "recall": 0.75,
                        "balanced_accuracy": 0.7,
                        "f1": 0.75,
                        "roc_auc": 0.8,
                        "class_counts": {"0": 2, "1": 2},
                        "confusion_matrix": [[1, 1], [1, 1]],
                    }
                }
            }
        }
        report = build_cycle_report(
            dataset=dataset,
            model=model,
            evaluation=evaluation,
            cycle_id=2,
            include_ground_truth=True,
            model_name="Random Forest",
            model_version="hydraulic-comparison-v1:random_forest",
            validation_method="four expanding blocked folds",
            include_probability=False,
            comparison_id="comparison-fixture",
        )
        self.assertTrue(validate_report(report).is_valid)
        payload = serialize_report(report)
        serialized = json.dumps(payload)
        self.assertEqual(payload["metadata"]["dataset_doi"], "10.24432/C5CW21")
        self.assertEqual(
            payload["metadata"]["project_name"],
            "Smart Laboratory Report Generator",
        )
        self.assertEqual(
            payload["metadata"]["archive_sha256"],
            "verified-fixture-sha256",
        )
        self.assertEqual(payload["metadata"]["model_name"], "Random Forest")
        self.assertEqual(
            payload["metadata"]["model_version"],
            "hydraulic-comparison-v1:random_forest",
        )
        self.assertEqual(
            payload["metadata"]["comparison_id"], "comparison-fixture"
        )
        for required in (
            "cycle 2",
            "10.24432/C5CW21",
            "10.1109/I2MTC.2015.7151267",
            "Positive class definition: label 1 = pump leakage present",
            "Model: Random Forest (hydraulic-comparison-v1:random_forest)",
            "Validation method: four expanding blocked folds",
            "Ground-truth pump-leakage class",
            "not direct laboratory measurements",
        ):
            self.assertIn(required, serialized)
        self.assertNotIn("Estimated model probability for predicted class", serialized)
        self.assertIn(
            "Ground-truth pump-leakage class",
            payload["sections"][3]["content"],
        )
        self.assertEqual(len(payload["measurements"]), 17 * len(FEATURE_STATISTICS))

    def test_report_omits_ground_truth_without_explicit_retrospective_option(self) -> None:
        rng = np.random.default_rng(12)
        names = [
            f"{sensor}__{statistic}"
            for sensor in PHYSICAL_SENSOR_NAMES
            for statistic in FEATURE_STATISTICS
        ]
        training = pd.DataFrame(rng.normal(size=(12, len(names))), columns=names)
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        ).fit(training, np.tile([0, 1], 6))
        all_names = [
            feature_name(sensor, statistic)
            for sensor in SENSOR_SPECS
            for statistic in FEATURE_STATISTICS
        ]
        features = pd.DataFrame(
            np.ones((1, len(all_names))),
            index=pd.Index([1], name="cycle_id"),
            columns=all_names,
        )
        targets = pd.DataFrame(np.zeros((1, 5)), index=features.index, columns=PROFILE_COLUMNS)
        dataset = HydraulicDataset(
            cycle_ids=np.array([1]),
            features=features,
            profile_targets=targets,
            pump_leakage_grade=pd.Series([1], index=features.index),
            pump_leakage_binary=pd.Series([1], index=features.index),
            archive_sha256="fixture",
        )
        evaluation = {
            "splits": {
                "test": {
                    "cycle_id_start": 1935,
                    "cycle_id_end": 2205,
                    "selected_model": {
                        "accuracy": 0.5,
                        "precision": 0.5,
                        "recall": 0.5,
                        "balanced_accuracy": 0.5,
                        "f1": 0.5,
                        "roc_auc": None,
                        "class_counts": {"0": 1, "1": 1},
                        "confusion_matrix": [[1, 0], [1, 0]],
                    }
                }
            }
        }
        report = build_cycle_report(
            dataset=dataset,
            model=model,
            evaluation=evaluation,
            cycle_id=1,
        )
        payload = serialize_report(report)
        self.assertNotIn("ground_truth_pump_leakage_binary", payload["sections"][3]["content"])
        self.assertIn(SectionName.RESULTS.value, [section["name"] for section in payload["sections"]])


if __name__ == "__main__":
    unittest.main()
