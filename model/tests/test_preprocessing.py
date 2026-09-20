"""Unit tests for model.preprocessing."""

from __future__ import annotations

import unittest

from model.preprocessing import PreprocessingError, preprocess
from model.schemas import ExperimentInput, Measurement


class PreprocessTests(unittest.TestCase):
    def test_minimal_valid_input(self) -> None:
        result = preprocess({"title": "Titration", "objective": "measure pH"})
        self.assertIsInstance(result, ExperimentInput)
        self.assertEqual(result.title, "Titration")
        self.assertEqual(result.objective, "measure pH")
        self.assertEqual(result.materials, [])
        self.assertEqual(result.measurements, [])

    def test_whitespace_is_collapsed(self) -> None:
        result = preprocess({"title": "  Acid   Test \n", "objective": "  find\tvalue "})
        self.assertEqual(result.title, "Acid Test")
        self.assertEqual(result.objective, "find value")

    def test_missing_required_field_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            preprocess({"title": "No objective"})

    def test_empty_required_field_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            preprocess({"title": "   ", "objective": "x"})

    def test_non_dict_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            preprocess(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_materials_from_newline_string(self) -> None:
        result = preprocess(
            {"title": "t", "objective": "o", "materials": "beaker\n\nflask\n"}
        )
        self.assertEqual(result.materials, ["beaker", "flask"])

    def test_materials_from_list_drops_empties(self) -> None:
        result = preprocess(
            {"title": "t", "objective": "o", "materials": ["a", "  ", "b"]}
        )
        self.assertEqual(result.materials, ["a", "b"])

    def test_measurement_unit_normalization(self) -> None:
        result = preprocess(
            {
                "title": "t",
                "objective": "o",
                "measurements": [
                    {"name": "mass", "value": "12.5", "unit": "grams"},
                    {"name": "temp", "value": 20, "unit": "Celsius"},
                    {"name": "vol", "value": 5, "unit": "ml"},
                ],
            }
        )
        self.assertEqual(
            result.measurements,
            [
                Measurement("mass", 12.5, "g"),
                Measurement("temp", 20.0, "C"),
                Measurement("vol", 5.0, "mL"),
            ],
        )

    def test_unparseable_measurement_value_is_skipped(self) -> None:
        result = preprocess(
            {
                "title": "t",
                "objective": "o",
                "measurements": [
                    {"name": "good", "value": "1.0", "unit": "g"},
                    {"name": "bad", "value": "not-a-number", "unit": "g"},
                ],
            }
        )
        self.assertEqual(len(result.measurements), 1)
        self.assertEqual(result.measurements[0].name, "good")

    def test_bad_measurement_container_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            preprocess({"title": "t", "objective": "o", "measurements": {"x": 1}})

    def test_bad_metadata_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            preprocess({"title": "t", "objective": "o", "metadata": "nope"})

    def test_metadata_cleaned(self) -> None:
        result = preprocess(
            {"title": "t", "objective": "o", "metadata": {" author ": " Ada "}}
        )
        self.assertEqual(result.metadata, {"author": "Ada"})


if __name__ == "__main__":
    unittest.main()
