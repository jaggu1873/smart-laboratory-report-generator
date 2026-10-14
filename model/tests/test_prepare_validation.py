"""Unit tests for raw Wine Quality data validation."""

from __future__ import annotations

import importlib.util
import unittest

DATA_VALIDATION_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("pandas", "yaml")
)

if DATA_VALIDATION_DEPENDENCIES_AVAILABLE:
    import pandas as pd

    from src.prepare import EXPECTED_COLUMNS, validate_data
else:
    pd = None
    EXPECTED_COLUMNS = ()
    validate_data = None


@unittest.skipUnless(
    DATA_VALIDATION_DEPENDENCIES_AVAILABLE,
    "pandas and PyYAML are required for raw-data validation tests",
)
class PrepareValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            [[7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 5]],
            columns=EXPECTED_COLUMNS,
        )

    def test_valid_data_passes(self) -> None:
        validate_data(self.data)

    def test_missing_column_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing expected columns: quality"):
            validate_data(self.data.drop(columns="quality"))

    def test_null_value_is_reported(self) -> None:
        data = self.data.copy()
        data.loc[0, "alcohol"] = float("nan")

        with self.assertRaisesRegex(ValueError, "null values in columns: alcohol"):
            validate_data(data)

    def test_out_of_range_values_are_reported(self) -> None:
        data = self.data.copy()
        data.loc[0, "pH"] = -1
        data.loc[0, "alcohol"] = -0.1
        data.loc[0, "quality"] = 11

        with self.assertRaises(ValueError) as context:
            validate_data(data)

        message = str(context.exception)
        self.assertIn("pH values must be 0 to 14", message)
        self.assertIn("alcohol values must be >= 0", message)
        self.assertIn("quality values must be 0 to 10", message)

    def test_non_numeric_values_are_reported(self) -> None:
        data = self.data.copy()
        data["alcohol"] = "not numeric"

        with self.assertRaisesRegex(ValueError, "non-numeric columns: alcohol"):
            validate_data(data)


if __name__ == "__main__":
    unittest.main()
