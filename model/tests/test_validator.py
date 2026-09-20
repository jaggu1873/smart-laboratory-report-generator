"""Unit tests for model.validator (deterministic structural checks)."""

from __future__ import annotations

import unittest

from model.schemas import (
    LabReport,
    Measurement,
    ReportSection,
    ReportStatus,
    SectionName,
)
from model.validator import validate_report


def _valid_report() -> LabReport:
    return LabReport(
        title="Valid Experiment",
        sections=[
            ReportSection(SectionName.OBJECTIVE, "The objective is to test."),
            ReportSection(SectionName.PROCEDURE, "1. Do the thing."),
            ReportSection(SectionName.RESULTS, "- mass: 1.0 g"),
            ReportSection(SectionName.CONCLUSION, "It worked."),
        ],
        measurements=[Measurement("mass", 1.0, "g")],
        status=ReportStatus.GENERATED,
    )


class ValidatorTests(unittest.TestCase):
    def test_valid_report_passes(self) -> None:
        result = validate_report(_valid_report())
        self.assertTrue(result.is_valid, msg=str([i.message for i in result.issues]))
        self.assertEqual(result.issues, [])

    def test_missing_required_section(self) -> None:
        report = _valid_report()
        report.sections = [s for s in report.sections if s.name != SectionName.RESULTS]
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "missing_section" for i in result.issues))

    def test_empty_title(self) -> None:
        report = _valid_report()
        report.title = "   "
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "empty_field" for i in result.issues))

    def test_empty_section_content(self) -> None:
        report = _valid_report()
        report.sections[0].content = ""
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "empty_field" for i in result.issues))

    def test_non_finite_measurement_value(self) -> None:
        report = _valid_report()
        report.measurements = [Measurement("mass", float("inf"), "g")]
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "invalid_value" for i in result.issues))

    def test_non_numeric_measurement_value(self) -> None:
        report = _valid_report()
        report.measurements = [Measurement("mass", "heavy", "g")]  # type: ignore[arg-type]
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "invalid_value" for i in result.issues))

    def test_bool_is_rejected_as_value(self) -> None:
        report = _valid_report()
        report.measurements = [Measurement("flag", True, "g")]  # type: ignore[arg-type]
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "invalid_value" for i in result.issues))

    def test_invalid_unit(self) -> None:
        report = _valid_report()
        report.measurements = [Measurement("mass", 1.0, "furlongs")]
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "invalid_unit" for i in result.issues))

    def test_duplicate_measurement_name(self) -> None:
        report = _valid_report()
        report.measurements = [
            Measurement("mass", 1.0, "g"),
            Measurement("Mass", 2.0, "g"),
        ]
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "duplicate_measurement" for i in result.issues))

    def test_empty_measurement_name(self) -> None:
        report = _valid_report()
        report.measurements = [Measurement("  ", 1.0, "g")]
        result = validate_report(report)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "empty_field" for i in result.issues))

    def test_non_report_input(self) -> None:
        result = validate_report({"not": "a report"})  # type: ignore[arg-type]
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "schema_error" for i in result.issues))

    def test_empty_unit_is_allowed(self) -> None:
        report = _valid_report()
        report.measurements = [Measurement("count", 5.0, "")]
        result = validate_report(report)
        self.assertTrue(result.is_valid, msg=str([i.message for i in result.issues]))


if __name__ == "__main__":
    unittest.main()
