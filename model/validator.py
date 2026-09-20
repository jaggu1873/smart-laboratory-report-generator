"""Deterministic structural and data validation of generated reports.

This validator performs only deterministic structural/data checks. It does
NOT perform any semantic or model-based validation (for example, judging
whether a conclusion is scientifically supported by the measurements).

Checks performed:
    * required report sections are present
    * required fields are non-empty
    * measurement values are valid, finite numbers
    * measurement units are non-empty and recognized
    * no duplicate measurement names
    * schema consistency (correct types across the report)
"""

from __future__ import annotations

import math
from numbers import Real

from .schemas import (
    LabReport,
    Measurement,
    REQUIRED_SECTIONS,
    ReportSection,
    SectionName,
    ValidationIssue,
    ValidationResult,
)
from .units import is_recognized_unit


def _check_required_sections(report: LabReport, issues: list[ValidationIssue]) -> None:
    present = {s.name for s in report.sections if isinstance(s, ReportSection)}
    for required in REQUIRED_SECTIONS:
        if required not in present:
            issues.append(
                ValidationIssue(
                    code="missing_section",
                    message=f"required section '{required.value}' is missing",
                )
            )


def _check_fields(report: LabReport, issues: list[ValidationIssue]) -> None:
    if not isinstance(report.title, str) or not report.title.strip():
        issues.append(
            ValidationIssue(code="empty_field", message="report title is empty")
        )
    for section in report.sections:
        if not isinstance(section, ReportSection):
            issues.append(
                ValidationIssue(
                    code="schema_error",
                    message=f"section is not a ReportSection: {section!r}",
                )
            )
            continue
        if not isinstance(section.name, SectionName):
            issues.append(
                ValidationIssue(
                    code="schema_error",
                    message=f"section name is not a SectionName: {section.name!r}",
                )
            )
        if not isinstance(section.content, str) or not section.content.strip():
            name = getattr(section.name, "value", section.name)
            issues.append(
                ValidationIssue(
                    code="empty_field",
                    message=f"section '{name}' has empty content",
                )
            )


def _check_measurements(report: LabReport, issues: list[ValidationIssue]) -> None:
    seen_names: set[str] = set()
    for m in report.measurements:
        if not isinstance(m, Measurement):
            issues.append(
                ValidationIssue(
                    code="schema_error",
                    message=f"measurement is not a Measurement: {m!r}",
                )
            )
            continue

        if not isinstance(m.name, str) or not m.name.strip():
            issues.append(
                ValidationIssue(
                    code="empty_field",
                    message="measurement has an empty name",
                )
            )
        else:
            key = m.name.strip().lower()
            if key in seen_names:
                issues.append(
                    ValidationIssue(
                        code="duplicate_measurement",
                        message=f"duplicate measurement name: '{m.name}'",
                    )
                )
            seen_names.add(key)

        # Reject bools (bool is a subclass of int) and non-real numeric types.
        if isinstance(m.value, bool) or not isinstance(m.value, Real):
            issues.append(
                ValidationIssue(
                    code="invalid_value",
                    message=f"measurement '{m.name}' has a non-numeric value: {m.value!r}",
                )
            )
        elif not math.isfinite(float(m.value)):
            issues.append(
                ValidationIssue(
                    code="invalid_value",
                    message=f"measurement '{m.name}' has a non-finite value: {m.value!r}",
                )
            )

        if not isinstance(m.unit, str):
            issues.append(
                ValidationIssue(
                    code="invalid_unit",
                    message=f"measurement '{m.name}' unit is not a string: {m.unit!r}",
                )
            )
        elif not is_recognized_unit(m.unit):
            issues.append(
                ValidationIssue(
                    code="invalid_unit",
                    message=f"measurement '{m.name}' has an unrecognized unit: '{m.unit}'",
                )
            )


def validate_report(report: LabReport) -> ValidationResult:
    """Validate a report against deterministic structural/data constraints.

    Args:
        report: The generated report to validate.

    Returns:
        A :class:`ValidationResult`. ``is_valid`` is ``True`` only when no
        issues were found.
    """
    if not isinstance(report, LabReport):
        return ValidationResult.failed(
            [ValidationIssue(code="schema_error", message="not a LabReport")]
        )

    issues: list[ValidationIssue] = []
    _check_required_sections(report, issues)
    _check_fields(report, issues)
    _check_measurements(report, issues)

    if issues:
        return ValidationResult.failed(issues)
    return ValidationResult.ok()
