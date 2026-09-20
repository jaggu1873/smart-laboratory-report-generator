"""Structured input/output schemas for laboratory reports.

These dataclasses form the data contract between pipeline stages. Using
stdlib :mod:`dataclasses` keeps the model layer dependency-free; the
schemas are isolated enough to swap for Pydantic later if desired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReportStatus(str, Enum):
    """Lifecycle status of a generated report."""

    DRAFT = "draft"
    GENERATED = "generated"
    VALIDATED = "validated"
    REJECTED = "rejected"


class SectionName(str, Enum):
    """Canonical section names for a laboratory report."""

    TITLE = "title"
    OBJECTIVE = "objective"
    MATERIALS = "materials"
    PROCEDURE = "procedure"
    RESULTS = "results"
    CONCLUSION = "conclusion"


#: Sections that every complete report must contain.
REQUIRED_SECTIONS: tuple[SectionName, ...] = (
    SectionName.OBJECTIVE,
    SectionName.PROCEDURE,
    SectionName.RESULTS,
    SectionName.CONCLUSION,
)


@dataclass
class Measurement:
    """A single numeric measurement taken during an experiment.

    Attributes:
        name: Human-readable name of the measured quantity.
        value: Numeric value of the measurement.
        unit: Unit of measure (e.g. ``"g"``, ``"mL"``, ``"C"``).
    """

    name: str
    value: float
    unit: str


@dataclass
class ExperimentInput:
    """Cleaned, normalized input describing a laboratory experiment.

    This is the structured output of :mod:`model.preprocessing` and the
    input to :mod:`model.generator`.

    Attributes:
        title: Title of the experiment.
        objective: What the experiment set out to determine.
        materials: List of materials/equipment used.
        procedure: Ordered list of procedure steps.
        measurements: List of :class:`Measurement` records.
        metadata: Free-form additional key/value context.
    """

    title: str
    objective: str
    materials: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ReportSection:
    """A single section of a generated laboratory report.

    Attributes:
        name: Canonical section name.
        content: Rendered textual content of the section.
    """

    name: SectionName
    content: str


@dataclass
class LabReport:
    """A generated laboratory report.

    Attributes:
        title: Report title.
        sections: Ordered list of report sections.
        measurements: Measurements carried through from the input, used by
            the validator for data-level checks.
        status: Current lifecycle status of the report.
        model_id: Identifier of the model/backend that produced the report.
    """

    title: str
    sections: list[ReportSection] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    status: ReportStatus = ReportStatus.DRAFT
    model_id: str = ""

    def get_section(self, name: SectionName) -> Optional[ReportSection]:
        """Return the first section matching ``name``, or ``None``."""
        for section in self.sections:
            if section.name == name:
                return section
        return None


@dataclass
class ValidationIssue:
    """A single problem found while validating a report.

    Attributes:
        code: Machine-readable issue category.
        message: Human-readable description of the problem.
    """

    code: str
    message: str


@dataclass
class ValidationResult:
    """Outcome of validating a :class:`LabReport`.

    Attributes:
        is_valid: ``True`` when no issues were found.
        issues: List of :class:`ValidationIssue` records.
    """

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "ValidationResult":
        """Return a passing result with no issues."""
        return cls(is_valid=True, issues=[])

    @classmethod
    def failed(cls, issues: list[ValidationIssue]) -> "ValidationResult":
        """Return a failing result carrying the given ``issues``."""
        return cls(is_valid=False, issues=issues)
