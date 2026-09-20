"""Smart Laboratory Report Generator — model layer.

Public API for generating and validating laboratory reports. The underlying
model is pluggable via the :class:`ModelBackend` abstraction; the default
:class:`MockModelBackend` is deterministic and offline.
"""

from __future__ import annotations

from .config import ModelConfig
from .generator import DEFAULT_SECTIONS, LabReportGenerator
from .inference import MockModelBackend, ModelBackend, build_backend
from .preprocessing import PreprocessingError, preprocess
from .schemas import (
    ExperimentInput,
    LabReport,
    Measurement,
    REQUIRED_SECTIONS,
    ReportSection,
    ReportStatus,
    SectionName,
    ValidationIssue,
    ValidationResult,
)
from .validator import validate_report

__all__ = [
    "ModelConfig",
    "LabReportGenerator",
    "DEFAULT_SECTIONS",
    "ModelBackend",
    "MockModelBackend",
    "build_backend",
    "preprocess",
    "PreprocessingError",
    "ExperimentInput",
    "LabReport",
    "Measurement",
    "ReportSection",
    "ReportStatus",
    "SectionName",
    "REQUIRED_SECTIONS",
    "ValidationIssue",
    "ValidationResult",
    "validate_report",
]
