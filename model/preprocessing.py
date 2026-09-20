"""Cleaning and normalization of raw laboratory experiment input.

Turns a loosely-structured ``dict`` (as might come from a form, API, or
file upload) into a validated :class:`~model.schemas.ExperimentInput`.
This stage is deterministic and does not call any model.
"""

from __future__ import annotations

from typing import Any

from .schemas import ExperimentInput, Measurement
from .units import normalize_unit

#: Raw keys that must be present (and non-empty after cleaning).
REQUIRED_RAW_KEYS: tuple[str, ...] = ("title", "objective")


class PreprocessingError(ValueError):
    """Raised when raw input cannot be turned into a valid experiment."""


def _clean_text(value: Any) -> str:
    """Collapse internal whitespace and strip a value coerced to ``str``."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _clean_str_list(value: Any) -> list[str]:
    """Normalize a value into a list of non-empty cleaned strings."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split("\n")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    cleaned = [_clean_text(item) for item in items]
    return [item for item in cleaned if item]


def _normalize_unit(unit: Any) -> str:
    """Normalize a measurement unit string.

    Trims whitespace and maps known synonyms to their canonical forms using
    the shared :mod:`model.units` registry. Casing is preserved for units
    where it is significant (e.g. ``mL``).
    """
    return normalize_unit(_clean_text(unit))


def _clean_measurements(value: Any) -> list[Measurement]:
    """Coerce raw measurement records into :class:`Measurement` objects.

    Each raw record is expected to be a mapping with ``name``, ``value``,
    and ``unit`` keys. Records whose value cannot be coerced to a float are
    skipped here; the validator is responsible for flagging missing data.
    """
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        raise PreprocessingError("'measurements' must be a list of records")

    measurements: list[Measurement] = []
    for record in value:
        if not isinstance(record, dict):
            raise PreprocessingError(f"measurement record must be a mapping: {record!r}")
        name = _clean_text(record.get("name"))
        unit = _normalize_unit(record.get("unit"))
        raw_val = record.get("value")
        try:
            numeric = float(raw_val)
        except (TypeError, ValueError):
            # Leave out unparseable values; validator will catch a missing
            # required measurement. We do not silently invent a number.
            continue
        measurements.append(Measurement(name=name, value=numeric, unit=unit))
    return measurements


def preprocess(raw: dict[str, Any]) -> ExperimentInput:
    """Clean and normalize raw experiment input.

    Args:
        raw: Loosely-structured experiment data.

    Returns:
        A normalized :class:`ExperimentInput`.

    Raises:
        PreprocessingError: If ``raw`` is not a mapping or a required key is
            missing/empty after cleaning.
    """
    if not isinstance(raw, dict):
        raise PreprocessingError("raw input must be a dict")

    title = _clean_text(raw.get("title"))
    objective = _clean_text(raw.get("objective"))

    for key in REQUIRED_RAW_KEYS:
        if not _clean_text(raw.get(key)):
            raise PreprocessingError(f"missing required field: '{key}'")

    metadata_raw = raw.get("metadata") or {}
    if not isinstance(metadata_raw, dict):
        raise PreprocessingError("'metadata' must be a mapping")
    metadata = {
        _clean_text(k): _clean_text(v)
        for k, v in metadata_raw.items()
        if _clean_text(k)
    }

    return ExperimentInput(
        title=title,
        objective=objective,
        materials=_clean_str_list(raw.get("materials")),
        procedure=_clean_str_list(raw.get("procedure")),
        measurements=_clean_measurements(raw.get("measurements")),
        metadata=metadata,
    )
