"""Strict loading and cycle-local feature extraction for the UCI dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "inspection"
    / "uci-hydraulic-condition-monitoring"
    / "extracted"
)
DATASET_NAME = "UCI Condition Monitoring of Hydraulic Systems"
DATASET_DOI = "10.24432/C5CW21"
DATASET_URL = "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems"
ARCHIVE_SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
EXPECTED_CYCLES = 2205
PROFILE_FILE_BYTES = 31196
FEATURE_RECIPE_VERSION = "cycle-summary-v1"
FEATURE_STATISTICS = ("mean", "std", "min", "max", "rms", "p25", "median", "p75")
PROFILE_COLUMNS = (
    "cooler_condition_percent",
    "valve_condition_percent",
    "pump_leakage_grade",
    "accumulator_pressure_bar",
    "stable_flag",
)

SENSOR_SPECS: dict[str, dict[str, object]] = {
    "PS1": {"unit": "bar", "sampling_rate_hz": 100, "columns": 6000, "bytes": 91391892},
    "PS2": {"unit": "bar", "sampling_rate_hz": 100, "columns": 6000, "bytes": 82373611},
    "PS3": {"unit": "bar", "sampling_rate_hz": 100, "columns": 6000, "bytes": 69916801},
    "PS4": {"unit": "bar", "sampling_rate_hz": 100, "columns": 6000, "bytes": 46698180},
    "PS5": {"unit": "bar", "sampling_rate_hz": 100, "columns": 6000, "bytes": 77776429},
    "PS6": {"unit": "bar", "sampling_rate_hz": 100, "columns": 6000, "bytes": 77875407},
    "EPS1": {"unit": "W", "sampling_rate_hz": 100, "columns": 6000, "bytes": 87411594},
    "FS1": {"unit": "l/min", "sampling_rate_hz": 10, "columns": 600, "bytes": 7581926},
    "FS2": {"unit": "l/min", "sampling_rate_hz": 10, "columns": 600, "bytes": 8271923},
    "TS1": {"unit": "C", "sampling_rate_hz": 1, "columns": 60, "bytes": 904586},
    "TS2": {"unit": "C", "sampling_rate_hz": 1, "columns": 60, "bytes": 910594},
    "TS3": {"unit": "C", "sampling_rate_hz": 1, "columns": 60, "bytes": 908805},
    "TS4": {"unit": "C", "sampling_rate_hz": 1, "columns": 60, "bytes": 910127},
    "VS1": {"unit": "mm/s", "sampling_rate_hz": 1, "columns": 60, "bytes": 778499},
    "CE": {"unit": "%", "sampling_rate_hz": 1, "columns": 60, "bytes": 911516},
    "CP": {"unit": "kW", "sampling_rate_hz": 1, "columns": 60, "bytes": 779343},
    "SE": {"unit": "%", "sampling_rate_hz": 1, "columns": 60, "bytes": 824838},
}
PHYSICAL_SENSOR_NAMES = (
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "EPS1",
    "FS1",
    "FS2",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "VS1",
)
VIRTUAL_SENSOR_NAMES = ("CE", "CP", "SE")
PROFILE_VALUE_COUNTS = (
    {3: 732, 20: 732, 100: 741},
    {73: 360, 80: 360, 90: 360, 100: 1125},
    {0: 1221, 1: 492, 2: 492},
    {90: 808, 100: 399, 115: 399, 130: 599},
    {0: 1449, 1: 756},
)


class DatasetIntegrityError(ValueError):
    """Raised when source files contradict the audited dataset contract."""


@dataclass(frozen=True)
class HydraulicDataset:
    """Cycle-aligned model features and profile targets from the audited files."""

    cycle_ids: np.ndarray
    features: pd.DataFrame
    profile_targets: pd.DataFrame
    pump_leakage_grade: pd.Series
    pump_leakage_binary: pd.Series
    archive_sha256: str

    def model_features(self) -> pd.DataFrame:
        """Return only physical-channel features allowed for the baseline."""
        columns = [
            feature_name(sensor_name, statistic)
            for sensor_name in PHYSICAL_SENSOR_NAMES
            for statistic in FEATURE_STATISTICS
        ]
        return self.features.loc[:, columns]


def feature_name(sensor_name: str, statistic: str) -> str:
    """Return a stable feature name for a sensor summary."""
    if sensor_name not in SENSOR_SPECS:
        raise ValueError(f"unknown hydraulic sensor: {sensor_name}")
    if statistic not in FEATURE_STATISTICS:
        raise ValueError(f"unknown cycle statistic: {statistic}")
    return f"{sensor_name}__{statistic}"


def load_sensor_matrix(
    path: Path,
    expected_rows: int,
    expected_columns: int,
) -> np.ndarray:
    """Read a tab-delimited sensor matrix and reject shape or value errors."""
    try:
        matrix = np.loadtxt(path, delimiter="\t", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise DatasetIntegrityError(f"cannot parse sensor file {path}: {exc}") from exc
    if matrix.shape != (expected_rows, expected_columns):
        raise DatasetIntegrityError(
            f"{path.name} has shape {matrix.shape}; expected "
            f"({expected_rows}, {expected_columns})"
        )
    if not np.isfinite(matrix).all():
        raise DatasetIntegrityError(f"{path.name} contains non-finite values")
    return matrix


def extract_sensor_features(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Compute eight per-cycle statistics without using any target values."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("a sensor matrix must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("sensor values must all be finite")
    percentiles = np.percentile(values, [25, 50, 75], axis=1)
    statistics = {
        "mean": np.mean(values, axis=1),
        "std": np.std(values, axis=1, ddof=0),
        "min": np.min(values, axis=1),
        "max": np.max(values, axis=1),
        "rms": np.sqrt(np.mean(np.square(values), axis=1)),
        "p25": percentiles[0],
        "median": percentiles[1],
        "p75": percentiles[2],
    }
    if any(not np.isfinite(result).all() for result in statistics.values()):
        raise ValueError("feature extraction produced a non-finite result")
    return statistics


def make_cycle_ids(sensor_row_counts: dict[str, int], profile_rows: int) -> np.ndarray:
    """Validate positional row alignment and assign documented 1-based IDs."""
    if not sensor_row_counts:
        raise DatasetIntegrityError("no sensor row counts were supplied")
    mismatched = {
        name: rows for name, rows in sensor_row_counts.items() if rows != profile_rows
    }
    if mismatched:
        raise DatasetIntegrityError(
            f"sensor rows do not align with {profile_rows} profile rows: {mismatched}"
        )
    return np.arange(1, profile_rows + 1, dtype=np.int64)


def validate_profile(profile: np.ndarray, expected_rows: int = EXPECTED_CYCLES) -> None:
    """Validate the five profile targets and their audited value distributions."""
    values = np.asarray(profile)
    if values.shape != (expected_rows, len(PROFILE_COLUMNS)):
        raise DatasetIntegrityError(
            f"profile has shape {values.shape}; expected "
            f"({expected_rows}, {len(PROFILE_COLUMNS)})"
        )
    if not np.isfinite(values).all():
        raise DatasetIntegrityError("profile contains non-finite values")
    if not np.equal(values, np.floor(values)).all():
        raise DatasetIntegrityError("profile target values must be integers")
    for column_index, expected_counts in enumerate(PROFILE_VALUE_COUNTS):
        observed, counts = np.unique(values[:, column_index], return_counts=True)
        observed_counts = {
            int(value): int(count) for value, count in zip(observed, counts, strict=True)
        }
        if observed_counts != expected_counts:
            column_name = PROFILE_COLUMNS[column_index]
            raise DatasetIntegrityError(
                f"{column_name} distribution differs from the audited source: "
                f"{observed_counts} != {expected_counts}"
            )


def map_pump_leakage_grade(grades: np.ndarray) -> np.ndarray:
    """Map profile grade 0 to 0 and weak/severe grades 1/2 to 1."""
    values = np.asarray(grades)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("pump grades must be a finite one-dimensional array")
    if not np.equal(values, np.floor(values)).all() or not np.isin(
        values, (0, 1, 2)
    ).all():
        raise ValueError("pump grades must contain only integer values 0, 1, and 2")
    return (values != 0).astype(np.int8)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetIntegrityError(f"cannot read source archive {path}: {exc}") from exc
    return digest.hexdigest()


def load_hydraulic_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    verify_archive: bool = True,
) -> HydraulicDataset:
    """Load and strictly verify the audited local UCI archive extraction.

    Alignment is positional: row *i* in every matrix and profile is the same
    cycle, following the UCI documentation. The source files contain no
    explicit cycle IDs or timestamps.
    """
    data_dir = Path(data_dir)
    archive_hash = ARCHIVE_SHA256
    if verify_archive:
        archive_path = data_dir.parent / "uci_dataset_447.zip"
        if not archive_path.is_file():
            raise DatasetIntegrityError(
                f"missing official hydraulic dataset archive: {archive_path}. "
                "From the repository root, run "
                "`python scripts/setup_hydraulic_dataset.py` to download and "
                "verify the UCI dataset (DOI 10.24432/C5CW21)."
            )
        archive_hash = sha256_file(archive_path)
        if archive_hash != ARCHIVE_SHA256:
            raise DatasetIntegrityError(
                f"source archive SHA-256 mismatch: {archive_hash} != {ARCHIVE_SHA256}"
            )

    profile_path = data_dir / "profile.txt"
    if not profile_path.is_file():
        raise DatasetIntegrityError(
            f"missing profile file: {profile_path}. Run "
            "`python scripts/setup_hydraulic_dataset.py` to verify and extract "
            "the official UCI hydraulic dataset."
        )
    if profile_path.stat().st_size != PROFILE_FILE_BYTES:
        raise DatasetIntegrityError(
            f"profile.txt byte size changed: {profile_path.stat().st_size} "
            f"!= {PROFILE_FILE_BYTES}"
        )
    try:
        profile = np.loadtxt(profile_path, delimiter="\t", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise DatasetIntegrityError(f"cannot parse profile file: {exc}") from exc
    validate_profile(profile)

    feature_columns: dict[str, np.ndarray] = {}
    sensor_row_counts: dict[str, int] = {}
    for sensor_name, spec in SENSOR_SPECS.items():
        sensor_path = data_dir / f"{sensor_name}.txt"
        if not sensor_path.is_file():
            raise DatasetIntegrityError(
                f"missing sensor file: {sensor_path}. Run "
                "`python scripts/setup_hydraulic_dataset.py` to verify and "
                "extract the official UCI hydraulic dataset."
            )
        if sensor_path.stat().st_size != spec["bytes"]:
            raise DatasetIntegrityError(
                f"{sensor_path.name} byte size changed: {sensor_path.stat().st_size} "
                f"!= {spec['bytes']}"
            )
        matrix = load_sensor_matrix(
            sensor_path,
            expected_rows=EXPECTED_CYCLES,
            expected_columns=int(spec["columns"]),
        )
        sensor_row_counts[sensor_name] = matrix.shape[0]
        statistics = extract_sensor_features(matrix)
        for statistic, result in statistics.items():
            feature_columns[feature_name(sensor_name, statistic)] = result
        del matrix

    cycle_ids = make_cycle_ids(sensor_row_counts, profile.shape[0])
    index = pd.Index(cycle_ids, name="cycle_id")
    features = pd.DataFrame(feature_columns, index=index)
    profile_targets = pd.DataFrame(
        profile.astype(np.int64), columns=PROFILE_COLUMNS, index=index
    )
    grade = profile_targets["pump_leakage_grade"].astype(np.int64).rename(
        "pump_leakage_grade"
    )
    binary = pd.Series(
        map_pump_leakage_grade(grade.to_numpy()),
        index=index,
        name="pump_leakage_binary",
    )
    return HydraulicDataset(
        cycle_ids=cycle_ids,
        features=features,
        profile_targets=profile_targets,
        pump_leakage_grade=grade,
        pump_leakage_binary=binary,
        archive_sha256=archive_hash,
    )
