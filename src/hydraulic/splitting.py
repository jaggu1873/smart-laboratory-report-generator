"""Forward-in-time cycle splits with an explicit boundary purge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_EMBARGO_CYCLES = 60


@dataclass(frozen=True)
class CycleSplits:
    """Position indexes and excluded intervals for a reproducible split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    boundaries: dict[str, tuple[int, int]]


def chronological_splits(
    n_cycles: int,
    embargo_cycles: int = DEFAULT_EMBARGO_CYCLES,
) -> CycleSplits:
    """Create 70/15/15 chronological partitions and purge both boundaries.

    The default 60-cycle purge is a one-hour gap on each side of each split
    boundary because UCI documents 60-second cycles. It is a conservative
    buffer, not evidence that serial dependence has disappeared.
    """
    if n_cycles < 1:
        raise ValueError("n_cycles must be positive")
    if embargo_cycles < 0:
        raise ValueError("embargo_cycles cannot be negative")
    train_end = int(n_cycles * 0.70)
    validation_end = int(n_cycles * 0.85)
    train_stop = train_end - embargo_cycles
    validation_start = train_end + embargo_cycles
    validation_stop = validation_end - embargo_cycles
    test_start = validation_end + embargo_cycles
    if train_stop <= 0 or validation_start >= validation_stop or test_start >= n_cycles:
        raise ValueError("cycle count is too small for the requested split and embargo")

    train = np.arange(0, train_stop, dtype=np.int64)
    validation = np.arange(validation_start, validation_stop, dtype=np.int64)
    test = np.arange(test_start, n_cycles, dtype=np.int64)
    used = np.concatenate((train, validation, test))
    purged = np.setdiff1d(np.arange(n_cycles, dtype=np.int64), used, assume_unique=True)
    boundaries = {
        "train": (int(train[0]), int(train[-1])),
        "validation": (int(validation[0]), int(validation[-1])),
        "test": (int(test[0]), int(test[-1])),
    }
    return CycleSplits(train, validation, test, purged, boundaries)


def validate_no_cross_split_duplicates(
    features: np.ndarray,
    splits: CycleSplits,
) -> None:
    """Reject identical complete feature vectors in different partitions."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("duplicate checking requires a finite two-dimensional matrix")
    ownership: dict[tuple[float, ...], str] = {}
    for split_name in ("train", "validation", "test"):
        for row_index in getattr(splits, split_name):
            key = tuple(values[int(row_index)].tolist())
            previous = ownership.get(key)
            if previous is not None and previous != split_name:
                raise ValueError(
                    "identical cycle feature vectors cross the "
                    f"{previous}/{split_name} split boundary"
                )
            ownership[key] = split_name
