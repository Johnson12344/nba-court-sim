from __future__ import annotations

import numpy as np


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted values must have equal non-zero length")
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def interval_coverage(actual: list[float], low: list[float], high: list[float]) -> float:
    if not (len(actual) == len(low) == len(high)) or not actual:
        raise ValueError("Coverage inputs must have equal non-zero length")
    a, lo, hi = np.asarray(actual), np.asarray(low), np.asarray(high)
    return float(np.mean((a >= lo) & (a <= hi)))

