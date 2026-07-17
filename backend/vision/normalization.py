"""Robust normalization of GuardianPixel feature maps."""

from __future__ import annotations

import numpy as np


class MapNormalizationError(ValueError):
    """Raised when a feature map cannot be normalized."""


def robust_normalize(
    feature_map: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Normalize a two-dimensional feature map to the range [0, 1].

    Percentile clipping prevents a small number of extreme values
    from controlling the complete output range.
    """

    if not isinstance(feature_map, np.ndarray):
        raise MapNormalizationError(
            "Feature map must be a NumPy array."
        )

    if feature_map.ndim != 2:
        raise MapNormalizationError(
            "Feature map must have shape H×W."
        )

    if feature_map.size == 0:
        raise MapNormalizationError(
            "Feature map cannot be empty."
        )

    if not 0 <= low_percentile < high_percentile <= 100:
        raise MapNormalizationError(
            "Percentiles must satisfy "
            "0 <= low < high <= 100."
        )

    feature_float = feature_map.astype(
        np.float32,
        copy=False,
    )

    finite_mask = np.isfinite(feature_float)
    finite_values = feature_float[finite_mask]

    if finite_values.size == 0:
        return np.zeros(
            feature_float.shape,
            dtype=np.float32,
        )

    replacement_value = float(np.median(finite_values))

    cleaned = np.where(
        finite_mask,
        feature_float,
        replacement_value,
    ).astype(np.float32)

    lower_value = float(
        np.percentile(cleaned, low_percentile)
    )

    upper_value = float(
        np.percentile(cleaned, high_percentile)
    )

    value_range = upper_value - lower_value

    if value_range <= np.finfo(np.float32).eps:
        return np.zeros(
            cleaned.shape,
            dtype=np.float32,
        )

    clipped = np.clip(
        cleaned,
        lower_value,
        upper_value,
    )

    normalized = (
        clipped - lower_value
    ) / value_range

    return np.clip(
        normalized,
        0.0,
        1.0,
    ).astype(np.float32)