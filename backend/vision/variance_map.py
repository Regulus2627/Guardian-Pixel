"""Local variance feature extraction for GuardianPixel."""

from __future__ import annotations

import cv2
import numpy as np


class VarianceMapError(ValueError):
    """Raised when a local variance map cannot be calculated."""


def calculate_local_variance(
    luminance: np.ndarray,
    window_size: int = 11,
) -> np.ndarray:
    """
    Calculate local luminance variance.

    Higher values represent areas with greater local brightness
    variation, such as texture, edges and detailed regions.
    """

    if not isinstance(luminance, np.ndarray):
        raise VarianceMapError(
            "Luminance input must be a NumPy array."
        )

    if luminance.ndim != 2:
        raise VarianceMapError(
            "Luminance input must have shape H×W."
        )

    if luminance.size == 0:
        raise VarianceMapError(
            "Luminance input cannot be empty."
        )

    if luminance.dtype != np.uint8:
        raise VarianceMapError(
            "Luminance input must use uint8 values."
        )

    if window_size < 3 or window_size % 2 == 0:
        raise VarianceMapError(
            "window_size must be an odd integer "
            "greater than or equal to 3."
        )

    luminance_float = luminance.astype(np.float32)

    local_mean = cv2.boxFilter(
        luminance_float,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    local_squared_mean = cv2.boxFilter(
        luminance_float * luminance_float,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    variance_map = (
        local_squared_mean - local_mean * local_mean
    )

    # Small negative values can occur because of floating-point
    # rounding. Real variance cannot be negative.
    variance_map = np.maximum(
        variance_map,
        0.0,
    ).astype(np.float32)

    if variance_map.shape != luminance.shape:
        raise VarianceMapError(
            "Variance output shape does not match input shape."
        )

    if not np.all(np.isfinite(variance_map)):
        raise VarianceMapError(
            "Variance map contains invalid numerical values."
        )

    return variance_map