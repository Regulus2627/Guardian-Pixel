"""Feature-map fusion for GuardianPixel."""

from __future__ import annotations

import math

import cv2
import numpy as np

from backend.vision.normalization import robust_normalize
from backend.vision.schemas import FeatureMaps


class FeatureFusionError(ValueError):
    """Raised when feature maps cannot be fused."""


def _validate_feature_map(
    feature_map: np.ndarray,
    name: str,
) -> None:
    """Validate one two-dimensional feature map."""

    if not isinstance(feature_map, np.ndarray):
        raise FeatureFusionError(
            f"{name} must be a NumPy array."
        )

    if feature_map.ndim != 2:
        raise FeatureFusionError(
            f"{name} must have shape H×W."
        )

    if feature_map.size == 0:
        raise FeatureFusionError(
            f"{name} cannot be empty."
        )

    if not np.all(np.isfinite(feature_map)):
        raise FeatureFusionError(
            f"{name} contains invalid numerical values."
        )


def fuse_feature_maps(
    hed_map: np.ndarray,
    entropy_map: np.ndarray,
    variance_map: np.ndarray,
    hed_weight: float = 0.45,
    entropy_weight: float = 0.35,
    variance_weight: float = 0.20,
    median_kernel: int = 3,
) -> FeatureMaps:
    """
    Normalize and combine HED, entropy and variance maps.

    Returns normalized individual maps and the final normalized
    suitability heatmap.
    """

    _validate_feature_map(hed_map, "hed_map")
    _validate_feature_map(entropy_map, "entropy_map")
    _validate_feature_map(variance_map, "variance_map")

    if not (
        hed_map.shape
        == entropy_map.shape
        == variance_map.shape
    ):
        raise FeatureFusionError(
            "All feature maps must have the same shape."
        )

    weights = (
        float(hed_weight),
        float(entropy_weight),
        float(variance_weight),
    )

    if any(weight < 0 for weight in weights):
        raise FeatureFusionError(
            "Fusion weights cannot be negative."
        )

    if not math.isclose(
        sum(weights),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise FeatureFusionError(
            "Fusion weights must add up to 1.0."
        )

    if median_kernel != 1 and (
        median_kernel < 3
        or median_kernel % 2 == 0
    ):
        raise FeatureFusionError(
            "median_kernel must be 1 or an odd integer "
            "greater than or equal to 3."
        )

    normalized_hed = robust_normalize(hed_map)
    normalized_entropy = robust_normalize(entropy_map)
    normalized_variance = robust_normalize(variance_map)

    fused = (
        weights[0] * normalized_hed
        + weights[1] * normalized_entropy
        + weights[2] * normalized_variance
    ).astype(np.float32)

    if median_kernel > 1:
        fused = cv2.medianBlur(
            fused,
            median_kernel,
        )

    fused = robust_normalize(fused)

    return FeatureMaps(
        hed_map=normalized_hed,
        entropy_map=normalized_entropy,
        variance_map=normalized_variance,
        fused_heatmap=fused,
    )