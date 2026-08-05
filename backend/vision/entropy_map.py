"""Local Shannon-entropy feature extraction for GuardianPixel."""

from __future__ import annotations

import numpy as np
from skimage.filters.rank import entropy


class EntropyMapError(ValueError):
    """Raised when a local entropy map cannot be calculated."""


def calculate_local_entropy(
    luminance: np.ndarray,
    window_size: int = 11,
    bins: int = 32,
) -> np.ndarray:
    """
    Calculate local Shannon entropy from a uint8 luminance image.

    Higher values represent more locally unpredictable or textured
    image regions.
    """

    if not isinstance(luminance, np.ndarray):
        raise EntropyMapError(
            "Luminance input must be a NumPy array."
        )

    if luminance.ndim != 2:
        raise EntropyMapError(
            "Luminance input must have shape H×W."
        )

    if luminance.size == 0:
        raise EntropyMapError(
            "Luminance input cannot be empty."
        )

    if luminance.dtype != np.uint8:
        raise EntropyMapError(
            "Luminance input must use uint8 values."
        )

    if window_size < 3 or window_size % 2 == 0:
        raise EntropyMapError(
            "window_size must be an odd integer "
            "greater than or equal to 3."
        )

    if not 2 <= bins <= 256:
        raise EntropyMapError(
            "bins must be between 2 and 256."
        )

    # Quantize the 256 possible intensities into the requested bins.
    # uint16 prevents overflow during multiplication.
    quantized = (
        luminance.astype(np.uint16) * bins // 256
    ).astype(np.uint8)

    footprint = np.ones(
        (window_size, window_size),
        dtype=np.uint8,
    )

    entropy_map = entropy(
        quantized,
        footprint=footprint,
    )

    entropy_map = np.asarray(
        entropy_map,
        dtype=np.float32,
    )

    if entropy_map.shape != luminance.shape:
        raise EntropyMapError(
            "Entropy output shape does not match input shape."
        )

    if not np.all(np.isfinite(entropy_map)):
        raise EntropyMapError(
            "Entropy map contains invalid numerical values."
        )

    return entropy_map