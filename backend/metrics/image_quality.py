"""Image-quality and payload metrics for GuardianPixel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.metrics import (
    structural_similarity,
)


class ImageMetricError(ValueError):
    """Raised when image metrics cannot be calculated."""


@dataclass(frozen=True, slots=True)
class ImageQualityMetrics:
    """Image-quality measurements."""

    mse: float
    psnr: float
    ssim: float

    modified_channel_count: int
    modified_pixel_count: int
    maximum_absolute_change: int

    payload_bits: int
    payload_bpp: float
    total_embedded_bits: int
    total_bpp: float


def calculate_image_quality(
    cover: np.ndarray,
    stego: np.ndarray,
    payload_bits: int,
    total_embedded_bits: int,
) -> ImageQualityMetrics:
    """Compare a cover image with its stego image."""

    if not isinstance(
        cover,
        np.ndarray,
    ) or not isinstance(
        stego,
        np.ndarray,
    ):
        raise ImageMetricError(
            "Cover and stego images must be NumPy arrays."
        )

    if (
        cover.ndim != 3
        or cover.shape[2] != 3
        or stego.ndim != 3
        or stego.shape[2] != 3
    ):
        raise ImageMetricError(
            "Images must have shape H×W×3."
        )

    if cover.shape != stego.shape:
        raise ImageMetricError(
            "Cover and stego dimensions must match."
        )

    if (
        cover.dtype != np.uint8
        or stego.dtype != np.uint8
    ):
        raise ImageMetricError(
            "Images must use uint8 values."
        )

    if payload_bits < 0:
        raise ImageMetricError(
            "payload_bits cannot be negative."
        )

    if total_embedded_bits < payload_bits:
        raise ImageMetricError(
            "total_embedded_bits cannot be smaller "
            "than payload_bits."
        )

    cover_float = cover.astype(
        np.float64
    )

    stego_float = stego.astype(
        np.float64
    )

    difference = (
        stego_float - cover_float
    )

    squared_difference = (
        difference * difference
    )

    mse = float(
        np.mean(squared_difference)
    )

    if mse == 0:
        psnr = math.inf
    else:
        psnr = float(
            10.0
            * math.log10(
                (255.0 * 255.0) / mse
            )
        )

    ssim = float(
        structural_similarity(
            cover,
            stego,
            data_range=255,
            channel_axis=2,
        )
    )

    absolute_difference = np.abs(
        stego.astype(np.int16)
        - cover.astype(np.int16)
    )

    modified_channel_count = int(
        np.count_nonzero(
            absolute_difference
        )
    )

    modified_pixel_count = int(
        np.count_nonzero(
            np.any(
                absolute_difference > 0,
                axis=2,
            )
        )
    )

    maximum_absolute_change = int(
        absolute_difference.max()
    )

    total_pixels = (
        cover.shape[0]
        * cover.shape[1]
    )

    payload_bpp = (
        payload_bits / total_pixels
    )

    total_bpp = (
        total_embedded_bits
        / total_pixels
    )

    return ImageQualityMetrics(
        mse=mse,
        psnr=psnr,
        ssim=ssim,
        modified_channel_count=(
            modified_channel_count
        ),
        modified_pixel_count=(
            modified_pixel_count
        ),
        maximum_absolute_change=(
            maximum_absolute_change
        ),
        payload_bits=payload_bits,
        payload_bpp=float(payload_bpp),
        total_embedded_bits=(
            total_embedded_bits
        ),
        total_bpp=float(total_bpp),
    )