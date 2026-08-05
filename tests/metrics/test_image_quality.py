import math

import numpy as np
import pytest

from backend.metrics.image_quality import (
    ImageMetricError,
    calculate_image_quality,
)


def test_identical_images():
    image = np.full(
        (64, 64, 3),
        100,
        dtype=np.uint8,
    )

    result = calculate_image_quality(
        cover=image,
        stego=image.copy(),
        payload_bits=0,
        total_embedded_bits=0,
    )

    assert result.mse == 0
    assert math.isinf(result.psnr)
    assert result.ssim == pytest.approx(1.0)
    assert result.modified_channel_count == 0
    assert result.maximum_absolute_change == 0


def test_one_value_change():
    cover = np.full(
        (64, 64, 3),
        100,
        dtype=np.uint8,
    )

    stego = cover.copy()
    stego[10, 10, 1] = 101

    result = calculate_image_quality(
        cover=cover,
        stego=stego,
        payload_bits=100,
        total_embedded_bits=200,
    )

    assert result.mse > 0
    assert result.psnr > 0
    assert 0 < result.ssim <= 1

    assert result.modified_channel_count == 1
    assert result.modified_pixel_count == 1
    assert result.maximum_absolute_change == 1


def test_bpp_calculation():
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = calculate_image_quality(
        cover=image,
        stego=image.copy(),
        payload_bits=1000,
        total_embedded_bits=2000,
    )

    assert result.payload_bpp == pytest.approx(
        0.1
    )

    assert result.total_bpp == pytest.approx(
        0.2
    )


def test_mismatched_shapes_are_rejected():
    cover = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )

    stego = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        ImageMetricError,
        match="dimensions",
    ):
        calculate_image_quality(
            cover,
            stego,
            payload_bits=0,
            total_embedded_bits=0,
        )