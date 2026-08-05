import io

import numpy as np
import pytest
from PIL import Image

from backend.vision.preprocessing import (
    ImagePreprocessingError,
    load_and_validate_image,
    rgb_to_luminance,
)


def image_to_bytes(
    mode: str,
    size: tuple[int, int],
    color,
    image_format: str = "PNG",
) -> bytes:
    image = Image.new(mode, size, color)

    buffer = io.BytesIO()
    image.save(buffer, format=image_format)

    return buffer.getvalue()


def test_load_rgb_image():
    data = image_to_bytes(
        mode="RGB",
        size=(32, 24),
        color=(100, 150, 200),
    )

    rgb, info = load_and_validate_image(
        data,
        max_pixels=1_000_000,
    )

    assert rgb.shape == (24, 32, 3)
    assert rgb.dtype == np.uint8

    assert info.width == 32
    assert info.height == 24
    assert info.channels == 3
    assert info.total_pixels == 768
    assert info.original_format == "PNG"
    assert info.original_mode == "RGB"


def test_grayscale_image_is_converted_to_rgb():
    data = image_to_bytes(
        mode="L",
        size=(20, 10),
        color=128,
    )

    rgb, info = load_and_validate_image(
        data,
        max_pixels=1_000_000,
    )

    assert rgb.shape == (10, 20, 3)
    assert info.original_mode == "L"

    assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
    assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])


def test_rgba_image_is_converted_to_rgb():
    data = image_to_bytes(
        mode="RGBA",
        size=(16, 12),
        color=(255, 0, 0, 100),
    )

    rgb, info = load_and_validate_image(
        data,
        max_pixels=1_000_000,
    )

    assert rgb.shape == (12, 16, 3)
    assert info.original_mode == "RGBA"


def test_invalid_image_is_rejected():
    with pytest.raises(
        ImagePreprocessingError,
        match="not a valid supported image",
    ):
        load_and_validate_image(
            b"this is not an image",
            max_pixels=1_000_000,
        )


def test_empty_image_data_is_rejected():
    with pytest.raises(
        ImagePreprocessingError,
        match="empty",
    ):
        load_and_validate_image(
            b"",
            max_pixels=1_000_000,
        )


def test_image_above_pixel_limit_is_rejected():
    data = image_to_bytes(
        mode="RGB",
        size=(100, 100),
        color=(0, 0, 0),
    )

    with pytest.raises(
        ImagePreprocessingError,
        match="exceeds the limit",
    ):
        load_and_validate_image(
            data,
            max_pixels=5_000,
        )


def test_rgb_to_luminance():
    rgb = np.array(
        [
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ]
        ],
        dtype=np.uint8,
    )

    luminance = rgb_to_luminance(rgb)

    assert luminance.shape == (1, 3)
    assert luminance.dtype == np.uint8

    assert luminance[0, 0] == 76
    assert luminance[0, 1] == 150
    assert luminance[0, 2] == 29


def test_invalid_rgb_shape_is_rejected():
    grayscale = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(
        ImagePreprocessingError,
        match="shape",
    ):
        rgb_to_luminance(grayscale)