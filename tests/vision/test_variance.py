import numpy as np
import pytest

from backend.vision.variance_map import (
    VarianceMapError,
    calculate_local_variance,
)


def test_constant_image_has_zero_variance():
    luminance = np.full(
        (64, 64),
        128,
        dtype=np.uint8,
    )

    variance_map = calculate_local_variance(
        luminance,
        window_size=11,
    )

    assert variance_map.shape == luminance.shape
    assert variance_map.dtype == np.float32
    assert np.max(variance_map) == pytest.approx(
        0.0,
        abs=1e-5,
    )


def test_random_image_has_higher_variance():
    random_generator = np.random.default_rng(42)

    random_image = random_generator.integers(
        0,
        256,
        size=(64, 64),
        dtype=np.uint8,
    )

    constant_image = np.full(
        (64, 64),
        128,
        dtype=np.uint8,
    )

    random_variance = calculate_local_variance(
        random_image,
        window_size=11,
    )

    constant_variance = calculate_local_variance(
        constant_image,
        window_size=11,
    )

    assert np.mean(random_variance) > np.mean(
        constant_variance
    )


def test_variance_map_is_nonnegative():
    luminance = np.tile(
        np.arange(64, dtype=np.uint8),
        (64, 1),
    )

    variance_map = calculate_local_variance(
        luminance,
        window_size=11,
    )

    assert np.all(variance_map >= 0)


def test_variance_map_is_finite():
    random_generator = np.random.default_rng(100)

    luminance = random_generator.integers(
        0,
        256,
        size=(32, 32),
        dtype=np.uint8,
    )

    variance_map = calculate_local_variance(
        luminance,
        window_size=5,
    )

    assert np.all(np.isfinite(variance_map))


def test_three_channel_input_is_rejected():
    rgb = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        VarianceMapError,
        match="H×W",
    ):
        calculate_local_variance(rgb)


def test_non_uint8_input_is_rejected():
    luminance = np.zeros(
        (32, 32),
        dtype=np.float32,
    )

    with pytest.raises(
        VarianceMapError,
        match="uint8",
    ):
        calculate_local_variance(luminance)


def test_even_window_is_rejected():
    luminance = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(
        VarianceMapError,
        match="odd integer",
    ):
        calculate_local_variance(
            luminance,
            window_size=10,
        )


def test_window_below_three_is_rejected():
    luminance = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(
        VarianceMapError,
        match="greater than or equal to 3",
    ):
        calculate_local_variance(
            luminance,
            window_size=1,
        )


def test_empty_input_is_rejected():
    luminance = np.empty(
        (0, 0),
        dtype=np.uint8,
    )

    with pytest.raises(
        VarianceMapError,
        match="empty",
    ):
        calculate_local_variance(luminance)