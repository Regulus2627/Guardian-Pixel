import numpy as np
import pytest

from backend.vision.entropy_map import (
    EntropyMapError,
    calculate_local_entropy,
)


def test_constant_image_has_zero_entropy():
    luminance = np.full(
        (64, 64),
        128,
        dtype=np.uint8,
    )

    entropy_map = calculate_local_entropy(
        luminance,
        window_size=11,
        bins=32,
    )

    assert entropy_map.shape == luminance.shape
    assert entropy_map.dtype == np.float32
    assert np.max(entropy_map) == pytest.approx(0.0)


def test_random_image_has_higher_entropy():
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

    random_entropy = calculate_local_entropy(
        random_image,
        window_size=11,
        bins=32,
    )

    constant_entropy = calculate_local_entropy(
        constant_image,
        window_size=11,
        bins=32,
    )

    assert np.mean(random_entropy) > np.mean(
        constant_entropy
    )


def test_entropy_output_is_finite_and_nonnegative():
    luminance = np.tile(
        np.arange(64, dtype=np.uint8),
        (64, 1),
    )

    entropy_map = calculate_local_entropy(
        luminance,
        window_size=11,
        bins=32,
    )

    assert np.all(np.isfinite(entropy_map))
    assert np.all(entropy_map >= 0)


def test_three_channel_input_is_rejected():
    rgb = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        EntropyMapError,
        match="H×W",
    ):
        calculate_local_entropy(rgb)


def test_non_uint8_input_is_rejected():
    luminance = np.zeros(
        (32, 32),
        dtype=np.float32,
    )

    with pytest.raises(
        EntropyMapError,
        match="uint8",
    ):
        calculate_local_entropy(luminance)


def test_even_window_is_rejected():
    luminance = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(
        EntropyMapError,
        match="odd integer",
    ):
        calculate_local_entropy(
            luminance,
            window_size=10,
        )


def test_invalid_bin_count_is_rejected():
    luminance = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(
        EntropyMapError,
        match="between 2 and 256",
    ):
        calculate_local_entropy(
            luminance,
            bins=1,
        )