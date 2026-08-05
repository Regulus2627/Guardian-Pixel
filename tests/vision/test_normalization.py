import numpy as np
import pytest

from backend.vision.normalization import (
    MapNormalizationError,
    robust_normalize,
)


def test_normalized_map_has_correct_range():
    feature_map = np.array(
        [
            [0.0, 5.0],
            [10.0, 15.0],
        ],
        dtype=np.float32,
    )

    normalized = robust_normalize(
        feature_map,
        low_percentile=0,
        high_percentile=100,
    )

    assert normalized.shape == feature_map.shape
    assert normalized.dtype == np.float32
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_constant_map_returns_zeros():
    feature_map = np.full(
        (20, 20),
        7.0,
        dtype=np.float32,
    )

    normalized = robust_normalize(feature_map)

    assert np.all(normalized == 0)
    assert normalized.dtype == np.float32


def test_nan_and_infinity_are_handled():
    feature_map = np.array(
        [
            [0.0, np.nan],
            [np.inf, 10.0],
        ],
        dtype=np.float32,
    )

    normalized = robust_normalize(
        feature_map,
        low_percentile=0,
        high_percentile=100,
    )

    assert np.all(np.isfinite(normalized))
    assert normalized.min() >= 0
    assert normalized.max() <= 1


def test_invalid_map_shape_is_rejected():
    feature_map = np.zeros(
        (10, 10, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        MapNormalizationError,
        match="H×W",
    ):
        robust_normalize(feature_map)


def test_invalid_percentiles_are_rejected():
    feature_map = np.zeros(
        (10, 10),
        dtype=np.float32,
    )

    with pytest.raises(
        MapNormalizationError,
        match="Percentiles",
    ):
        robust_normalize(
            feature_map,
            low_percentile=90,
            high_percentile=10,
        )