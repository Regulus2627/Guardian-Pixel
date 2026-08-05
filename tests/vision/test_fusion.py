import numpy as np
import pytest

from backend.vision.fusion import (
    FeatureFusionError,
    fuse_feature_maps,
)


def create_feature_maps(
    height: int = 32,
    width: int = 48,
):
    y_values = np.linspace(
        0,
        1,
        height,
        dtype=np.float32,
    ).reshape(height, 1)

    x_values = np.linspace(
        0,
        1,
        width,
        dtype=np.float32,
    ).reshape(1, width)

    hed = np.broadcast_to(
        x_values,
        (height, width),
    ).copy()

    entropy = np.broadcast_to(
        y_values,
        (height, width),
    ).copy()

    variance = (
        hed * entropy
    ).astype(np.float32)

    return hed, entropy, variance


def test_fusion_produces_correct_shapes():
    hed, entropy, variance = create_feature_maps()

    result = fuse_feature_maps(
        hed,
        entropy,
        variance,
    )

    assert result.hed_map.shape == hed.shape
    assert result.entropy_map.shape == hed.shape
    assert result.variance_map.shape == hed.shape
    assert result.fused_heatmap.shape == hed.shape


def test_all_maps_are_float32_and_normalized():
    hed, entropy, variance = create_feature_maps()

    result = fuse_feature_maps(
        hed,
        entropy,
        variance,
    )

    maps = [
        result.hed_map,
        result.entropy_map,
        result.variance_map,
        result.fused_heatmap,
    ]

    for feature_map in maps:
        assert feature_map.dtype == np.float32
        assert feature_map.min() >= 0
        assert feature_map.max() <= 1
        assert np.all(np.isfinite(feature_map))


def test_constant_maps_produce_zero_heatmap():
    feature_map = np.full(
        (32, 32),
        5.0,
        dtype=np.float32,
    )

    result = fuse_feature_maps(
        feature_map,
        feature_map,
        feature_map,
    )

    assert np.all(result.fused_heatmap == 0)


def test_mismatched_shapes_are_rejected():
    hed = np.zeros(
        (32, 32),
        dtype=np.float32,
    )

    entropy = np.zeros(
        (32, 31),
        dtype=np.float32,
    )

    variance = np.zeros(
        (32, 32),
        dtype=np.float32,
    )

    with pytest.raises(
        FeatureFusionError,
        match="same shape",
    ):
        fuse_feature_maps(
            hed,
            entropy,
            variance,
        )


def test_invalid_weight_total_is_rejected():
    hed, entropy, variance = create_feature_maps()

    with pytest.raises(
        FeatureFusionError,
        match="add up to 1.0",
    ):
        fuse_feature_maps(
            hed,
            entropy,
            variance,
            hed_weight=0.80,
            entropy_weight=0.35,
            variance_weight=0.20,
        )


def test_negative_weight_is_rejected():
    hed, entropy, variance = create_feature_maps()

    with pytest.raises(
        FeatureFusionError,
        match="cannot be negative",
    ):
        fuse_feature_maps(
            hed,
            entropy,
            variance,
            hed_weight=0.60,
            entropy_weight=0.50,
            variance_weight=-0.10,
        )


def test_even_median_kernel_is_rejected():
    hed, entropy, variance = create_feature_maps()

    with pytest.raises(
        FeatureFusionError,
        match="median_kernel",
    ):
        fuse_feature_maps(
            hed,
            entropy,
            variance,
            median_kernel=4,
        )


def test_three_channel_map_is_rejected():
    invalid_map = np.zeros(
        (32, 32, 3),
        dtype=np.float32,
    )

    valid_map = np.zeros(
        (32, 32),
        dtype=np.float32,
    )

    with pytest.raises(
        FeatureFusionError,
        match="H×W",
    ):
        fuse_feature_maps(
            invalid_map,
            valid_map,
            valid_map,
        )