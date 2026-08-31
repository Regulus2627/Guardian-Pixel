import numpy as np
import pytest

from backend.vision.block_map import (
    BlockMapError,
    calculate_block_scores,
    generate_capacity_aware_block_map,
)


def test_512_image_produces_64_by_64_blocks():
    heatmap = np.zeros(
        (512, 512),
        dtype=np.float32,
    )

    scores, pixel_counts = (
        calculate_block_scores(
            heatmap,
            block_size=8,
        )
    )

    assert scores.shape == (64, 64)
    assert pixel_counts.shape == (64, 64)
    assert np.all(pixel_counts == 64)


def test_1024_image_produces_128_by_128_blocks():
    heatmap = np.zeros(
        (1024, 1024),
        dtype=np.float32,
    )

    scores, pixel_counts = (
        calculate_block_scores(
            heatmap,
            block_size=8,
        )
    )

    assert scores.shape == (128, 128)
    assert pixel_counts.shape == (128, 128)


def test_non_divisible_dimensions_are_supported():
    heatmap = np.zeros(
        (533, 800),
        dtype=np.float32,
    )

    scores, pixel_counts = (
        calculate_block_scores(
            heatmap,
            block_size=8,
        )
    )

    assert scores.shape == (67, 100)

    # Last block row contains only 5 valid image rows.
    assert pixel_counts[-1, 0] == 5 * 8


def test_highest_scoring_block_is_selected_first():
    heatmap = np.zeros(
        (16, 16),
        dtype=np.float32,
    )

    heatmap[0:8, 0:8] = 1.0
    heatmap[0:8, 8:16] = 0.8
    heatmap[8:16, 0:8] = 0.2
    heatmap[8:16, 8:16] = 0.1

    result = (
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=64,
            block_size=8,
            safety_margin=0,
        )
    )

    assert result.selected_block_count == 1
    assert result.selected_blocks[0, 0]


def test_larger_payload_selects_more_blocks():
    heatmap = np.linspace(
        0,
        1,
        64 * 64,
        dtype=np.float32,
    ).reshape(64, 64)

    small_result = (
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=64,
            block_size=8,
            safety_margin=0,
        )
    )

    large_result = (
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=512,
            block_size=8,
            safety_margin=0,
        )
    )

    assert (
        large_result.selected_block_count
        > small_result.selected_block_count
    )


def test_selected_capacity_is_sufficient():
    heatmap = np.random.default_rng(
        42
    ).random(
        (128, 128),
        dtype=np.float32,
    )

    result = (
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=5000,
            block_size=8,
            safety_margin=0.10,
        )
    )

    assert (
        result.usable_position_count
        >= 5500
    )


def test_insufficient_capacity_is_rejected():
    heatmap = np.zeros(
        (16, 16),
        dtype=np.float32,
    )

    with pytest.raises(
        BlockMapError,
        match="does not provide enough capacity",
    ):
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=1000,
            block_size=8,
            safety_margin=0,
        )


def test_zero_payload_selects_no_blocks():
    heatmap = np.ones(
        (32, 32),
        dtype=np.float32,
    )

    result = (
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=0,
            block_size=8,
            safety_margin=0,
        )
    )

    assert result.selected_block_count == 0
    assert not np.any(
        result.selected_blocks
    )


def test_invalid_heatmap_shape_is_rejected():
    invalid_heatmap = np.zeros(
        (32, 32, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        BlockMapError,
        match="H×W",
    ):
        calculate_block_scores(
            invalid_heatmap
        )


def test_safety_margin_has_no_floating_point_error():
    heatmap = np.ones(
        (512, 512),
        dtype=np.float32,
    )

    result = (
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=50_000,
            block_size=8,
            safety_margin=0.10,
        )
    )

    assert (
        result.target_position_count
        == 55_000
    )

    assert (
        result.usable_position_count
        >= 55_000
    )