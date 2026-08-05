import numpy as np
import pytest

from backend.core.positions import (
    flatten_channel_position,
)
from backend.stego.lsb_matching import (
    LSBMatchingError,
    embed_bits_at_positions,
    extract_bits_at_positions,
)


DIRECTION_KEY = b"D" * 32


def test_embed_extract_round_trip():
    generator = np.random.default_rng(
        42
    )

    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )

    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )

    positions = np.arange(
        1000,
        dtype=np.int64,
    )

    result = embed_bits_at_positions(
        rgb=rgb,
        bits=bits,
        positions=positions,
        direction_key=DIRECTION_KEY,
    )

    extracted = extract_bits_at_positions(
        result.stego_image,
        positions,
    )

    assert np.array_equal(
        extracted,
        bits,
    )


def test_original_image_is_not_modified():
    rgb = np.full(
        (16, 16, 3),
        100,
        dtype=np.uint8,
    )

    original_copy = rgb.copy()

    bits = np.ones(
        100,
        dtype=np.uint8,
    )

    positions = np.arange(
        100,
        dtype=np.int64,
    )

    embed_bits_at_positions(
        rgb,
        bits,
        positions,
        DIRECTION_KEY,
    )

    assert np.array_equal(
        rgb,
        original_copy,
    )


def test_maximum_change_is_one():
    rgb = np.full(
        (32, 32, 3),
        100,
        dtype=np.uint8,
    )

    bits = np.ones(
        500,
        dtype=np.uint8,
    )

    positions = np.arange(
        500,
        dtype=np.int64,
    )

    result = embed_bits_at_positions(
        rgb,
        bits,
        positions,
        DIRECTION_KEY,
    )

    difference = np.abs(
        result.stego_image.astype(
            np.int16
        )
        - rgb.astype(np.int16)
    )

    assert difference.max() <= 1
    assert (
        result.maximum_absolute_change
        <= 1
    )


def test_boundary_values_are_safe():
    rgb = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    flattened = rgb.reshape(-1)

    flattened[0] = 0
    flattened[1] = 255

    bits = np.array(
        [1, 0],
        dtype=np.uint8,
    )

    positions = np.array(
        [0, 1],
        dtype=np.int64,
    )

    result = embed_bits_at_positions(
        rgb,
        bits,
        positions,
        DIRECTION_KEY,
    )

    result_flat = (
        result.stego_image.reshape(-1)
    )

    assert result_flat[0] == 1
    assert result_flat[1] == 254


def test_embedding_is_deterministic():
    rgb = np.full(
        (32, 32, 3),
        100,
        dtype=np.uint8,
    )

    bits = np.ones(
        500,
        dtype=np.uint8,
    )

    positions = np.arange(
        500,
        dtype=np.int64,
    )

    first = embed_bits_at_positions(
        rgb,
        bits,
        positions,
        DIRECTION_KEY,
    )

    second = embed_bits_at_positions(
        rgb,
        bits,
        positions,
        DIRECTION_KEY,
    )

    assert np.array_equal(
        first.stego_image,
        second.stego_image,
    )


def test_changed_channels_match_report():
    rgb = np.full(
        (16, 16, 3),
        100,
        dtype=np.uint8,
    )

    bits = np.ones(
        100,
        dtype=np.uint8,
    )

    positions = np.arange(
        100,
        dtype=np.int64,
    )

    result = embed_bits_at_positions(
        rgb,
        bits,
        positions,
        DIRECTION_KEY,
    )

    changed = np.count_nonzero(
        result.stego_image != rgb
    )

    assert (
        changed
        == result.modified_channel_count
    )


def test_bit_and_position_count_must_match():
    rgb = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        LSBMatchingError,
        match="must match",
    ):
        embed_bits_at_positions(
            rgb=rgb,
            bits=[0, 1, 0],
            positions=[0, 1],
            direction_key=DIRECTION_KEY,
        )


def test_duplicate_positions_are_rejected():
    rgb = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        LSBMatchingError,
        match="unique",
    ):
        embed_bits_at_positions(
            rgb=rgb,
            bits=[0, 1],
            positions=[5, 5],
            direction_key=DIRECTION_KEY,
        )


def test_out_of_range_position_is_rejected():
    rgb = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        LSBMatchingError,
        match="outside",
    ):
        extract_bits_at_positions(
            rgb,
            positions=[rgb.size],
        )


def test_specific_flattened_position():
    rgb = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    position = flatten_channel_position(
        row=4,
        column=5,
        channel=2,
        image_width=10,
    )

    result = embed_bits_at_positions(
        rgb=rgb,
        bits=[1],
        positions=[position],
        direction_key=DIRECTION_KEY,
    )

    assert result.stego_image[
        4,
        5,
        2,
    ] == 1