import numpy as np
import pytest

from backend.core.positions import (
    BOOTSTRAP_BIT_COUNT,
    PositionError,
    build_selected_block_candidates,
    flatten_channel_position,
    generate_keyed_positions,
    generate_payload_positions,
    generate_public_bootstrap_positions,
    unflatten_channel_position,
)


POSITION_KEY = b"P" * 32


def test_flatten_round_trip():
    position = flatten_channel_position(
        row=10,
        column=20,
        channel=2,
        image_width=100,
    )

    recovered = (
        unflatten_channel_position(
            position=position,
            image_width=100,
            image_height=50,
        )
    )

    assert recovered == (10, 20, 2)


def test_bootstrap_positions_are_deterministic():
    first = (
        generate_public_bootstrap_positions(
            image_height=533,
            image_width=800,
        )
    )

    second = (
        generate_public_bootstrap_positions(
            image_height=533,
            image_width=800,
        )
    )

    assert np.array_equal(
        first,
        second,
    )


def test_bootstrap_positions_are_unique():
    positions = (
        generate_public_bootstrap_positions(
            image_height=533,
            image_width=800,
        )
    )

    assert positions.size == (
        BOOTSTRAP_BIT_COUNT
    )

    assert np.unique(
        positions
    ).size == positions.size


def test_bootstrap_positions_are_in_range():
    height = 100
    width = 120

    positions = (
        generate_public_bootstrap_positions(
            image_height=height,
            image_width=width,
        )
    )

    assert positions.min() >= 0

    assert positions.max() < (
        height * width * 3
    )


def test_keyed_positions_are_deterministic():
    candidates = np.arange(
        10_000,
        dtype=np.int64,
    )

    first = generate_keyed_positions(
        candidate_positions=candidates,
        count=1000,
        key=POSITION_KEY,
        domain=b"test-domain",
    )

    second = generate_keyed_positions(
        candidate_positions=candidates,
        count=1000,
        key=POSITION_KEY,
        domain=b"test-domain",
    )

    assert np.array_equal(
        first,
        second,
    )


def test_different_domains_change_positions():
    candidates = np.arange(
        10_000,
        dtype=np.int64,
    )

    first = generate_keyed_positions(
        candidates,
        count=1000,
        key=POSITION_KEY,
        domain=b"domain-one",
    )

    second = generate_keyed_positions(
        candidates,
        count=1000,
        key=POSITION_KEY,
        domain=b"domain-two",
    )

    assert not np.array_equal(
        first,
        second,
    )


def test_excluded_positions_are_not_selected():
    candidates = np.arange(
        1000,
        dtype=np.int64,
    )

    excluded = np.arange(
        100,
        dtype=np.int64,
    )

    selected = generate_keyed_positions(
        candidates,
        count=500,
        key=POSITION_KEY,
        domain=b"excluded-test",
        excluded_positions=excluded,
    )

    assert not np.any(
        np.isin(
            selected,
            excluded,
        )
    )


def test_selected_blocks_create_one_channel_per_pixel():
    selected_blocks = np.array(
        [
            [True, False],
            [False, False],
        ],
        dtype=bool,
    )

    candidates = (
        build_selected_block_candidates(
            selected_blocks=selected_blocks,
            image_height=16,
            image_width=16,
            block_size=8,
            position_key=POSITION_KEY,
        )
    )

    # One 8×8 block = 64 selected pixels.
    assert candidates.size == 64

    assert np.unique(
        candidates
    ).size == 64


def test_partial_border_block_capacity():
    selected_blocks = np.zeros(
        (2, 2),
        dtype=bool,
    )

    selected_blocks[-1, -1] = True

    candidates = (
        build_selected_block_candidates(
            selected_blocks=selected_blocks,
            image_height=10,
            image_width=10,
            block_size=8,
            position_key=POSITION_KEY,
        )
    )

    # Final block contains only 2×2 valid pixels.
    assert candidates.size == 4


def test_payload_positions_stay_inside_selected_blocks():
    selected_blocks = np.array(
        [
            [True, False],
            [False, False],
        ],
        dtype=bool,
    )

    positions = generate_payload_positions(
        selected_blocks=selected_blocks,
        image_height=16,
        image_width=16,
        block_size=8,
        payload_bit_count=32,
        position_key=POSITION_KEY,
    )

    for position in positions:
        row, column, channel = (
            unflatten_channel_position(
                int(position),
                image_width=16,
                image_height=16,
            )
        )

        assert row < 8
        assert column < 8
        assert channel in {0, 1, 2}


def test_insufficient_positions_are_rejected():
    candidates = np.arange(
        10,
        dtype=np.int64,
    )

    with pytest.raises(
        PositionError,
        match="Not enough",
    ):
        generate_keyed_positions(
            candidates,
            count=11,
            key=POSITION_KEY,
            domain=b"too-many",
        )