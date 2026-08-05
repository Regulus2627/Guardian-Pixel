import numpy as np
import pytest

from backend.vision.map_encoding import (
    MapEncodingError,
    decode_block_map,
    encode_block_map,
)


@pytest.mark.parametrize(
    "block_map",
    [
        np.zeros(
            (64, 64),
            dtype=bool,
        ),
        np.ones(
            (64, 64),
            dtype=bool,
        ),
        np.indices(
            (32, 32)
        ).sum(axis=0) % 2 == 0,
        np.random.default_rng(
            42
        ).random((32, 32)) > 0.8,
        np.random.default_rng(
            100
        ).random((32, 32)) > 0.5,
    ],
)
def test_map_round_trip(block_map):
    encoded = encode_block_map(
        block_map
    )

    decoded = decode_block_map(
        encoding_type=(
            encoded.encoding_type
        ),
        encoded_data=(
            encoded.encoded_data
        ),
        block_rows=block_map.shape[0],
        block_columns=block_map.shape[1],
    )

    assert np.array_equal(
        decoded,
        block_map,
    )


def test_all_selected_uses_special_encoding():
    block_map = np.ones(
        (16, 16),
        dtype=bool,
    )

    encoded = encode_block_map(
        block_map
    )

    assert (
        encoded.encoding_type
        == "ALL_SELECTED"
    )

    assert encoded.encoded_byte_count == 0


def test_none_selected_uses_special_encoding():
    block_map = np.zeros(
        (16, 16),
        dtype=bool,
    )

    encoded = encode_block_map(
        block_map
    )

    assert (
        encoded.encoding_type
        == "NONE_SELECTED"
    )

    assert encoded.encoded_byte_count == 0


def test_encoded_size_is_not_larger_than_raw_candidate():
    block_map = np.zeros(
        (128, 128),
        dtype=bool,
    )

    block_map[20:40, 30:60] = True

    encoded = encode_block_map(
        block_map
    )

    assert (
        encoded.encoded_byte_count
        <= encoded.raw_byte_count
    )


def test_invalid_map_shape_is_rejected():
    invalid_map = np.zeros(
        (10, 10, 3),
        dtype=bool,
    )

    with pytest.raises(
        MapEncodingError,
        match="rows×columns",
    ):
        encode_block_map(invalid_map)


def test_unknown_encoding_is_rejected():
    with pytest.raises(
        MapEncodingError,
        match="Unsupported",
    ):
        decode_block_map(
            encoding_type="UNKNOWN",
            encoded_data=b"",
            block_rows=10,
            block_columns=10,
        )