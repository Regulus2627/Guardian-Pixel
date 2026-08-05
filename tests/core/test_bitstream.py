import numpy as np
import pytest

from backend.core.bitstream import (
    BitstreamError,
    bits_to_bytes,
    bits_to_integer,
    bytes_to_bits,
    integer_to_bits,
)


def test_bytes_round_trip():
    original = b"GuardianPixel"

    bits = bytes_to_bits(original)
    recovered = bits_to_bytes(bits)

    assert recovered == original


def test_empty_bytes_round_trip():
    bits = bytes_to_bits(b"")

    assert bits.size == 0
    assert bits_to_bytes(bits) == b""


def test_known_byte_bit_order():
    bits = bytes_to_bits(b"A")

    expected = np.array(
        [0, 1, 0, 0, 0, 0, 0, 1],
        dtype=np.uint8,
    )

    assert np.array_equal(
        bits,
        expected,
    )


def test_random_binary_round_trip():
    generator = np.random.default_rng(
        42
    )

    original = generator.integers(
        0,
        256,
        size=1024,
        dtype=np.uint8,
    ).tobytes()

    recovered = bits_to_bytes(
        bytes_to_bits(original)
    )

    assert recovered == original


def test_integer_round_trip():
    original = 50_000

    bits = integer_to_bits(
        original,
        width=32,
    )

    recovered = bits_to_integer(bits)

    assert recovered == original


def test_integer_width_is_preserved():
    bits = integer_to_bits(
        value=5,
        width=8,
    )

    assert bits.size == 8

    assert np.array_equal(
        bits,
        np.array(
            [0, 0, 0, 0, 0, 1, 0, 1],
            dtype=np.uint8,
        ),
    )


def test_invalid_bit_value_is_rejected():
    with pytest.raises(
        BitstreamError,
        match="0 or 1",
    ):
        bits_to_bytes(
            [0, 1, 2, 0, 0, 0, 0, 0]
        )


def test_non_byte_aligned_bits_are_rejected():
    with pytest.raises(
        BitstreamError,
        match="divisible by 8",
    ):
        bits_to_bytes(
            [0, 1, 0]
        )


def test_integer_overflow_is_rejected():
    with pytest.raises(
        BitstreamError,
        match="does not fit",
    ):
        integer_to_bits(
            value=256,
            width=8,
        )


def test_negative_integer_is_rejected():
    with pytest.raises(
        BitstreamError,
        match="cannot be negative",
    ):
        integer_to_bits(
            value=-1,
            width=8,
        )