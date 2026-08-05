"""Bit and byte conversion utilities for GuardianPixel."""

from __future__ import annotations

import numpy as np


class BitstreamError(ValueError):
    """Raised when bitstream data is invalid."""


def bytes_to_bits(
    data: bytes | bytearray,
) -> np.ndarray:
    """
    Convert bytes to a one-dimensional uint8 bit array.

    Bit order is most-significant-bit first.
    """

    if not isinstance(
        data,
        (bytes, bytearray),
    ):
        raise BitstreamError(
            "Input must be bytes or bytearray."
        )

    byte_array = np.frombuffer(
        bytes(data),
        dtype=np.uint8,
    )

    return np.unpackbits(
        byte_array,
        bitorder="big",
    ).astype(np.uint8)


def _validate_bits(
    bits,
) -> np.ndarray:
    """Validate and return a one-dimensional uint8 bit array."""

    bit_array = np.asarray(bits)

    if bit_array.ndim != 1:
        raise BitstreamError(
            "Bits must be a one-dimensional sequence."
        )

    if bit_array.size > 0 and not np.all(
        (bit_array == 0)
        | (bit_array == 1)
    ):
        raise BitstreamError(
            "Bit values must be 0 or 1."
        )

    return bit_array.astype(
        np.uint8,
        copy=False,
    )


def bits_to_bytes(
    bits,
) -> bytes:
    """
    Convert a bit sequence to bytes.

    The number of bits must be divisible by eight.
    """

    bit_array = _validate_bits(bits)

    if bit_array.size % 8 != 0:
        raise BitstreamError(
            "Bit count must be divisible by 8."
        )

    if bit_array.size == 0:
        return b""

    packed = np.packbits(
        bit_array,
        bitorder="big",
    )

    return packed.tobytes()


def integer_to_bits(
    value: int,
    width: int,
) -> np.ndarray:
    """Convert a non-negative integer to fixed-width bits."""

    if not isinstance(value, int):
        raise BitstreamError(
            "Integer value must be an int."
        )

    if value < 0:
        raise BitstreamError(
            "Integer value cannot be negative."
        )

    if width <= 0:
        raise BitstreamError(
            "Bit width must be greater than zero."
        )

    maximum_value = (
        1 << width
    ) - 1

    if value > maximum_value:
        raise BitstreamError(
            f"Value {value} does not fit "
            f"inside {width} bits."
        )

    bit_string = format(
        value,
        f"0{width}b",
    )

    return np.fromiter(
        (
            int(character)
            for character in bit_string
        ),
        dtype=np.uint8,
        count=width,
    )


def bits_to_integer(
    bits,
) -> int:
    """Convert a most-significant-bit-first sequence to an integer."""

    bit_array = _validate_bits(bits)

    if bit_array.size == 0:
        raise BitstreamError(
            "Cannot convert an empty bit sequence."
        )

    value = 0

    for bit in bit_array:
        value = (
            value << 1
        ) | int(bit)

    return value