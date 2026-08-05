"""Adaptive LSB matching and extraction for GuardianPixel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.positions import (
    HMACRandom,
)


class LSBMatchingError(ValueError):
    """Raised when LSB embedding or extraction cannot be completed."""


@dataclass(frozen=True, slots=True)
class LSBEmbeddingResult:
    """Result of embedding bits into an RGB image."""

    stego_image: np.ndarray
    embedded_bit_count: int
    modified_channel_count: int
    unchanged_channel_count: int
    maximum_absolute_change: int


def _validate_rgb_image(
    rgb: np.ndarray,
) -> None:
    if not isinstance(rgb, np.ndarray):
        raise LSBMatchingError(
            "RGB image must be a NumPy array."
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise LSBMatchingError(
            "RGB image must have shape H×W×3."
        )

    if rgb.size == 0:
        raise LSBMatchingError(
            "RGB image cannot be empty."
        )

    if rgb.dtype != np.uint8:
        raise LSBMatchingError(
            "RGB image must use uint8 values."
        )


def _validate_bits(
    bits,
) -> np.ndarray:
    bit_array = np.asarray(bits)

    if bit_array.ndim != 1:
        raise LSBMatchingError(
            "Bits must be a one-dimensional sequence."
        )

    if bit_array.size > 0 and not np.all(
        (bit_array == 0)
        | (bit_array == 1)
    ):
        raise LSBMatchingError(
            "Bit values must be 0 or 1."
        )

    return bit_array.astype(
        np.uint8,
        copy=False,
    )


def _validate_positions(
    positions,
    maximum_position_count: int,
) -> np.ndarray:
    position_array = np.asarray(
        positions,
        dtype=np.int64,
    )

    if position_array.ndim != 1:
        raise LSBMatchingError(
            "Positions must be one-dimensional."
        )

    if position_array.size > 0:
        if position_array.min() < 0:
            raise LSBMatchingError(
                "Positions cannot be negative."
            )

        if (
            position_array.max()
            >= maximum_position_count
        ):
            raise LSBMatchingError(
                "Position is outside the image."
            )

        if (
            np.unique(position_array).size
            != position_array.size
        ):
            raise LSBMatchingError(
                "Embedding positions must be unique."
            )

    return position_array


def embed_bits_at_positions(
    rgb: np.ndarray,
    bits,
    positions,
    direction_key: bytes,
) -> LSBEmbeddingResult:
    """
    Embed bits using ±1 LSB matching at exact channel positions.

    The original RGB array is not modified.
    """

    _validate_rgb_image(rgb)

    bit_array = _validate_bits(bits)

    position_array = _validate_positions(
        positions,
        maximum_position_count=rgb.size,
    )

    if bit_array.size != position_array.size:
        raise LSBMatchingError(
            "Bit count and position count must match."
        )

    if not isinstance(direction_key, bytes):
        raise LSBMatchingError(
            "direction_key must be bytes."
        )

    if len(direction_key) < 16:
        raise LSBMatchingError(
            "direction_key must contain at least 16 bytes."
        )

    stego_image = rgb.copy()
    flattened = stego_image.reshape(-1)

    direction_stream = HMACRandom(
        key=direction_key,
        domain=b"lsb-matching-directions",
    )

    modified_channel_count = 0

    for bit, position in zip(
        bit_array,
        position_array,
        strict=True,
    ):
        position_integer = int(position)
        required_bit = int(bit)

        current_value = int(
            flattened[position_integer]
        )

        current_bit = (
            current_value & 1
        )

        if current_bit == required_bit:
            continue

        if current_value == 0:
            new_value = 1

        elif current_value == 255:
            new_value = 254

        elif direction_stream.randbelow(2) == 0:
            new_value = current_value - 1

        else:
            new_value = current_value + 1

        flattened[position_integer] = (
            new_value
        )

        modified_channel_count += 1

    if bit_array.size == 0:
        maximum_change = 0
    else:
        difference = np.abs(
            stego_image.astype(np.int16)
            - rgb.astype(np.int16)
        )

        maximum_change = int(
            difference.max()
        )

    return LSBEmbeddingResult(
        stego_image=stego_image,
        embedded_bit_count=int(
            bit_array.size
        ),
        modified_channel_count=(
            modified_channel_count
        ),
        unchanged_channel_count=int(
            bit_array.size
            - modified_channel_count
        ),
        maximum_absolute_change=(
            maximum_change
        ),
    )


def extract_bits_at_positions(
    rgb: np.ndarray,
    positions,
) -> np.ndarray:
    """Extract LSB values from exact channel positions."""

    _validate_rgb_image(rgb)

    position_array = _validate_positions(
        positions,
        maximum_position_count=rgb.size,
    )

    flattened = rgb.reshape(-1)

    extracted = (
        flattened[position_array] & 1
    )

    return extracted.astype(
        np.uint8,
        copy=False,
    )