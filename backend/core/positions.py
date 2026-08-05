"""Deterministic GuardianPixel embedding-position generation."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np

from backend.core.protocol import (
    BOOTSTRAP_SIZE,
)


class PositionError(ValueError):
    """Raised when embedding positions cannot be generated."""


RGB_CHANNEL_COUNT = 3
BOOTSTRAP_BIT_COUNT = BOOTSTRAP_SIZE * 8


def flatten_channel_position(
    row: int,
    column: int,
    channel: int,
    image_width: int,
) -> int:
    """Convert row, column and channel into one integer position."""

    if row < 0 or column < 0:
        raise PositionError(
            "Row and column cannot be negative."
        )

    if image_width <= 0:
        raise PositionError(
            "Image width must be positive."
        )

    if not 0 <= channel < RGB_CHANNEL_COUNT:
        raise PositionError(
            "RGB channel must be 0, 1 or 2."
        )

    return (
        (
            row * image_width
            + column
        )
        * RGB_CHANNEL_COUNT
        + channel
    )


def unflatten_channel_position(
    position: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int]:
    """Convert a flattened position back to row, column and channel."""

    if image_width <= 0 or image_height <= 0:
        raise PositionError(
            "Image dimensions must be positive."
        )

    maximum_positions = (
        image_width
        * image_height
        * RGB_CHANNEL_COUNT
    )

    if not 0 <= position < maximum_positions:
        raise PositionError(
            "Flattened position is outside the image."
        )

    pixel_index, channel = divmod(
        position,
        RGB_CHANNEL_COUNT,
    )

    row, column = divmod(
        pixel_index,
        image_width,
    )

    return row, column, channel


@dataclass
class HMACRandom:
    """Small deterministic HMAC-SHA256 random stream."""

    key: bytes
    domain: bytes

    def __post_init__(self):
        if not isinstance(self.key, bytes):
            raise PositionError(
                "Position key must be bytes."
            )

        if len(self.key) < 16:
            raise PositionError(
                "Position key must contain at least 16 bytes."
            )

        if not isinstance(self.domain, bytes):
            raise PositionError(
                "Position domain must be bytes."
            )

        if not self.domain:
            raise PositionError(
                "Position domain cannot be empty."
            )

        self._counter = 0
        self._buffer = bytearray()

    def _refill(self) -> None:
        message = (
            self.domain
            + self._counter.to_bytes(
                8,
                byteorder="big",
            )
        )

        digest = hmac.new(
            self.key,
            message,
            hashlib.sha256,
        ).digest()

        self._buffer.extend(digest)
        self._counter += 1

    def random_uint64(self) -> int:
        """Return one deterministic unsigned 64-bit value."""

        while len(self._buffer) < 8:
            self._refill()

        value_bytes = bytes(
            self._buffer[:8]
        )

        del self._buffer[:8]

        return int.from_bytes(
            value_bytes,
            byteorder="big",
        )

    def randbelow(
        self,
        upper_bound: int,
    ) -> int:
        """Return an unbiased value from 0 to upper_bound - 1."""

        if upper_bound <= 0:
            raise PositionError(
                "upper_bound must be positive."
            )

        value_space = 1 << 64

        acceptable_limit = (
            value_space
            - value_space % upper_bound
        )

        while True:
            value = self.random_uint64()

            if value < acceptable_limit:
                return value % upper_bound


def generate_keyed_positions(
    candidate_positions: np.ndarray,
    count: int,
    key: bytes,
    domain: bytes,
    excluded_positions=(),
) -> np.ndarray:
    """
    Select deterministic unique positions without replacement.

    A partial Fisher-Yates shuffle is used so selecting a large number
    of positions remains efficient.
    """

    candidates = np.asarray(
        candidate_positions,
        dtype=np.int64,
    )

    if candidates.ndim != 1:
        raise PositionError(
            "Candidate positions must be one-dimensional."
        )

    if candidates.size == 0 and count > 0:
        raise PositionError(
            "Candidate positions cannot be empty."
        )

    if count < 0:
        raise PositionError(
            "Position count cannot be negative."
        )

    if candidates.size > 0:
        unique_candidates = np.unique(
            candidates
        )

        if (
            unique_candidates.size
            != candidates.size
        ):
            raise PositionError(
                "Candidate positions must be unique."
            )

    excluded_set = {
        int(position)
        for position in excluded_positions
    }

    if excluded_set:
        keep_mask = np.fromiter(
            (
                int(position)
                not in excluded_set
                for position in candidates
            ),
            dtype=bool,
            count=candidates.size,
        )

        available = candidates[
            keep_mask
        ]
    else:
        available = candidates.copy()

    if count > available.size:
        raise PositionError(
            "Not enough available positions."
        )

    if count == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    random_stream = HMACRandom(
        key=key,
        domain=domain,
    )

    # Partial Fisher-Yates shuffle using an index-swap dictionary.
    swaps: dict[int, int] = {}

    selected = np.empty(
        count,
        dtype=np.int64,
    )

    total_available = int(
        available.size
    )

    for index in range(count):
        random_index = (
            index
            + random_stream.randbelow(
                total_available - index
            )
        )

        mapped_random = swaps.get(
            random_index,
            random_index,
        )

        mapped_current = swaps.get(
            index,
            index,
        )

        swaps[random_index] = (
            mapped_current
        )

        swaps[index] = mapped_random

        selected[index] = available[
            mapped_random
        ]

    return selected


def generate_public_bootstrap_positions(
    image_height: int,
    image_width: int,
    count: int = BOOTSTRAP_BIT_COUNT,
) -> np.ndarray:
    """
    Generate public deterministic bootstrap channel positions.

    These positions depend only on image geometry and protocol version.
    """

    if image_height <= 0 or image_width <= 0:
        raise PositionError(
            "Image dimensions must be positive."
        )

    total_channel_positions = (
        image_height
        * image_width
        * RGB_CHANNEL_COUNT
    )

    if count > total_channel_positions:
        raise PositionError(
            "Image is too small for the bootstrap."
        )

    public_seed = hashlib.sha256(
        (
            f"GuardianPixel-GPX1-bootstrap:"
            f"{image_height}x{image_width}"
        ).encode("ascii")
    ).digest()

    all_positions = np.arange(
        total_channel_positions,
        dtype=np.int64,
    )

    return generate_keyed_positions(
        candidate_positions=all_positions,
        count=count,
        key=public_seed,
        domain=b"public-bootstrap",
    )


def generate_metadata_positions(
    image_height: int,
    image_width: int,
    metadata_bit_count: int,
    position_key: bytes,
    bootstrap_positions: np.ndarray,
) -> np.ndarray:
    """Generate keyed metadata positions outside the bootstrap."""

    if image_height <= 0 or image_width <= 0:
        raise PositionError(
            "Image dimensions must be positive."
        )

    total_channel_positions = (
        image_height
        * image_width
        * RGB_CHANNEL_COUNT
    )

    all_positions = np.arange(
        total_channel_positions,
        dtype=np.int64,
    )

    return generate_keyed_positions(
        candidate_positions=all_positions,
        count=metadata_bit_count,
        key=position_key,
        domain=b"metadata-positions",
        excluded_positions=(
            bootstrap_positions
        ),
    )


def build_selected_block_candidates(
    selected_blocks: np.ndarray,
    image_height: int,
    image_width: int,
    block_size: int,
    position_key: bytes,
) -> np.ndarray:
    """
    Create one deterministic RGB-channel position per selected pixel.

    The block map is traversed in row-major order. The position key
    chooses one of R, G or B for every eligible pixel.
    """

    if not isinstance(
        selected_blocks,
        np.ndarray,
    ):
        raise PositionError(
            "Selected block map must be a NumPy array."
        )

    if selected_blocks.ndim != 2:
        raise PositionError(
            "Selected block map must have shape rows×columns."
        )

    if image_height <= 0 or image_width <= 0:
        raise PositionError(
            "Image dimensions must be positive."
        )

    if block_size <= 0:
        raise PositionError(
            "block_size must be positive."
        )

    expected_rows = (
        image_height
        + block_size
        - 1
    ) // block_size

    expected_columns = (
        image_width
        + block_size
        - 1
    ) // block_size

    if selected_blocks.shape != (
        expected_rows,
        expected_columns,
    ):
        raise PositionError(
            "Selected block-map dimensions do not "
            "match the image."
        )

    channel_stream = HMACRandom(
        key=position_key,
        domain=b"selected-pixel-channels",
    )

    candidates = []

    selected_indices = np.argwhere(
        selected_blocks
    )

    for block_row, block_column in (
        selected_indices
    ):
        y_start = int(
            block_row * block_size
        )

        y_end = min(
            y_start + block_size,
            image_height,
        )

        x_start = int(
            block_column * block_size
        )

        x_end = min(
            x_start + block_size,
            image_width,
        )

        for row in range(
            y_start,
            y_end,
        ):
            for column in range(
                x_start,
                x_end,
            ):
                channel = (
                    channel_stream.randbelow(3)
                )

                candidates.append(
                    flatten_channel_position(
                        row=row,
                        column=column,
                        channel=channel,
                        image_width=image_width,
                    )
                )

    return np.asarray(
        candidates,
        dtype=np.int64,
    )


def generate_payload_positions(
    selected_blocks: np.ndarray,
    image_height: int,
    image_width: int,
    block_size: int,
    payload_bit_count: int,
    position_key: bytes,
    excluded_positions=(),
) -> np.ndarray:
    """Generate exact keyed payload positions inside selected blocks."""

    candidates = (
        build_selected_block_candidates(
            selected_blocks=selected_blocks,
            image_height=image_height,
            image_width=image_width,
            block_size=block_size,
            position_key=position_key,
        )
    )

    return generate_keyed_positions(
        candidate_positions=candidates,
        count=payload_bit_count,
        key=position_key,
        domain=b"payload-positions",
        excluded_positions=(
            excluded_positions
        ),
    )