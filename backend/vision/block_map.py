"""Capacity-aware 8×8 block-map generation."""

from __future__ import annotations

import math

import numpy as np

from backend.vision.schemas import BlockMapResult


class BlockMapError(ValueError):
    """Raised when a block map cannot be generated."""


def _validate_heatmap(
    heatmap: np.ndarray,
) -> None:
    if not isinstance(heatmap, np.ndarray):
        raise BlockMapError(
            "Heatmap must be a NumPy array."
        )

    if heatmap.ndim != 2:
        raise BlockMapError(
            "Heatmap must have shape H×W."
        )

    if heatmap.size == 0:
        raise BlockMapError(
            "Heatmap cannot be empty."
        )

    if not np.all(np.isfinite(heatmap)):
        raise BlockMapError(
            "Heatmap contains invalid numerical values."
        )


def calculate_block_scores(
    heatmap: np.ndarray,
    block_size: int = 8,
    score_method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate suitability and valid pixel counts for every block.

    Border blocks may contain fewer pixels when image dimensions are
    not exactly divisible by the configured block size.
    """

    _validate_heatmap(heatmap)

    if block_size <= 0:
        raise BlockMapError(
            "block_size must be greater than zero."
        )

    if score_method not in {
        "mean",
        "percentile",
    }:
        raise BlockMapError(
            "score_method must be mean or percentile."
        )

    height, width = heatmap.shape

    block_rows = math.ceil(
        height / block_size
    )

    block_columns = math.ceil(
        width / block_size
    )

    block_scores = np.zeros(
        (block_rows, block_columns),
        dtype=np.float32,
    )

    block_pixel_counts = np.zeros(
        (block_rows, block_columns),
        dtype=np.int64,
    )

    for block_row in range(block_rows):
        y_start = block_row * block_size
        y_end = min(
            y_start + block_size,
            height,
        )

        for block_column in range(
            block_columns
        ):
            x_start = (
                block_column * block_size
            )

            x_end = min(
                x_start + block_size,
                width,
            )

            block = heatmap[
                y_start:y_end,
                x_start:x_end,
            ]

            if score_method == "mean":
                score = float(
                    np.mean(block)
                )
            else:
                score = float(
                    np.percentile(block, 75)
                )

            block_scores[
                block_row,
                block_column,
            ] = score

            block_pixel_counts[
                block_row,
                block_column,
            ] = block.size

    return block_scores, block_pixel_counts


def generate_capacity_aware_block_map(
    heatmap: np.ndarray,
    required_payload_bits: int,
    block_size: int = 8,
    channels_per_selected_pixel: int = 1,
    reserved_position_count: int = 0,
    safety_margin: float = 0.10,
    score_method: str = "mean",
) -> BlockMapResult:
    """
    Select the highest-scoring blocks until capacity is sufficient.

    required_payload_bits should eventually include the exact
    encrypted packet size provided by Member 2.
    """

    if required_payload_bits < 0:
        raise BlockMapError(
            "required_payload_bits cannot be negative."
        )

    if channels_per_selected_pixel <= 0:
        raise BlockMapError(
            "channels_per_selected_pixel must be positive."
        )

    if reserved_position_count < 0:
        raise BlockMapError(
            "reserved_position_count cannot be negative."
        )

    if safety_margin < 0:
        raise BlockMapError(
            "safety_margin cannot be negative."
        )

    block_scores, block_pixel_counts = (
        calculate_block_scores(
            heatmap=heatmap,
            block_size=block_size,
            score_method=score_method,
        )
    )

    payload_target = math.ceil(
        required_payload_bits
        * (1.0 + safety_margin)
    )

    total_target = (
        payload_target
        + reserved_position_count
    )

    block_capacities = (
        block_pixel_counts
        * channels_per_selected_pixel
    )

    maximum_capacity = int(
        np.sum(block_capacities)
    )

    if total_target > maximum_capacity:
        raise BlockMapError(
            "The image does not provide enough capacity. "
            f"Required positions: {total_target}; "
            f"maximum positions: {maximum_capacity}."
        )

    selected_blocks = np.zeros(
        block_scores.shape,
        dtype=bool,
    )

    selected_position_count = 0

    if total_target > 0:
        flat_scores = block_scores.ravel()

        ranked_indices = np.argsort(
            -flat_scores,
            kind="stable",
        )

        flat_selected = (
            selected_blocks.ravel()
        )

        flat_capacities = (
            block_capacities.ravel()
        )

        for flat_index in ranked_indices:
            flat_selected[flat_index] = True

            selected_position_count += int(
                flat_capacities[flat_index]
            )

            if (
                selected_position_count
                >= total_target
            ):
                break

    usable_position_count = max(
        0,
        selected_position_count
        - reserved_position_count,
    )

    return BlockMapResult(
        block_size=block_size,
        block_rows=block_scores.shape[0],
        block_columns=block_scores.shape[1],
        selected_blocks=selected_blocks,
        block_scores=block_scores,
        selected_block_count=int(
            np.count_nonzero(
                selected_blocks
            )
        ),
        total_block_count=int(
            selected_blocks.size
        ),
        selected_position_count=(
            selected_position_count
        ),
        reserved_position_count=(
            reserved_position_count
        ),
        usable_position_count=(
            usable_position_count
        ),
        required_payload_bits=(
            required_payload_bits
        ),
        target_position_count=total_target,
    )