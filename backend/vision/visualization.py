"""Visualization utilities for GuardianPixel."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class VisualizationError(ValueError):
    """Raised when a visualization cannot be generated."""


def feature_map_to_uint8(
    feature_map: np.ndarray,
) -> np.ndarray:
    """Convert a normalized map to uint8 grayscale."""

    if not isinstance(feature_map, np.ndarray):
        raise VisualizationError(
            "Feature map must be a NumPy array."
        )

    if feature_map.ndim != 2:
        raise VisualizationError(
            "Feature map must have shape H×W."
        )

    if feature_map.size == 0:
        raise VisualizationError(
            "Feature map cannot be empty."
        )

    if not np.all(np.isfinite(feature_map)):
        raise VisualizationError(
            "Feature map contains invalid values."
        )

    clipped = np.clip(
        feature_map.astype(np.float32),
        0.0,
        1.0,
    )

    return np.rint(
        clipped * 255
    ).astype(np.uint8)


def save_grayscale_map(
    feature_map: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a normalized feature map as grayscale PNG."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    uint8_map = feature_map_to_uint8(
        feature_map
    )

    Image.fromarray(uint8_map).save(path)

    return path


def save_color_heatmap(
    feature_map: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a normalized feature map using the Turbo colour map."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    uint8_map = feature_map_to_uint8(
        feature_map
    )

    bgr_heatmap = cv2.applyColorMap(
        uint8_map,
        cv2.COLORMAP_TURBO,
    )

    rgb_heatmap = cv2.cvtColor(
        bgr_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    Image.fromarray(rgb_heatmap).save(path)

    return path


def save_binary_block_map(
    selected_blocks: np.ndarray,
    output_path: str | Path,
    display_scale: int = 8,
) -> Path:
    """Save a Boolean block map as a visible PNG."""

    if not isinstance(
        selected_blocks,
        np.ndarray,
    ):
        raise VisualizationError(
            "Block map must be a NumPy array."
        )

    if selected_blocks.ndim != 2:
        raise VisualizationError(
            "Block map must have shape rows×columns."
        )

    if selected_blocks.size == 0:
        raise VisualizationError(
            "Block map cannot be empty."
        )

    if display_scale <= 0:
        raise VisualizationError(
            "display_scale must be positive."
        )

    block_image = (
        selected_blocks.astype(np.uint8)
        * 255
    )

    visible_map = np.repeat(
        np.repeat(
            block_image,
            display_scale,
            axis=0,
        ),
        display_scale,
        axis=1,
    )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(visible_map).save(path)

    return path


def save_selected_blocks_overlay(
    rgb: np.ndarray,
    selected_blocks: np.ndarray,
    block_size: int,
    output_path: str | Path,
    alpha: float = 0.30,
) -> Path:
    """Overlay selected blocks in green on the cover image."""

    if not isinstance(rgb, np.ndarray):
        raise VisualizationError(
            "RGB image must be a NumPy array."
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise VisualizationError(
            "RGB image must have shape H×W×3."
        )

    if rgb.dtype != np.uint8:
        raise VisualizationError(
            "RGB image must use uint8 values."
        )

    if not isinstance(
        selected_blocks,
        np.ndarray,
    ):
        raise VisualizationError(
            "Block map must be a NumPy array."
        )

    if selected_blocks.ndim != 2:
        raise VisualizationError(
            "Block map must have shape rows×columns."
        )

    if block_size <= 0:
        raise VisualizationError(
            "block_size must be positive."
        )

    if not 0 <= alpha <= 1:
        raise VisualizationError(
            "alpha must be between 0 and 1."
        )

    height, width = rgb.shape[:2]

    expected_rows = math.ceil(
        height / block_size
    )

    expected_columns = math.ceil(
        width / block_size
    )

    if selected_blocks.shape != (
        expected_rows,
        expected_columns,
    ):
        raise VisualizationError(
            "Block-map dimensions do not match "
            "the RGB image."
        )

    overlay = rgb.astype(
        np.float32
    ).copy()

    green = np.array(
        [0, 255, 0],
        dtype=np.float32,
    )

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
            height,
        )

        x_start = int(
            block_column * block_size
        )

        x_end = min(
            x_start + block_size,
            width,
        )

        original_block = overlay[
            y_start:y_end,
            x_start:x_end,
        ]

        overlay[
            y_start:y_end,
            x_start:x_end,
        ] = (
            (1.0 - alpha)
            * original_block
            + alpha
            * green
        )

    overlay_uint8 = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        overlay_uint8
    ).save(path)

    return path