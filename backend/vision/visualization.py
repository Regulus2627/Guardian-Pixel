"""Visualization utilities for GuardianPixel feature maps."""

from __future__ import annotations

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