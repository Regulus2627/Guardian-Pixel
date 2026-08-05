"""Overlapping tiled HED inference for large cover images."""

from __future__ import annotations

from time import perf_counter

import numpy as np


class HEDTilingError(ValueError):
    """Raised when tiled HED inference cannot be completed."""


def calculate_tile_starts(
    length: int,
    tile_size: int,
    overlap: int,
) -> list[int]:
    """
    Calculate tile starting positions for one image dimension.

    The final tile is forced to end at the image boundary.
    """

    if length <= 0:
        raise HEDTilingError(
            "Image dimension must be greater than zero."
        )

    if tile_size <= 0:
        raise HEDTilingError(
            "tile_size must be greater than zero."
        )

    if overlap < 0:
        raise HEDTilingError(
            "overlap cannot be negative."
        )

    if overlap >= tile_size:
        raise HEDTilingError(
            "overlap must be smaller than tile_size."
        )

    if length <= tile_size:
        return [0]

    stride = tile_size - overlap

    starts = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    final_start = length - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def create_blending_window(
    height: int,
    width: int,
) -> np.ndarray:
    """
    Create a smooth two-dimensional blending window.

    Overlapping tile predictions are multiplied by this window before
    being added. This reduces visible seams between tiles.
    """

    if height <= 0 or width <= 0:
        raise HEDTilingError(
            "Blending-window dimensions must be positive."
        )

    if height <= 2:
        vertical = np.ones(
            height,
            dtype=np.float32,
        )
    else:
        vertical = np.hanning(
            height
        ).astype(np.float32)

    if width <= 2:
        horizontal = np.ones(
            width,
            dtype=np.float32,
        )
    else:
        horizontal = np.hanning(
            width
        ).astype(np.float32)

    window = np.outer(
        vertical,
        horizontal,
    ).astype(np.float32)

    # A pure Hann window becomes zero at its outer border.
    # A small minimum keeps full-image border pixels valid.
    return np.clip(
        window,
        1e-3,
        None,
    ).astype(np.float32)


class TiledHEDInference:
    """Run HED on overlapping tiles and blend their output maps."""

    def __init__(
        self,
        base_inference,
        tile_size: int = 512,
        overlap: int = 32,
    ):
        if base_inference is None:
            raise HEDTilingError(
                "base_inference cannot be None."
            )

        if tile_size <= 0:
            raise HEDTilingError(
                "tile_size must be greater than zero."
            )

        if overlap < 0:
            raise HEDTilingError(
                "overlap cannot be negative."
            )

        if overlap >= tile_size:
            raise HEDTilingError(
                "overlap must be smaller than tile_size."
            )

        self.base_inference = base_inference
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)

        self.last_inference_seconds = 0.0
        self.last_tile_count = 0

    def predict(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:
        """Generate a full-size tiled HED probability map."""

        if not isinstance(rgb, np.ndarray):
            raise HEDTilingError(
                "Tiled HED input must be a NumPy array."
            )

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise HEDTilingError(
                "Tiled HED input must have shape H×W×3."
            )

        if rgb.size == 0:
            raise HEDTilingError(
                "Tiled HED input cannot be empty."
            )

        if rgb.dtype != np.uint8:
            raise HEDTilingError(
                "Tiled HED input must use uint8 values."
            )

        height, width = rgb.shape[:2]

        y_starts = calculate_tile_starts(
            length=height,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

        x_starts = calculate_tile_starts(
            length=width,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

        accumulated_edges = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        accumulated_weights = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        tile_count = 0
        start_time = perf_counter()

        for y_start in y_starts:
            y_end = min(
                y_start + self.tile_size,
                height,
            )

            for x_start in x_starts:
                x_end = min(
                    x_start + self.tile_size,
                    width,
                )

                tile = rgb[
                    y_start:y_end,
                    x_start:x_end,
                ]

                tile_edges = (
                    self.base_inference.predict(tile)
                )

                expected_shape = (
                    y_end - y_start,
                    x_end - x_start,
                )

                if tile_edges.shape != expected_shape:
                    raise HEDTilingError(
                        "Tile HED output shape does not "
                        "match the tile input shape."
                    )

                if not np.all(
                    np.isfinite(tile_edges)
                ):
                    raise HEDTilingError(
                        "Tile HED output contains "
                        "invalid values."
                    )

                blending_window = (
                    create_blending_window(
                        height=expected_shape[0],
                        width=expected_shape[1],
                    )
                )

                accumulated_edges[
                    y_start:y_end,
                    x_start:x_end,
                ] += (
                    tile_edges
                    * blending_window
                )

                accumulated_weights[
                    y_start:y_end,
                    x_start:x_end,
                ] += blending_window

                tile_count += 1

        self.last_inference_seconds = (
            perf_counter() - start_time
        )

        self.last_tile_count = tile_count

        if np.any(accumulated_weights <= 0):
            raise HEDTilingError(
                "Some image pixels received no tile weight."
            )

        final_map = (
            accumulated_edges
            / accumulated_weights
        )

        final_map = np.clip(
            final_map,
            0.0,
            1.0,
        ).astype(np.float32)

        if final_map.shape != (height, width):
            raise HEDTilingError(
                "Final tiled HED map does not match "
                "the original image dimensions."
            )

        if not np.all(np.isfinite(final_map)):
            raise HEDTilingError(
                "Final tiled HED map contains "
                "invalid values."
            )

        return final_map