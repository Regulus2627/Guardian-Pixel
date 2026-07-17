"""Cover-image loading and preprocessing for GuardianPixel."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.vision.schemas import ImageInfo


class ImagePreprocessingError(ValueError):
    """Raised when a cover image cannot be safely processed."""


ImageSource = str | Path | bytes | bytearray | BinaryIO


def _read_source_bytes(source: ImageSource) -> bytes:
    """Read an image source into bytes."""

    if isinstance(source, (str, Path)):
        path = Path(source)

        if not path.exists():
            raise ImagePreprocessingError(
                f"Image file was not found: {path}"
            )

        if not path.is_file():
            raise ImagePreprocessingError(
                f"Image path is not a file: {path}"
            )

        try:
            data = path.read_bytes()
        except OSError as error:
            raise ImagePreprocessingError(
                f"Could not read image file: {error}"
            ) from error

    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)

    elif hasattr(source, "read"):
        try:
            data = source.read()
        except OSError as error:
            raise ImagePreprocessingError(
                f"Could not read image stream: {error}"
            ) from error

    else:
        raise ImagePreprocessingError(
            "Image source must be a path, bytes or readable stream."
        )

    if not data:
        raise ImagePreprocessingError("The supplied image is empty.")

    return data


def load_and_validate_image(
    source: ImageSource,
    max_pixels: int,
) -> tuple[np.ndarray, ImageInfo]:
    """
    Load a cover image and convert it into a standard RGB array.

    Returns:
        RGB uint8 NumPy array with shape H×W×3.
        ImageInfo containing original image information.
    """

    if max_pixels <= 0:
        raise ImagePreprocessingError(
            "max_pixels must be greater than zero."
        )

    data = _read_source_bytes(source)

    try:
        with Image.open(io.BytesIO(data)) as image:
            original_format = image.format or "UNKNOWN"
            original_mode = image.mode

            # Apply camera/phone orientation before removing EXIF data.
            corrected_image = ImageOps.exif_transpose(image)

            width, height = corrected_image.size

            if width <= 0 or height <= 0:
                raise ImagePreprocessingError(
                    "Image dimensions must be greater than zero."
                )

            total_pixels = width * height

            if total_pixels > max_pixels:
                raise ImagePreprocessingError(
                    f"Image contains {total_pixels:,} pixels, "
                    f"which exceeds the limit of {max_pixels:,}."
                )

            # Produces a consistent three-channel image and removes alpha.
            rgb_image = corrected_image.convert("RGB")

            # copy() separates the array from Pillow's internal buffer.
            rgb_array = np.asarray(rgb_image, dtype=np.uint8).copy()

    except ImagePreprocessingError:
        raise

    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImagePreprocessingError(
            "The supplied file is not a valid supported image."
        ) from error

    expected_shape = (height, width, 3)

    if rgb_array.shape != expected_shape:
        raise ImagePreprocessingError(
            f"Unexpected RGB image shape: {rgb_array.shape}."
        )

    image_info = ImageInfo(
        width=width,
        height=height,
        channels=3,
        original_format=original_format.upper(),
        original_mode=original_mode,
    )

    return rgb_array, image_info


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB uint8 image to uint8 luminance.

    Uses the standard weighted luminance approximation:
    Y = 0.299R + 0.587G + 0.114B
    """

    if not isinstance(rgb, np.ndarray):
        raise ImagePreprocessingError(
            "RGB input must be a NumPy array."
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImagePreprocessingError(
            "RGB input must have shape H×W×3."
        )

    if rgb.dtype != np.uint8:
        raise ImagePreprocessingError(
            "RGB input must use uint8 values."
        )

    rgb_float = rgb.astype(np.float32)

    luminance = (
        0.299 * rgb_float[:, :, 0]
        + 0.587 * rgb_float[:, :, 1]
        + 0.114 * rgb_float[:, :, 2]
    )

    return np.clip(
        np.rint(luminance),
        0,
        255,
    ).astype(np.uint8)