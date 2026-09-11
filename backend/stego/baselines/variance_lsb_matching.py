"""Variance-only LSB matching and extraction for GuardianPixel."""

import hashlib

import numpy as np

from backend.core.positions import PositionError, flatten_channel_position
from backend.stego.lsb_matching import (
    LSBMatchingError,
    embed_bits_at_positions,
    extract_bits_at_positions,
)
from backend.vision.preprocessing import ImagePreprocessingError, rgb_to_luminance
from backend.vision.variance_map import VarianceMapError, calculate_local_variance

__all__ = ["embed_variance_matching", "extract_variance_matching"]

_DIRECTION_KEY = hashlib.sha256(
    b"GuardianPixel-baseline-variance-matching"
).digest()

# Fixed for the batch-runner interface: embed/extract only expose
# (rgb, bits, channel), so window_size is pinned here rather than
# being caller-configurable.
_WINDOW_SIZE = 11  # matches calculate_local_variance's own default


def _validate_rgb(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise ValueError(f"rgb must be a numpy ndarray, got {type(rgb).__name__}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {rgb.dtype}")


def _validate_channel(channel: int) -> None:
    if channel not in (0, 1, 2):
        raise ValueError(f"channel must be 0, 1, or 2, got {channel!r}")


def _validate_and_convert_bits(bits) -> np.ndarray:
    bits_arr = np.asarray(bits)
    if bits_arr.ndim != 1:
        raise ValueError(f"bits must be a 1-D sequence, got shape {bits_arr.shape}")
    if bits_arr.size == 0:
        raise ValueError("bits must not be empty")
    if not np.all((bits_arr == 0) | (bits_arr == 1)):
        raise ValueError("bits must contain only 0s and 1s")
    return bits_arr.astype(np.uint8)


def _validate_num_bits(num_bits: int) -> None:
    if not isinstance(num_bits, int) or num_bits <= 0:
        raise ValueError(f"num_bits must be a positive integer, got {num_bits!r}")


def _select_variance_positions(
    rgb: np.ndarray, num_positions: int, channel: int
) -> np.ndarray:
    _, w, _ = rgb.shape

    try:
        luminance = rgb_to_luminance(rgb)
    except ImagePreprocessingError as error:
        # _validate_rgb already ran before this is called, so this path is
        # defensive. Wrapped for a consistent ValueError boundary.
        raise ValueError(f"luminance conversion failed: {error}") from error

    try:
        variance_map = calculate_local_variance(luminance, _WINDOW_SIZE)
    except VarianceMapError:
        raise

    flat_variance = variance_map.reshape(-1)

    if num_positions > flat_variance.size:
        raise ValueError(
            f"payload requires {num_positions} pixels but the image only "
            f"has {flat_variance.size} pixels available."
        )

    order = np.argsort(-flat_variance, kind="stable")
    top_indices = order[:num_positions]

    rows = top_indices // w
    cols = top_indices % w

    try:
        return np.array(
            [flatten_channel_position(r, c, channel, w) for r, c in zip(rows, cols)],
            dtype=np.int64,
        )
    except PositionError as error:
        raise ValueError(f"variance-matching position mapping failed: {error}") from error


def embed_variance_matching(rgb: np.ndarray, bits, channel: int = 2) -> np.ndarray:
    """
    Embed `bits` into the highest-local-variance pixels of `rgb` using ±1
    matching. Raises ValueError on capacity failure, invalid input, or
    selection instability.

    Selection instability: because positions are derived from the cover
    image's content, the ±1 changes made during embedding can (rarely)
    perturb local variance enough to shift the top-N ordering. A
    self-verification step detects this before returning the stego image.
    If triggered, try a different cover image or a smaller payload.
    """
    _validate_rgb(rgb)
    _validate_channel(channel)
    bits_uint8 = _validate_and_convert_bits(bits)

    positions = _select_variance_positions(rgb, bits_uint8.size, channel)

    try:
        result = embed_bits_at_positions(rgb, bits_uint8, positions, _DIRECTION_KEY)
    except (LSBMatchingError, PositionError) as error:
        raise ValueError(f"variance-matching embed failed: {error}") from error

    stego = result.stego_image

    verify_positions = _select_variance_positions(stego, bits_uint8.size, channel)
    if not np.array_equal(positions, verify_positions):
        raise ValueError(
            "variance-based position selection is unstable for this "
            "cover/payload combination: recomputing variance on the stego "
            "image selected a different pixel set than the cover image "
            "did, which would make extraction fail. Try a different cover "
            "image or a smaller payload."
        )

    return stego


def extract_variance_matching(rgb: np.ndarray, bits: int, channel: int = 2) -> np.ndarray:
    """
    Extract `bits` (interpreted as num_bits) from `rgb`, mirroring
    embed_variance_matching's position selection exactly.
    """
    _validate_rgb(rgb)
    _validate_channel(channel)
    _validate_num_bits(bits)

    positions = _select_variance_positions(rgb, bits, channel)

    try:
        return extract_bits_at_positions(rgb, positions)
    except (LSBMatchingError, PositionError) as error:
        raise ValueError(f"variance-matching extract failed: {error}") from error