import hashlib

import numpy as np

from backend.core.positions import PositionError, flatten_channel_position
from backend.stego.lsb_matching import (
    LSBMatchingError,
    embed_bits_at_positions,
    extract_bits_at_positions,
)
from backend.vision.entropy_map import EntropyMapError, calculate_local_entropy
from backend.vision.preprocessing import ImagePreprocessingError, rgb_to_luminance

__all__ = ["embed_entropy_matching", "extract_entropy_matching"]

_DIRECTION_KEY = hashlib.sha256(
    b"GuardianPixel-baseline-entropy-matching"
).digest()

_WINDOW_SIZE = 11  # matches calculate_local_entropy's own default
_BINS = 32


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


def _select_entropy_positions(
    rgb: np.ndarray, num_positions: int, channel: int
) -> np.ndarray:
    h, w, _ = rgb.shape

    try:
        luminance = rgb_to_luminance(rgb)
    except ImagePreprocessingError as error:
        raise ValueError(f"luminance conversion failed: {error}") from error

    try:
        entropy_map = calculate_local_entropy(luminance, _WINDOW_SIZE, _BINS)
    except EntropyMapError:
        raise
    flat_entropy = entropy_map.reshape(-1)
    if num_positions > flat_entropy.size:
        raise ValueError(
            f"payload requires {num_positions} pixels but the image only "
            f"has {flat_entropy.size} pixels available. Not falling back "
            f"to alternate selection — that would defeat the purpose of "
            f"testing entropy-only selection specifically."
        )

    order = np.argsort(-flat_entropy, kind="stable")
    top_indices = order[:num_positions]

    rows = top_indices // w
    cols = top_indices % w

    try:
        return np.array(
            [flatten_channel_position(r, c, channel, w) for r, c in zip(rows, cols)],
            dtype=np.int64,
        )
    except PositionError as error:
        raise ValueError(f"entropy-matching position mapping failed: {error}") from error


def embed_entropy_matching(rgb: np.ndarray, bits, channel: int = 2) -> np.ndarray:
    """
    Embed `bits` into the highest-local-entropy pixels of `rgb` using ±1
    matching. Raises ValueError on capacity failure (not enough pixels),
    invalid entropy-map input, or selection instability (see below).
    """
    _validate_rgb(rgb)
    _validate_channel(channel)
    bits_uint8 = _validate_and_convert_bits(bits)

    positions = _select_entropy_positions(rgb, bits_uint8.size, channel)

    try:
        result = embed_bits_at_positions(rgb, bits_uint8, positions, _DIRECTION_KEY)
    except (LSBMatchingError, PositionError) as error:
        raise ValueError(f"entropy-matching embed failed: {error}") from error

    stego = result.stego_image

    verify_positions = _select_entropy_positions(stego, bits_uint8.size, channel)
    if not np.array_equal(positions, verify_positions):
        raise ValueError(
            "entropy-based position selection is unstable for this "
            "cover/payload combination: recomputing entropy on the stego "
            "image selected a different pixel set than the cover image "
            "did, which would make extraction fail. This is an inherent "
            "limitation of content-derived selection in a baseline that "
            "(unlike the main adaptive system) doesn't transmit position "
            "metadata separately. Try a different cover image or a "
            "smaller payload to reduce sensitivity."
        )

    return stego


def extract_entropy_matching(rgb: np.ndarray, bits: int, channel: int = 2) -> np.ndarray:
    """
    Extract `bits` (interpreted as num_bits to extract) from `rgb`,
    mirroring embed_entropy_matching's position selection.
    """
    _validate_rgb(rgb)
    _validate_channel(channel)
    _validate_num_bits(bits)

    positions = _select_entropy_positions(rgb, bits, channel)

    try:
        return extract_bits_at_positions(rgb, positions)
    except (LSBMatchingError, PositionError) as error:
        raise ValueError(f"entropy-matching extract failed: {error}") from error