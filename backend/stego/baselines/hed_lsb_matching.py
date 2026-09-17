"""
HED-only LSB matching baseline.
"""

import hashlib
from functools import lru_cache

import numpy as np

from backend.core.positions import PositionError, flatten_channel_position
from backend.stego.lsb_matching import (
    LSBMatchingError,
    embed_bits_at_positions,
    extract_bits_at_positions,
)
from backend.vision.config import VisionConfigError, load_vision_config
from backend.vision.hed.inference import HEDInference, HEDInferenceError
from backend.vision.hed.model import HEDModelError, load_hed_network
from backend.vision.hed.tiling import HEDTilingError, TiledHEDInference

__all__ = ["embed_hed_matching", "extract_hed_matching"]

_DIRECTION_KEY = hashlib.sha256(
    b"GuardianPixel-baseline-hed-matching"
).digest()


@lru_cache(maxsize=1)
def _load_default_hed_inference() -> TiledHEDInference:
    """
    Build the real, config-driven HED pipeline once per process:
    load configs/vision.yaml -> load the Caffe network from disk ->
    wrap in HEDInference (mean-subtraction + forward pass) -> wrap again
    in TiledHEDInference (tiling + Hann-window blending), using the
    tile_size/overlap this project's config pins for HED specifically.
    """
    try:
        config = load_vision_config()
    except VisionConfigError as error:
        raise ValueError(f"could not load vision config for HED: {error}") from error

    try:
        network = load_hed_network(
            config.hed.prototxt_path,
            config.hed.weights_path,
            device=config.hed.resolved_device,
        )
    except HEDModelError as error:
        raise ValueError(f"could not load HED network: {error}") from error

    base_inference = HEDInference(network, config.hed.mean_bgr)

    return TiledHEDInference(
        base_inference,
        tile_size=config.hed.tile_size,
        overlap=config.hed.overlap,
    )


def _validate_rgb_is_array(rgb) -> None:
    if not isinstance(rgb, np.ndarray):
        raise ValueError(f"rgb must be a numpy ndarray, got {type(rgb).__name__}")


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


def _compute_hed_edge_map(rgb: np.ndarray, hed_inference) -> np.ndarray:
    resolved = hed_inference if hed_inference is not None else _load_default_hed_inference()
    try:
        return np.asarray(resolved.predict(rgb), dtype=np.float64)
    except (HEDInferenceError, HEDTilingError):
        raise
    except Exception as error:
        raise ValueError(f"HED inference failed: {error}") from error


def _select_hed_positions(
    rgb: np.ndarray, num_positions: int, channel: int, hed_inference
) -> np.ndarray:
    edge_map = _compute_hed_edge_map(rgb, hed_inference)
    w = edge_map.shape[1]

    flat_edges = edge_map.reshape(-1)
    if num_positions > flat_edges.size:
        raise ValueError(
            f"payload requires {num_positions} pixels but the image only "
            f"has {flat_edges.size} pixels available. Not falling back to "
            f"alternate selection — that would defeat the purpose of "
            f"testing HED-only selection specifically."
        )

    order = np.argsort(-flat_edges, kind="stable")
    top_indices = order[:num_positions]

    rows = top_indices // w
    cols = top_indices % w

    return np.array(
        [flatten_channel_position(r, c, channel, w) for r, c in zip(rows, cols)],
        dtype=np.int64,
    )


def embed_hed_matching(
    rgb: np.ndarray,
    bits,
    channel: int = 2,
    hed_inference=None,
) -> np.ndarray:
    """
    Embed `bits` into the highest-HED-edge-probability pixels of `rgb`
    using ±1 matching.

    `hed_inference`: an object exposing `.predict(rgb) -> (H, W) float
    array in [0, 1]`. Defaults to the real config-driven HED pipeline
    (cached singleton); pass a fake for tests.

    Raises ValueError on capacity failure, HED inference failure, or
    selection instability (see module docstring).
    """
    _validate_rgb_is_array(rgb)
    _validate_channel(channel)
    bits_uint8 = _validate_and_convert_bits(bits)

    positions = _select_hed_positions(rgb, bits_uint8.size, channel, hed_inference)

    try:
        result = embed_bits_at_positions(rgb, bits_uint8, positions, _DIRECTION_KEY)
    except (LSBMatchingError, PositionError) as error:
        raise ValueError(f"hed-matching embed failed: {error}") from error

    stego = result.stego_image

    verify_positions = _select_hed_positions(
        stego, bits_uint8.size, channel, hed_inference
    )
    if not np.array_equal(positions, verify_positions):
        raise ValueError(
            "HED-based position selection is unstable for this "
            "cover/payload combination: recomputing the edge map on the "
            "stego image selected a different pixel set than the cover "
            "image did, which would make extraction fail. This is an "
            "inherent limitation of content-derived selection in a "
            "baseline that (unlike the main adaptive system) doesn't "
            "transmit position metadata separately. Try a different "
            "cover image or a smaller payload."
        )

    return stego


def extract_hed_matching(
    stego: np.ndarray,
    num_bits: int,
    channel: int = 2,
    hed_inference=None,
) -> np.ndarray:
    """Extract `num_bits` bits from `stego`, mirroring embed_hed_matching."""
    _validate_rgb_is_array(stego)
    _validate_channel(channel)
    _validate_num_bits(num_bits)

    positions = _select_hed_positions(stego, num_bits, channel, hed_inference)

    try:
        return extract_bits_at_positions(stego, positions)
    except (LSBMatchingError, PositionError) as error:
        raise ValueError(f"hed-matching extract failed: {error}") from error