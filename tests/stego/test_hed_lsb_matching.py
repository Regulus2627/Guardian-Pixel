"""Tests for the HED-only LSB matching baseline."""

import numpy as np
import pytest

from backend.stego.baselines.hed_lsb_matching import (
    embed_hed_matching,
    extract_hed_matching,
)


class _FakeHED:
    """
    Deterministic stand-in for TiledHEDInference. `edge_fn(rgb) -> (H,W)
    float array in [0,1]` lets each test control ranking directly instead
    of depending on real network output.
    """

    def __init__(self, edge_fn):
        self.edge_fn = edge_fn

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return self.edge_fn(rgb)


def _gradient_edge_map(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy * w + xx).astype(np.float64) % (h * w)) / (h * w)


def _stable_hed(rgb: np.ndarray):
    return _FakeHED(lambda img: _gradient_edge_map(img))


def _unstable_hed():
    """HED fake whose output differs between cover and any ±1-perturbed
    version of it, to exercise the self-verification failure path."""
    calls = {"n": 0}

    def edge_fn(rgb):
        calls["n"] += 1
        h, w = rgb.shape[:2]
        # Flip the ranking direction on the second call (simulates the
        # stego recompute landing on a different top-K set).
        base = _gradient_edge_map(rgb)
        return base if calls["n"] == 1 else (1.0 - base)

    return _FakeHED(edge_fn)


@pytest.fixture
def cover():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


def test_embed_rejects_non_ndarray():
    with pytest.raises(ValueError, match="numpy ndarray"):
        embed_hed_matching([[1, 2, 3]], [1, 0, 1], hed_inference=_stable_hed(None))


def test_embed_rejects_invalid_channel(cover):
    with pytest.raises(ValueError, match="channel"):
        embed_hed_matching(cover, [1, 0], channel=5, hed_inference=_stable_hed(cover))


def test_embed_rejects_empty_bits(cover):
    with pytest.raises(ValueError, match="empty"):
        embed_hed_matching(cover, [], hed_inference=_stable_hed(cover))


def test_embed_rejects_non_binary_bits(cover):
    with pytest.raises(ValueError, match="0s and 1s"):
        embed_hed_matching(cover, [0, 1, 2], hed_inference=_stable_hed(cover))


def test_embed_rejects_out_of_range_bits_before_cast(cover):
    with pytest.raises(ValueError, match="0s and 1s"):
        embed_hed_matching(cover, [-1, 0, 1], hed_inference=_stable_hed(cover))


def test_embed_rejects_multidimensional_bits(cover):
    with pytest.raises(ValueError, match="1-D"):
        embed_hed_matching(cover, [[1, 0], [0, 1]], hed_inference=_stable_hed(cover))


def test_extract_rejects_invalid_num_bits(cover):
    with pytest.raises(ValueError, match="positive integer"):
        extract_hed_matching(cover, num_bits=0, hed_inference=_stable_hed(cover))
    with pytest.raises(ValueError, match="positive integer"):
        extract_hed_matching(cover, num_bits=-3, hed_inference=_stable_hed(cover))


def test_embed_rejects_payload_exceeding_capacity(cover):
    h, w, _ = cover.shape
    too_many_bits = [1] * (h * w + 1)
    with pytest.raises(ValueError, match="only has"):
        embed_hed_matching(cover, too_many_bits, hed_inference=_stable_hed(cover))


def test_embed_extract_roundtrip(cover):
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    hed = _stable_hed(cover)

    stego = embed_hed_matching(cover, bits, channel=2, hed_inference=hed)
    recovered = extract_hed_matching(
        stego, num_bits=bits.size, channel=2, hed_inference=hed
    )

    assert np.array_equal(recovered, bits)


def test_embed_only_modifies_selected_channel(cover):
    bits = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
    hed = _stable_hed(cover)

    stego = embed_hed_matching(cover, bits, channel=0, hed_inference=hed)

    # Other two channels must be byte-for-byte untouched.
    assert np.array_equal(stego[:, :, 1], cover[:, :, 1])
    assert np.array_equal(stego[:, :, 2], cover[:, :, 2])


def test_embed_perturbation_is_at_most_one(cover):
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    hed = _stable_hed(cover)

    stego = embed_hed_matching(cover, bits, channel=1, hed_inference=hed)

    diff = np.abs(stego.astype(np.int16) - cover.astype(np.int16))
    assert diff.max() <= 1


def test_different_hed_instances_with_same_fn_still_roundtrip(cover):
    bits = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)

    stego = embed_hed_matching(cover, bits, channel=2, hed_inference=_stable_hed(cover))
    recovered = extract_hed_matching(
        stego, num_bits=bits.size, channel=2, hed_inference=_stable_hed(stego)
    )

    assert np.array_equal(recovered, bits)


def test_embed_raises_on_unstable_selection(cover):
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match="unstable"):
        embed_hed_matching(cover, bits, channel=2, hed_inference=_unstable_hed())


def test_embed_wraps_hed_inference_errors(cover):
    class _BrokenHED:
        def predict(self, rgb):
            raise RuntimeError("simulated cv2 backend failure")

    with pytest.raises(ValueError, match="HED inference failed"):
        embed_hed_matching(cover, [1, 0], hed_inference=_BrokenHED())