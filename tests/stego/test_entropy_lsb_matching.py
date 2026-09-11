from __future__ import annotations

import inspect

import numpy as np
import pytest

from backend.stego.baselines.entropy_lsb_matching import (
    embed_entropy_matching,
    extract_entropy_matching,
    _WINDOW_SIZE,
    _BINS,
    _select_entropy_positions,
)
from backend.vision.preprocessing import ImagePreprocessingError


def make_random_rgb(h=64, w=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def make_flat_rgb(h=32, w=32, value=128) -> np.ndarray:
    """A perfectly flat image: zero local entropy everywhere."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_textured_rgb(h=64, w=64, seed=1) -> np.ndarray:
    """A high-frequency checkerboard-plus-noise image with plenty of
    high-entropy regions, good for realistic embed/extract tests."""
    rng = np.random.default_rng(seed)
    base = np.indices((h, w)).sum(axis=0) % 2 * 255
    noise = rng.integers(-20, 21, size=(h, w))
    channel = np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.stack([channel] * 3, axis=-1)


def random_bits(n, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n).astype(np.uint8)


class TestSignatureContract:
    def test_embed_signature_is_rgb_bits_channel_only(self):
        sig = inspect.signature(embed_entropy_matching)
        params = list(sig.parameters.keys())
        assert params == ["rgb", "bits", "channel"], params

    def test_extract_signature_is_rgb_bits_channel_only(self):
        sig = inspect.signature(extract_entropy_matching)
        params = list(sig.parameters.keys())
        assert params == ["rgb", "bits", "channel"], params

    def test_channel_has_default(self):
        sig = inspect.signature(embed_entropy_matching)
        assert sig.parameters["channel"].default == 2
        sig = inspect.signature(extract_entropy_matching)
        assert sig.parameters["channel"].default == 2


class TestRgbValidation:
    @pytest.mark.parametrize("bad_rgb", [
        [[1, 2, 3]],                       # not ndarray
        "not an array",
        None,
    ])
    def test_embed_rejects_non_ndarray(self, bad_rgb):
        with pytest.raises(ValueError, match="numpy ndarray"):
            embed_entropy_matching(bad_rgb, [1, 0, 1])

    def test_embed_rejects_wrong_ndim(self):
        rgb = np.zeros((10, 10), dtype=np.uint8)  # missing channel dim
        with pytest.raises(ValueError, match="shape"):
            embed_entropy_matching(rgb, [1, 0, 1])

    def test_embed_rejects_wrong_channel_count(self):
        rgb = np.zeros((10, 10, 4), dtype=np.uint8)  # RGBA
        with pytest.raises(ValueError, match="shape"):
            embed_entropy_matching(rgb, [1, 0, 1])

    def test_embed_rejects_wrong_dtype(self):
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            embed_entropy_matching(rgb, [1, 0, 1])

    def test_extract_applies_same_rgb_validation(self):
        with pytest.raises(ValueError, match="numpy ndarray"):
            extract_entropy_matching("not an array", 4)

        rgb_bad_dtype = np.zeros((10, 10, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="dtype"):
            extract_entropy_matching(rgb_bad_dtype, 4)


class TestChannelValidation:
    @pytest.mark.parametrize("bad_channel", [-1, 3, 10, "r", 1.5, None])
    def test_embed_rejects_invalid_channel(self, bad_channel):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="channel"):
            embed_entropy_matching(rgb, [1, 0, 1], channel=bad_channel)

    @pytest.mark.parametrize("bad_channel", [-1, 3, 10, "b", None])
    def test_extract_rejects_invalid_channel(self, bad_channel):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="channel"):
            extract_entropy_matching(rgb, 4, channel=bad_channel)

    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_all_valid_channels_accepted(self, channel):
        rgb = make_textured_rgb()
        bits = random_bits(8)
        stego = embed_entropy_matching(rgb, bits, channel=channel)
        out = extract_entropy_matching(stego, 8, channel=channel)
        assert np.array_equal(bits, out)


class TestBitsValidation:
    def test_embed_rejects_empty_bits(self):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="not be empty"):
            embed_entropy_matching(rgb, [])

    def test_embed_rejects_non_binary_bits(self):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="0s and 1s"):
            embed_entropy_matching(rgb, [0, 1, 2])

    def test_embed_rejects_2d_bits(self):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="1-D"):
            embed_entropy_matching(rgb, [[0, 1], [1, 0]])

    def test_embed_accepts_list_tuple_and_ndarray(self):
        rgb = make_textured_rgb()
        for bits in ([1, 0, 1, 1], (1, 0, 1, 1), np.array([1, 0, 1, 1])):
            stego = embed_entropy_matching(rgb, bits)
            out = extract_entropy_matching(stego, 4)
            assert np.array_equal(np.array([1, 0, 1, 1], dtype=np.uint8), out)

    @pytest.mark.parametrize("bad_num_bits", [0, -1, 1.5, "4", None])
    def test_extract_rejects_invalid_num_bits(self, bad_num_bits):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="positive integer"):
            extract_entropy_matching(rgb, bad_num_bits)


class TestCapacity:
    def test_embed_rejects_payload_larger_than_image(self):
        rgb = make_textured_rgb(h=4, w=4)  # 16 pixels
        bits = random_bits(17)
        with pytest.raises(ValueError, match="only.*has"):
            embed_entropy_matching(rgb, bits)

    def test_embed_accepts_payload_equal_to_pixel_count(self):
        rgb = make_textured_rgb(h=8, w=8)  # 64 pixels
        bits = random_bits(64)
        stego = embed_entropy_matching(rgb, bits)
        out = extract_entropy_matching(stego, 64)
        assert np.array_equal(bits, out)


class TestRoundtrip:
    @pytest.mark.parametrize("n_bits", [1, 2, 8, 33, 100])
    def test_roundtrip_various_lengths(self, n_bits):
        rgb = make_textured_rgb(h=48, w=48, seed=n_bits)
        bits = random_bits(n_bits, seed=n_bits)
        stego = embed_entropy_matching(rgb, bits)
        recovered = extract_entropy_matching(stego, n_bits)
        assert np.array_equal(bits, recovered)

    def test_roundtrip_all_zero_bits(self):
        rgb = make_textured_rgb()
        bits = np.zeros(16, dtype=np.uint8)
        stego = embed_entropy_matching(rgb, bits)
        assert np.array_equal(bits, extract_entropy_matching(stego, 16))

    def test_roundtrip_all_one_bits(self):
        rgb = make_textured_rgb()
        bits = np.ones(16, dtype=np.uint8)
        stego = embed_entropy_matching(rgb, bits)
        assert np.array_equal(bits, extract_entropy_matching(stego, 16))

    def test_embed_does_not_mutate_input(self):
        rgb = make_textured_rgb()
        original = rgb.copy()
        embed_entropy_matching(rgb, random_bits(16))
        assert np.array_equal(rgb, original), "embed must not mutate the input array"

    def test_embed_returns_new_array_not_view(self):
        rgb = make_textured_rgb()
        stego = embed_entropy_matching(rgb, random_bits(16))
        assert stego is not rgb
        assert stego.base is not rgb  # not a view either

    def test_stego_output_shape_and_dtype_preserved(self):
        rgb = make_textured_rgb(h=40, w=50)
        stego = embed_entropy_matching(rgb, random_bits(20))
        assert stego.shape == rgb.shape
        assert stego.dtype == rgb.dtype

    def test_determinism_same_inputs_same_output(self):
        rgb = make_textured_rgb()
        bits = random_bits(20)
        stego1 = embed_entropy_matching(rgb, bits)
        stego2 = embed_entropy_matching(rgb, bits)
        assert np.array_equal(stego1, stego2)


class TestChannelIsolation:
    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_embedding_only_touches_target_channel(self, channel):
        rgb = make_textured_rgb()
        stego = embed_entropy_matching(rgb, random_bits(24), channel=channel)
        for c in (0, 1, 2):
            if c == channel:
                continue
            assert np.array_equal(rgb[:, :, c], stego[:, :, c]), (
                f"channel {c} was modified while embedding targeted channel {channel}"
            )

    def test_perturbations_are_at_most_one_lsb_step(self):
        rgb = make_textured_rgb()
        stego = embed_entropy_matching(rgb, random_bits(30), channel=1)
        diff = np.abs(stego[:, :, 1].astype(np.int16) - rgb[:, :, 1].astype(np.int16))
        assert diff.max() <= 1


class TestExtractRobustness:
    def test_extract_wrong_channel_does_not_crash_but_may_mismatch(self):
        rgb = make_textured_rgb()
        bits = random_bits(16)
        stego = embed_entropy_matching(rgb, bits, channel=0)
        out = extract_entropy_matching(stego, 16, channel=1)
        assert out.shape == bits.shape
        assert out.dtype == np.uint8

    def test_extract_on_never_embedded_image_returns_lsb_values(self):
        rgb = make_textured_rgb()
        out = extract_entropy_matching(rgb, 10)
        assert out.shape == (10,)
        assert set(np.unique(out)).issubset({0, 1})


class TestSelectionInstability:
    def test_flat_image_either_raises_instability_or_roundtrips(self):
        rgb = make_flat_rgb(h=16, w=16)
        bits = random_bits(20)
        try:
            stego = embed_entropy_matching(rgb, bits)
        except ValueError as e:
            assert "unstable" in str(e) or "entropy-matching embed failed" in str(e)
        else:
            recovered = extract_entropy_matching(stego, 20)
            assert np.array_equal(bits, recovered)

    def test_instability_error_is_valueerror_not_other_exception_type(self):
        rgb = make_flat_rgb(h=12, w=12)
        bits = random_bits(30)
        try:
            embed_entropy_matching(rgb, bits)
        except Exception as e:
            assert isinstance(e, ValueError)


class TestSelectEntropyPositionsHelper:
    def test_returns_unique_positions(self):
        rgb = make_textured_rgb()
        positions = _select_entropy_positions(rgb, 50, channel=2)
        assert len(positions) == 50
        assert len(np.unique(positions)) == 50

    def test_positions_within_bounds(self):
        rgb = make_textured_rgb(h=20, w=30)
        positions = _select_entropy_positions(rgb, 40, channel=0)
        assert positions.min() >= 0
        assert positions.max() < rgb.size

    def test_picks_highest_entropy_region_over_flat_region(self):
        """Build an image that's flat on the left half and noisy on the
        right half; positions for a small payload should land in the
        noisy half."""
        h, w = 32, 32
        rng = np.random.default_rng(42)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, : w // 2, :] = 100  # flat, zero-entropy region
        noisy = rng.integers(0, 256, size=(h, w // 2), dtype=np.uint8)
        for c in range(3):
            img[:, w // 2 :, c] = noisy

        positions = _select_entropy_positions(img, 20, channel=0)
        # Recover (row, col) from flattened channel-position indices.
        pixel_indices = positions // 3
        cols = pixel_indices % w
        assert np.all(cols >= w // 2), "expected selection to favor the noisy half"

    def test_window_and_bins_constants_are_sane(self):
        assert _WINDOW_SIZE >= 3 and _WINDOW_SIZE % 2 == 1
        assert 2 <= _BINS <= 256


class TestErrorWrapping:
    def test_embed_failure_is_wrapped_as_valueerror(self, monkeypatch):
        import backend.stego.baselines.entropy_lsb_matching as mod
        from backend.stego.lsb_matching import LSBMatchingError

        def boom(*args, **kwargs):
            raise LSBMatchingError("synthetic failure")

        monkeypatch.setattr(mod, "embed_bits_at_positions", boom)
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="entropy-matching embed failed"):
            embed_entropy_matching(rgb, random_bits(8))

    def test_extract_failure_is_wrapped_as_valueerror(self, monkeypatch):
        import backend.stego.baselines.entropy_lsb_matching as mod
        from backend.stego.lsb_matching import LSBMatchingError

        def boom(*args, **kwargs):
            raise LSBMatchingError("synthetic failure")

        monkeypatch.setattr(mod, "extract_bits_at_positions", boom)
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="entropy-matching extract failed"):
            extract_entropy_matching(rgb, 8)

    def test_luminance_conversion_failure_is_wrapped_as_valueerror(self, monkeypatch):
        """
        rgb_to_luminance is called on an rgb array that already passed
        _validate_rgb, so this path is defensive/unreachable in practice --
        but if it ever does fail, it must surface as ValueError, not an
        ImagePreprocessingError leaking out of this module's boundary.
        """
        import backend.stego.baselines.entropy_lsb_matching as mod

        def boom(*args, **kwargs):
            raise ImagePreprocessingError("synthetic conversion failure")

        monkeypatch.setattr(mod, "rgb_to_luminance", boom)
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="luminance conversion failed"):
            embed_entropy_matching(rgb, random_bits(8))


class TestLuminanceIntegration:
    def test_select_entropy_positions_calls_shared_rgb_to_luminance(self, monkeypatch):
        """
        Regression guard against reintroducing a local/inline luminance
        formula. Uses a call-counting spy that delegates to the real
        implementation (so entropy/positions stay correct) rather than a
        behavioral before/after diff -- local Shannon entropy is invariant
        under bijective intensity remaps (e.g. inversion), so a diff-based
        check can false-negative even when the function IS being called.
        """
        import backend.stego.baselines.entropy_lsb_matching as mod
        from backend.vision.preprocessing import rgb_to_luminance as real_fn

        call_count = 0

        def spy(rgb_array):
            nonlocal call_count
            call_count += 1
            return real_fn(rgb_array)

        monkeypatch.setattr(mod, "rgb_to_luminance", spy)

        rgb = make_textured_rgb()
        positions = _select_entropy_positions(rgb, 30, channel=0)

        assert call_count == 1
        assert len(positions) == 30

    def test_grayscale_rgb_still_produces_valid_positions(self):
        """Sanity check against the real rgb_to_luminance with an
        already-grayscale (R==G==B) image, a common edge case for
        luminance formulas."""
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        positions = _select_entropy_positions(rgb, 20, channel=2)
        assert len(positions) == 20
        assert positions.min() >= 0
        assert positions.max() < rgb.size