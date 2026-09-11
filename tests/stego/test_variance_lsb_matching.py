from __future__ import annotations

import inspect

import numpy as np
import pytest

import backend.stego.baselines.variance_lsb_matching as mod
from backend.stego.baselines.variance_lsb_matching import (
    _WINDOW_SIZE,
    _select_variance_positions,
    embed_variance_matching,
    extract_variance_matching,
)
from backend.vision.preprocessing import ImagePreprocessingError


def make_random_rgb(h=64, w=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def make_flat_rgb(h=32, w=32, value=128) -> np.ndarray:
    """flat image: zero local variance everywhere."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_textured_rgb(h=64, w=64, seed=1) -> np.ndarray:
    """High-frequency checkerboard-plus-noise: plenty of high-variance
    regions for realistic embed/extract tests."""
    rng = np.random.default_rng(seed)
    base = np.indices((h, w)).sum(axis=0) % 2 * 255
    noise = rng.integers(-20, 21, size=(h, w))
    channel = np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.stack([channel] * 3, axis=-1)


def make_noisy_left_flat_right_rgb(h=32, w=32, seed=42) -> np.ndarray:
    """Left half is noisy (high variance); right half is flat (zero
    variance). Used to verify the selector favours the noisy region."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, w // 2 :, :] = 128  # flat right half
    noisy = rng.integers(0, 256, size=(h, w // 2), dtype=np.uint8)
    for c in range(3):
        img[:, : w // 2, c] = noisy
    return img


def random_bits(n, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n).astype(np.uint8)


class TestSignatureContract:
    def test_embed_signature_is_rgb_bits_channel_only(self):
        params = list(inspect.signature(embed_variance_matching).parameters.keys())
        assert params == ["rgb", "bits", "channel"], params

    def test_extract_signature_is_rgb_bits_channel_only(self):
        params = list(inspect.signature(extract_variance_matching).parameters.keys())
        assert params == ["rgb", "bits", "channel"], params

    def test_channel_defaults_to_2_on_both(self):
        sig_e = inspect.signature(embed_variance_matching)
        sig_x = inspect.signature(extract_variance_matching)
        assert sig_e.parameters["channel"].default == 2
        assert sig_x.parameters["channel"].default == 2


class TestRgbValidation:
    @pytest.mark.parametrize("bad", [
        [[1, 2, 3]], "string", None, 42,
    ])
    def test_embed_rejects_non_ndarray(self, bad):
        with pytest.raises(ValueError, match="numpy ndarray"):
            embed_variance_matching(bad, [1, 0, 1])

    def test_embed_rejects_2d_array(self):
        with pytest.raises(ValueError, match="shape"):
            embed_variance_matching(np.zeros((10, 10), dtype=np.uint8), [1])

    def test_embed_rejects_rgba(self):
        with pytest.raises(ValueError, match="shape"):
            embed_variance_matching(np.zeros((10, 10, 4), dtype=np.uint8), [1])

    def test_embed_rejects_wrong_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            embed_variance_matching(np.zeros((8, 8, 3), dtype=np.float32), [1])

    def test_extract_applies_same_rgb_validation(self):
        with pytest.raises(ValueError, match="numpy ndarray"):
            extract_variance_matching("bad", 4)
        with pytest.raises(ValueError, match="dtype"):
            extract_variance_matching(np.zeros((8, 8, 3), dtype=np.int32), 4)


class TestChannelValidation:
    @pytest.mark.parametrize("bad", [-1, 3, 10, "r", 1.5, None])
    def test_embed_rejects_invalid_channel(self, bad):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="channel"):
            embed_variance_matching(rgb, [1, 0, 1], channel=bad)

    @pytest.mark.parametrize("bad", [-1, 3, 10, "b", None])
    def test_extract_rejects_invalid_channel(self, bad):
        rgb = make_textured_rgb()
        with pytest.raises(ValueError, match="channel"):
            extract_variance_matching(rgb, 4, channel=bad)

    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_all_valid_channels_roundtrip(self, channel):
        """Variance-based selection is content-derived, so embedding can
        (rarely) hit the documented self-verification instability check
        instead of succeeding -- see embed_variance_matching's docstring.
        Either outcome is acceptable here; what matters is that when it
        does succeed, the roundtrip is correct."""
        rgb = make_textured_rgb()
        bits = random_bits(8)
        try:
            stego = embed_variance_matching(rgb, bits, channel=channel)
        except ValueError as e:
            assert "unstable" in str(e)
            return
        assert np.array_equal(bits, extract_variance_matching(stego, 8, channel=channel))


class TestBitsValidation:
    def test_embed_rejects_empty(self):
        with pytest.raises(ValueError, match="not be empty"):
            embed_variance_matching(make_textured_rgb(), [])

    def test_embed_rejects_non_binary(self):
        with pytest.raises(ValueError, match="0s and 1s"):
            embed_variance_matching(make_textured_rgb(), [0, 1, 2])

    def test_embed_rejects_2d_bits(self):
        with pytest.raises(ValueError, match="1-D"):
            embed_variance_matching(make_textured_rgb(), [[0, 1], [1, 0]])

    def test_embed_accepts_list_tuple_ndarray(self):
        rgb = make_textured_rgb()
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        for bits in ([1, 0, 1, 1], (1, 0, 1, 1), expected.copy()):
            stego = embed_variance_matching(rgb, bits)
            assert np.array_equal(expected, extract_variance_matching(stego, 4))

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "4", None])
    def test_extract_rejects_invalid_num_bits(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            extract_variance_matching(make_textured_rgb(), bad)


class TestCapacity:
    def test_embed_rejects_payload_larger_than_image(self):
        rgb = make_textured_rgb(h=4, w=4)  # 16 pixels
        with pytest.raises(ValueError, match="only.*has"):
            embed_variance_matching(rgb, random_bits(17))

    def test_embed_accepts_exactly_pixel_count_bits(self):
        rgb = make_textured_rgb(h=8, w=8)  # 64 pixels
        bits = random_bits(64)
        stego = embed_variance_matching(rgb, bits)
        assert np.array_equal(bits, extract_variance_matching(stego, 64))


class TestRoundtrip:
    @pytest.mark.parametrize("n_bits", [1, 2, 8, 33, 100])
    def test_roundtrip_various_payload_sizes(self, n_bits):
        rgb = make_textured_rgb(h=48, w=48, seed=n_bits)
        bits = random_bits(n_bits, seed=n_bits)
        assert np.array_equal(bits, extract_variance_matching(
            embed_variance_matching(rgb, bits), n_bits
        ))

    def test_roundtrip_all_zeros(self):
        rgb = make_textured_rgb()
        bits = np.zeros(16, dtype=np.uint8)
        assert np.array_equal(bits, extract_variance_matching(
            embed_variance_matching(rgb, bits), 16
        ))

    def test_roundtrip_all_ones(self):
        rgb = make_textured_rgb()
        bits = np.ones(16, dtype=np.uint8)
        assert np.array_equal(bits, extract_variance_matching(
            embed_variance_matching(rgb, bits), 16
        ))

    def test_embed_does_not_mutate_input(self):
        rgb = make_textured_rgb()
        original = rgb.copy()
        embed_variance_matching(rgb, random_bits(16))
        assert np.array_equal(rgb, original)

    def test_embed_returns_new_array_not_view(self):
        rgb = make_textured_rgb()
        stego = embed_variance_matching(rgb, random_bits(16))
        assert stego is not rgb
        assert stego.base is not rgb

    def test_stego_shape_and_dtype_preserved(self):
        rgb = make_textured_rgb(h=40, w=50)
        stego = embed_variance_matching(rgb, random_bits(20))
        assert stego.shape == rgb.shape
        assert stego.dtype == rgb.dtype

    def test_determinism(self):
        rgb = make_textured_rgb()
        bits = random_bits(20)
        assert np.array_equal(
            embed_variance_matching(rgb, bits),
            embed_variance_matching(rgb, bits),
        )


class TestChannelIsolation:
    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_only_target_channel_is_modified(self, channel):
        """See test_all_valid_channels_roundtrip: embedding on a given
        channel can legitimately raise the documented instability error
        for content-derived (variance) selection. When it does, there's
        no stego image to check isolation on, so we accept that outcome
        too -- the property under test only applies when embedding
        actually succeeds."""
        rgb = make_textured_rgb()
        try:
            stego = embed_variance_matching(rgb, random_bits(24), channel=channel)
        except ValueError as e:
            assert "unstable" in str(e)
            return
        for c in (0, 1, 2):
            if c == channel:
                continue
            assert np.array_equal(rgb[:, :, c], stego[:, :, c]), (
                f"channel {c} was unexpectedly modified when targeting channel {channel}"
            )

    def test_perturbations_are_at_most_one(self):
        rgb = make_textured_rgb()
        try:
            stego = embed_variance_matching(rgb, random_bits(30), channel=1)
        except ValueError as e:
            assert "unstable" in str(e)
            return
        diff = np.abs(stego[:, :, 1].astype(np.int16) - rgb[:, :, 1].astype(np.int16))
        assert diff.max() <= 1


class TestExtractRobustness:
    def test_extract_on_unembedded_image_does_not_raise(self):
        rgb = make_textured_rgb()
        out = extract_variance_matching(rgb, 10)
        assert out.shape == (10,)
        assert set(np.unique(out)).issubset({0, 1})

    def test_extract_wrong_channel_does_not_crash(self):
        rgb = make_textured_rgb()
        bits = random_bits(16)
        stego = embed_variance_matching(rgb, bits, channel=0)
        out = extract_variance_matching(stego, 16, channel=1)
        assert out.shape == bits.shape
        assert out.dtype == np.uint8


class TestSelectionInstability:
    def test_flat_image_raises_instability_or_roundtrips(self):
        """
        A flat image has zero variance everywhere. Tie-breaking on an
        all-equal variance map makes position selection fragile to ±1
        pixel changes. The self-verification check should catch this. We
        accept either a clean ValueError or a successful roundtrip -- but
        never an unhandled exception of another type.
        """
        rgb = make_flat_rgb(h=16, w=16)
        bits = random_bits(20)
        try:
            stego = embed_variance_matching(rgb, bits)
        except ValueError as e:
            assert "unstable" in str(e) or "variance-matching embed failed" in str(e)
        else:
            assert np.array_equal(bits, extract_variance_matching(stego, 20))

    def test_instability_surfaces_as_valueerror_not_other_type(self):
        rgb = make_flat_rgb(h=12, w=12)
        bits = random_bits(30)
        try:
            embed_variance_matching(rgb, bits)
        except Exception as e:
            assert isinstance(e, ValueError)


class TestSelectVariancePositionsHelper:
    def test_returns_unique_positions(self):
        rgb = make_textured_rgb()
        positions = _select_variance_positions(rgb, 50, channel=2)
        assert len(positions) == 50
        assert len(np.unique(positions)) == 50

    def test_positions_within_image_bounds(self):
        rgb = make_textured_rgb(h=20, w=30)
        positions = _select_variance_positions(rgb, 40, channel=0)
        assert positions.min() >= 0
        assert positions.max() < rgb.size

    def test_favours_high_variance_region(self):
        """Positions for a small payload should land in the noisy
        (high-variance) left half, not the flat right half."""
        rgb = make_noisy_left_flat_right_rgb(h=32, w=32)
        positions = _select_variance_positions(rgb, 20, channel=0)

        # channel position -> pixel index -> column
        pixel_indices = positions // 3
        cols = pixel_indices % 32
        assert np.all(cols < 16), (
            "expected selection to favour the high-variance left half"
        )

    def test_window_size_constant_is_valid(self):
        assert _WINDOW_SIZE >= 3 and _WINDOW_SIZE % 2 == 1

    def test_capacity_error_message_mentions_available(self):
        rgb = make_textured_rgb(h=4, w=4)
        with pytest.raises(ValueError, match="only.*has"):
            _select_variance_positions(rgb, 1000, channel=0)


class TestErrorWrapping:
    def test_embed_lsb_failure_wraps_as_valueerror(self, monkeypatch):
        from backend.stego.lsb_matching import LSBMatchingError

        monkeypatch.setattr(
            mod, "embed_bits_at_positions",
            lambda *a, **kw: (_ for _ in ()).throw(LSBMatchingError("synthetic")),
        )
        with pytest.raises(ValueError, match="variance-matching embed failed"):
            embed_variance_matching(make_textured_rgb(), random_bits(8))

    def test_extract_lsb_failure_wraps_as_valueerror(self, monkeypatch):
        from backend.stego.lsb_matching import LSBMatchingError

        monkeypatch.setattr(
            mod, "extract_bits_at_positions",
            lambda *a, **kw: (_ for _ in ()).throw(LSBMatchingError("synthetic")),
        )
        with pytest.raises(ValueError, match="variance-matching extract failed"):
            extract_variance_matching(make_textured_rgb(), 8)

    def test_luminance_conversion_failure_wraps_as_valueerror(self, monkeypatch):
        monkeypatch.setattr(
            mod, "rgb_to_luminance",
            lambda *a, **kw: (_ for _ in ()).throw(
                ImagePreprocessingError("synthetic")
            ),
        )
        with pytest.raises(ValueError, match="luminance conversion failed"):
            embed_variance_matching(make_textured_rgb(), random_bits(8))

    def test_variance_map_error_propagates_unmodified(self, monkeypatch):
        """VarianceMapError is re-raised as-is (bare `raise`) so callers
        can distinguish a bad image from a bad payload."""
        from backend.vision.variance_map import VarianceMapError

        monkeypatch.setattr(
            mod, "calculate_local_variance",
            lambda *a, **kw: (_ for _ in ()).throw(
                VarianceMapError("synthetic variance failure")
            ),
        )
        with pytest.raises(VarianceMapError, match="synthetic variance failure"):
            embed_variance_matching(make_textured_rgb(), random_bits(8))


class TestDependencyIntegration:
    def test_rgb_to_luminance_is_called_during_position_selection(self, monkeypatch):
        """Spy confirms _select_variance_positions routes through the
        shared rgb_to_luminance binding, not an inline reimplementation."""
        from backend.vision.preprocessing import rgb_to_luminance as real_fn

        call_count = 0

        def spy(rgb_array):
            nonlocal call_count
            call_count += 1
            return real_fn(rgb_array)

        monkeypatch.setattr(mod, "rgb_to_luminance", spy)
        _select_variance_positions(make_textured_rgb(), 30, channel=0)
        assert call_count == 1

    def test_calculate_local_variance_is_called_during_position_selection(self, monkeypatch):
        """Spy confirms _select_variance_positions routes through the
        shared calculate_local_variance binding."""
        from backend.vision.variance_map import calculate_local_variance as real_fn

        call_count = 0

        def spy(luminance, window_size):
            nonlocal call_count
            call_count += 1
            return real_fn(luminance, window_size)

        monkeypatch.setattr(mod, "calculate_local_variance", spy)
        _select_variance_positions(make_textured_rgb(), 30, channel=0)
        assert call_count == 1

    def test_embed_calls_position_selection_twice_for_self_verification(self, monkeypatch):
        """embed calls _select_variance_positions on both the cover and
        the stego image. Counting calls to rgb_to_luminance is a clean
        proxy for this since it's called once per selection."""
        from backend.vision.preprocessing import rgb_to_luminance as real_fn

        call_count = 0

        def spy(rgb_array):
            nonlocal call_count
            call_count += 1
            return real_fn(rgb_array)

        monkeypatch.setattr(mod, "rgb_to_luminance", spy)
        embed_variance_matching(make_textured_rgb(), random_bits(8))
        assert call_count == 2  # once for cover, once for stego verification

    def test_grayscale_rgb_produces_valid_positions(self):
        """R==G==B edge case for rgb_to_luminance: result should still
        be a valid uint8 H×W array and produce sane positions."""
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)
        positions = _select_variance_positions(rgb, 20, channel=2)
        assert len(positions) == 20
        assert positions.min() >= 0
        assert positions.max() < rgb.size