import numpy as np
import pytest

from backend.vision.hed.tiling import (
    HEDTilingError,
    TiledHEDInference,
    calculate_tile_starts,
    create_blending_window,
)


class ConstantFakeHED:
    """Fake HED predictor returning a constant probability."""

    def __init__(self, value: float = 0.5):
        self.value = value
        self.call_count = 0

    def predict(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:
        self.call_count += 1

        return np.full(
            rgb.shape[:2],
            self.value,
            dtype=np.float32,
        )


def test_small_image_uses_one_tile():
    fake_hed = ConstantFakeHED(0.5)

    tiled = TiledHEDInference(
        base_inference=fake_hed,
        tile_size=512,
        overlap=32,
    )

    rgb = np.zeros(
        (128, 192, 3),
        dtype=np.uint8,
    )

    result = tiled.predict(rgb)

    assert result.shape == (128, 192)
    assert result.dtype == np.float32
    assert tiled.last_tile_count == 1
    assert fake_hed.call_count == 1
    assert np.allclose(result, 0.5)


def test_large_image_uses_multiple_tiles():
    fake_hed = ConstantFakeHED(0.75)

    tiled = TiledHEDInference(
        base_inference=fake_hed,
        tile_size=256,
        overlap=32,
    )

    rgb = np.zeros(
        (600, 900, 3),
        dtype=np.uint8,
    )

    result = tiled.predict(rgb)

    assert result.shape == (600, 900)
    assert tiled.last_tile_count > 1
    assert fake_hed.call_count > 1
    assert np.allclose(
        result,
        0.75,
        atol=1e-5,
    )


def test_tile_starts_cover_final_boundary():
    starts = calculate_tile_starts(
        length=900,
        tile_size=256,
        overlap=32,
    )

    assert starts[0] == 0
    assert starts[-1] == 900 - 256


def test_small_dimension_has_zero_start():
    starts = calculate_tile_starts(
        length=200,
        tile_size=512,
        overlap=32,
    )

    assert starts == [0]


def test_blending_window_is_positive():
    window = create_blending_window(
        height=64,
        width=80,
    )

    assert window.shape == (64, 80)
    assert window.dtype == np.float32
    assert np.all(window > 0)
    assert np.all(np.isfinite(window))


def test_invalid_overlap_is_rejected():
    with pytest.raises(
        HEDTilingError,
        match="smaller than tile_size",
    ):
        TiledHEDInference(
            base_inference=ConstantFakeHED(),
            tile_size=256,
            overlap=256,
        )


def test_three_channel_input_is_required():
    tiled = TiledHEDInference(
        base_inference=ConstantFakeHED(),
        tile_size=256,
        overlap=32,
    )

    grayscale = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    with pytest.raises(
        HEDTilingError,
        match="H×W×3",
    ):
        tiled.predict(grayscale)


def test_uint8_input_is_required():
    tiled = TiledHEDInference(
        base_inference=ConstantFakeHED(),
        tile_size=256,
        overlap=32,
    )

    rgb = np.zeros(
        (100, 100, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        HEDTilingError,
        match="uint8",
    ):
        tiled.predict(rgb)


def test_result_is_deterministic():
    fake_hed = ConstantFakeHED(0.25)

    tiled = TiledHEDInference(
        base_inference=fake_hed,
        tile_size=128,
        overlap=16,
    )

    rgb = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    first_result = tiled.predict(rgb)
    second_result = tiled.predict(rgb)

    assert np.array_equal(
        first_result,
        second_result,
    )