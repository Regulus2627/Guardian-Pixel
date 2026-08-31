import io

import numpy as np
import pytest
from PIL import Image

from backend.vision.block_map import (
    BlockMapError,
)
from backend.vision.config import (
    load_vision_config,
)
from backend.vision.map_encoding import (
    decode_block_map,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)


class FakeHEDPredictor:
    """Fast deterministic HED replacement for service tests."""

    def __init__(self):
        self.last_tile_count = 1
        self.last_inference_seconds = 0.0

    def predict(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:
        luminance = np.mean(
            rgb.astype(np.float32),
            axis=2,
        )

        minimum = float(
            luminance.min()
        )

        maximum = float(
            luminance.max()
        )

        value_range = maximum - minimum

        if value_range <= 0:
            return np.zeros(
                rgb.shape[:2],
                dtype=np.float32,
            )

        return (
            (
                luminance - minimum
            )
            / value_range
        ).astype(np.float32)


def create_test_image_bytes(
    width: int = 128,
    height: int = 128,
) -> bytes:
    generator = np.random.default_rng(
        42
    )

    rgb = generator.integers(
        0,
        256,
        size=(height, width, 3),
        dtype=np.uint8,
    )

    image = Image.fromarray(rgb)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def create_service():
    return GuardianPixelVisionService(
        config=load_vision_config(),
        hed_predictor=FakeHEDPredictor(),
    )


def test_complete_vision_service():
    service = create_service()

    result = service.analyze_cover(
        source=create_test_image_bytes(),
        required_payload_bits=2000,
        channels_per_selected_pixel=1,
        reserved_position_count=0,
    )

    expected_shape = (128, 128)

    assert result.image_info.width == 128
    assert result.image_info.height == 128

    assert (
        result.feature_maps.hed_map.shape
        == expected_shape
    )

    assert (
        result.feature_maps.entropy_map.shape
        == expected_shape
    )

    assert (
        result.feature_maps.variance_map.shape
        == expected_shape
    )

    assert (
        result.feature_maps
        .fused_heatmap.shape
        == expected_shape
    )

    assert (
        result.raw_feature_maps
        is not None
    )

    assert (
        result.raw_feature_maps
        .hed_map.shape
        == expected_shape
    )

    assert (
        result.raw_feature_maps
        .entropy_map.shape
        == expected_shape
    )

    assert (
        result.raw_feature_maps
        .variance_map.shape
        == expected_shape
    )

    assert (
        result.block_map
        .usable_position_count
        >= 2200
    )

    assert result.timings["total"] >= 0


def test_encoded_map_decodes_exactly():
    service = create_service()

    result = service.analyze_cover(
        source=create_test_image_bytes(),
        required_payload_bits=1000,
    )

    decoded = decode_block_map(
        encoding_type=(
            result.map_encoding
            .encoding_type
        ),
        encoded_data=(
            result.map_encoding
            .encoded_data
        ),
        block_rows=(
            result.block_map.block_rows
        ),
        block_columns=(
            result.block_map
            .block_columns
        ),
    )

    assert np.array_equal(
        decoded,
        result.block_map.selected_blocks,
    )


def test_service_is_deterministic():
    service = create_service()

    image_data = (
        create_test_image_bytes()
    )

    first = service.analyze_cover(
        source=image_data,
        required_payload_bits=1500,
    )

    second = service.analyze_cover(
        source=image_data,
        required_payload_bits=1500,
    )

    assert np.array_equal(
        first.block_map.selected_blocks,
        second.block_map.selected_blocks,
    )

    assert (
        first.map_encoding.encoding_type
        == second.map_encoding.encoding_type
    )

    assert (
        first.map_encoding.encoded_data
        == second.map_encoding.encoded_data
    )


def test_insufficient_capacity_is_rejected():
    service = create_service()

    with pytest.raises(
        BlockMapError,
        match="does not provide enough capacity",
    ):
        service.analyze_cover(
            source=create_test_image_bytes(
                width=32,
                height=32,
            ),
            required_payload_bits=100_000,
        )