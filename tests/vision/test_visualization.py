import numpy as np
import pytest
from PIL import Image

from backend.vision.visualization import (
    VisualizationError,
    feature_map_to_uint8,
    save_binary_block_map,
    save_color_heatmap,
    save_grayscale_map,
    save_selected_blocks_overlay,
)


def test_feature_map_to_uint8():
    feature_map = np.array(
        [
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    result = feature_map_to_uint8(
        feature_map
    )

    assert result.dtype == np.uint8
    assert result[0, 0] == 0
    assert result[0, 2] == 255


def test_save_grayscale_and_color_maps(tmp_path):
    feature_map = np.linspace(
        0,
        1,
        100,
        dtype=np.float32,
    ).reshape(10, 10)

    grayscale_path = (
        tmp_path / "grayscale.png"
    )

    color_path = (
        tmp_path / "color.png"
    )

    save_grayscale_map(
        feature_map,
        grayscale_path,
    )

    save_color_heatmap(
        feature_map,
        color_path,
    )

    assert grayscale_path.exists()
    assert color_path.exists()

    grayscale = Image.open(
        grayscale_path
    )

    color = Image.open(color_path)

    assert grayscale.size == (10, 10)
    assert color.size == (10, 10)


def test_save_binary_block_map(tmp_path):
    block_map = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    output_path = (
        tmp_path / "block_map.png"
    )

    save_binary_block_map(
        block_map,
        output_path,
        display_scale=8,
    )

    image = Image.open(output_path)

    assert image.size == (16, 16)


def test_selected_overlay_preserves_image_size(
    tmp_path,
):
    rgb = np.full(
        (17, 19, 3),
        100,
        dtype=np.uint8,
    )

    selected_blocks = np.zeros(
        (3, 3),
        dtype=bool,
    )

    selected_blocks[0, 0] = True

    output_path = (
        tmp_path / "overlay.png"
    )

    save_selected_blocks_overlay(
        rgb=rgb,
        selected_blocks=(
            selected_blocks
        ),
        block_size=8,
        output_path=output_path,
        alpha=0.30,
    )

    image = Image.open(output_path)

    assert image.size == (19, 17)


def test_mismatched_overlay_map_is_rejected(
    tmp_path,
):
    rgb = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    invalid_map = np.zeros(
        (2, 2),
        dtype=bool,
    )

    with pytest.raises(
        VisualizationError,
        match="do not match",
    ):
        save_selected_blocks_overlay(
            rgb=rgb,
            selected_blocks=invalid_map,
            block_size=8,
            output_path=(
                tmp_path / "invalid.png"
            ),
        )