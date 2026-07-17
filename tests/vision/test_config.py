from pathlib import Path

import pytest
import yaml

from backend.vision.config import (
    VisionConfigError,
    load_vision_config,
)


CONFIG_PATH = Path("configs/vision.yaml")


def test_load_valid_configuration():
    config = load_vision_config(CONFIG_PATH)

    assert config.image.convert_mode == "RGB"
    assert config.image.max_pixels == 16_000_000

    assert config.hed.tile_size == 512
    assert config.hed.overlap == 32
    assert config.hed.resolved_device in {"cpu", "cuda"}

    assert config.texture.entropy_window == 11
    assert config.texture.variance_window == 11

    assert config.fusion.hed_weight == pytest.approx(0.45)
    assert config.fusion.entropy_weight == pytest.approx(0.35)
    assert config.fusion.variance_weight == pytest.approx(0.20)

    assert config.block_map.block_size == 8
    assert config.block_map.score_method == "mean"


def test_missing_configuration_file():
    with pytest.raises(
        VisionConfigError,
        match="was not found",
    ):
        load_vision_config("configs/does_not_exist.yaml")


def test_invalid_fusion_weight_total(tmp_path):
    raw_config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    raw_config["fusion"]["hed_weight"] = 0.80

    invalid_path = tmp_path / "invalid_weights.yaml"
    invalid_path.write_text(
        yaml.safe_dump(raw_config),
        encoding="utf-8",
    )

    with pytest.raises(
        VisionConfigError,
        match="must add up to 1.0",
    ):
        load_vision_config(invalid_path)


def test_even_entropy_window_is_rejected(tmp_path):
    raw_config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    raw_config["texture"]["entropy_window"] = 10

    invalid_path = tmp_path / "invalid_window.yaml"
    invalid_path.write_text(
        yaml.safe_dump(raw_config),
        encoding="utf-8",
    )

    with pytest.raises(
        VisionConfigError,
        match="odd integer",
    ):
        load_vision_config(invalid_path)


def test_invalid_block_size_is_rejected(tmp_path):
    raw_config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    raw_config["block_map"]["block_size"] = 7

    invalid_path = tmp_path / "invalid_block.yaml"
    invalid_path.write_text(
        yaml.safe_dump(raw_config),
        encoding="utf-8",
    )

    with pytest.raises(
        VisionConfigError,
        match="must be 4, 8, 16 or 32",
    ):
        load_vision_config(invalid_path)