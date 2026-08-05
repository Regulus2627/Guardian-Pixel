"""Configuration loading and validation for the GuardianPixel vision module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


class VisionConfigError(ValueError):
    """Raised when the vision configuration is missing or invalid."""


@dataclass(frozen=True)
class ImageConfig:
    max_pixels: int
    convert_mode: str


# @dataclass(frozen=True)
# class HEDConfig:
#     device: str
#     tile_size: int
#     overlap: int
#     weights_path: str

#     @property
#     def resolved_device(self) -> str:
#         """Return the actual PyTorch device that should be used."""

#         if self.device == "auto":
#             return "cuda" if torch.cuda.is_available() else "cpu"

#         if self.device == "cuda" and not torch.cuda.is_available():
#             raise VisionConfigError(
#                 "CUDA was requested, but CUDA is not available."
#             )

#         return self.device
@dataclass(frozen=True)
class HEDConfig:
    framework: str
    device: str
    tile_size: int
    overlap: int
    prototxt_path: str
    weights_path: str
    mean_bgr: list[float]

    @property
    def resolved_device(self) -> str:
        """Return the requested HED inference device."""

        if self.device == "auto":
            return "cpu"

        return self.device


@dataclass(frozen=True)
class TextureConfig:
    entropy_window: int
    entropy_bins: int
    variance_window: int


@dataclass(frozen=True)
class FusionConfig:
    hed_weight: float
    entropy_weight: float
    variance_weight: float
    median_kernel: int


@dataclass(frozen=True)
class BlockMapConfig:
    block_size: int
    safety_margin: float
    score_method: str


@dataclass(frozen=True)
class VisualizationConfig:
    colormap: str
    overlay_alpha: float


@dataclass(frozen=True)
class VisionConfig:
    image: ImageConfig
    hed: HEDConfig
    texture: TextureConfig
    fusion: FusionConfig
    block_map: BlockMapConfig
    visualization: VisualizationConfig


def _require_section(data: dict, section: str) -> dict:
    """Return a required configuration section."""

    value = data.get(section)

    if not isinstance(value, dict):
        raise VisionConfigError(
            f"Missing or invalid configuration section: {section}"
        )

    return value


def _validate_odd_window(value: int, name: str) -> None:
    """Validate an odd local-window size."""

    if value < 3 or value % 2 == 0:
        raise VisionConfigError(
            f"{name} must be an odd integer greater than or equal to 3."
        )


def validate_vision_config(config: VisionConfig) -> None:
    """Validate all GuardianPixel vision configuration values."""

    if config.image.max_pixels <= 0:
        raise VisionConfigError("image.max_pixels must be greater than zero.")

    if config.image.convert_mode != "RGB":
        raise VisionConfigError(
            "image.convert_mode must be RGB for the current protocol."
        )

    # if config.hed.device not in {"auto", "cpu", "cuda"}:
    #     raise VisionConfigError(
    #         "hed.device must be one of: auto, cpu, cuda."
    #     )
    if config.hed.device not in {"auto", "cpu"}:
        raise VisionConfigError(
            "hed.device must be auto or cpu."
    )

    if config.hed.tile_size <= 0:
        raise VisionConfigError("hed.tile_size must be greater than zero.")

    if config.hed.overlap < 0:
        raise VisionConfigError("hed.overlap cannot be negative.")

    if config.hed.overlap >= config.hed.tile_size:
        raise VisionConfigError(
            "hed.overlap must be smaller than hed.tile_size."
        )

    if config.hed.framework != "opencv_dnn_caffe":
        raise VisionConfigError(
            "hed.framework must be opencv_dnn_caffe."
        )

    if config.hed.device not in {"auto", "cpu"}:
        raise VisionConfigError(
            "OpenCV HED currently supports auto or cpu."
        )

    if not config.hed.prototxt_path.strip():
        raise VisionConfigError(
            "hed.prototxt_path cannot be empty."
        )

    if not config.hed.weights_path.strip():
        raise VisionConfigError(
            "hed.weights_path cannot be empty."
        )

    if len(config.hed.mean_bgr) != 3:
        raise VisionConfigError(
            "hed.mean_bgr must contain three values."
        )

    if not all(
        isinstance(value, (int, float))
        for value in config.hed.mean_bgr
    ):
        raise VisionConfigError(
            "hed.mean_bgr values must be numerical."
        )

    _validate_odd_window(
        config.texture.entropy_window,
        "texture.entropy_window",
    )

    _validate_odd_window(
        config.texture.variance_window,
        "texture.variance_window",
    )

    if not 2 <= config.texture.entropy_bins <= 256:
        raise VisionConfigError(
            "texture.entropy_bins must be between 2 and 256."
        )

    fusion_weights = (
        config.fusion.hed_weight,
        config.fusion.entropy_weight,
        config.fusion.variance_weight,
    )

    if any(weight < 0 for weight in fusion_weights):
        raise VisionConfigError("Fusion weights cannot be negative.")

    if not math.isclose(
        sum(fusion_weights),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise VisionConfigError("Fusion weights must add up to 1.0.")

    _validate_odd_window(
        config.fusion.median_kernel,
        "fusion.median_kernel",
    )

    if config.block_map.block_size not in {4, 8, 16, 32}:
        raise VisionConfigError(
            "block_map.block_size must be 4, 8, 16 or 32."
        )

    if config.block_map.safety_margin < 0:
        raise VisionConfigError(
            "block_map.safety_margin cannot be negative."
        )

    if config.block_map.score_method not in {"mean", "percentile"}:
        raise VisionConfigError(
            "block_map.score_method must be mean or percentile."
        )

    if not 0 <= config.visualization.overlay_alpha <= 1:
        raise VisionConfigError(
            "visualization.overlay_alpha must be between 0 and 1."
        )


def load_vision_config(
    path: str | Path = "configs/vision.yaml",
) -> VisionConfig:
    """Load and validate the GuardianPixel vision configuration."""

    config_path = Path(path)

    if not config_path.exists():
        raise VisionConfigError(
            f"Vision configuration file was not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise VisionConfigError(
            f"Invalid YAML configuration: {error}"
        ) from error

    if not isinstance(raw_data, dict):
        raise VisionConfigError(
            "The vision configuration must contain YAML sections."
        )

    try:
        image_data = _require_section(raw_data, "image")
        hed_data = _require_section(raw_data, "hed")
        texture_data = _require_section(raw_data, "texture")
        fusion_data = _require_section(raw_data, "fusion")
        block_data = _require_section(raw_data, "block_map")
        visualization_data = _require_section(
            raw_data,
            "visualization",
        )

        config = VisionConfig(
            image=ImageConfig(**image_data),
            hed=HEDConfig(**hed_data),
            texture=TextureConfig(**texture_data),
            fusion=FusionConfig(**fusion_data),
            block_map=BlockMapConfig(**block_data),
            visualization=VisualizationConfig(
                **visualization_data
            ),
        )
    except TypeError as error:
        raise VisionConfigError(
            f"Missing or unexpected configuration value: {error}"
        ) from error

    validate_vision_config(config)

    return config