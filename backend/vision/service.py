"""Complete GuardianPixel sender-side vision analysis service."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from backend.vision.block_map import (
    generate_capacity_aware_block_map,
)
from backend.vision.config import (
    VisionConfig,
    load_vision_config,
)
from backend.vision.entropy_map import (
    calculate_local_entropy,
)
from backend.vision.fusion import (
    fuse_feature_maps,
)
from backend.vision.hed.inference import (
    HEDInference,
)
from backend.vision.hed.model import (
    load_hed_network,
)
from backend.vision.hed.tiling import (
    TiledHEDInference,
)
from backend.vision.map_encoding import (
    encode_block_map,
)
from backend.vision.preprocessing import (
    ImageSource,
    load_and_validate_image,
    rgb_to_luminance,
)
from backend.vision.schemas import (
    VisionAnalysisResult,
)
from backend.vision.variance_map import (
    calculate_local_variance,
)


class VisionServiceError(RuntimeError):
    """Raised when the complete vision service fails."""


class GuardianPixelVisionService:
    """
    Reusable sender-side vision-analysis service.

    The HED model is loaded once when this service is created and can
    then be reused for multiple cover images.
    """

    def __init__(
        self,
        config: VisionConfig,
        hed_predictor=None,
    ):
        self.config = config

        if hed_predictor is None:
            network = load_hed_network(
                prototxt_path=(
                    config.hed.prototxt_path
                ),
                weights_path=(
                    config.hed.weights_path
                ),
                device=(
                    config.hed.resolved_device
                ),
            )

            base_inference = HEDInference(
                network=network,
                mean_bgr=config.hed.mean_bgr,
            )

            self.hed_predictor = (
                TiledHEDInference(
                    base_inference=base_inference,
                    tile_size=(
                        config.hed.tile_size
                    ),
                    overlap=config.hed.overlap,
                )
            )

        else:
            self.hed_predictor = hed_predictor

    def analyze_cover(
        self,
        source: ImageSource,
        required_payload_bits: int,
        channels_per_selected_pixel: int = 1,
        reserved_position_count: int = 0,
    ) -> VisionAnalysisResult:
        """Run the complete Member 1 vision pipeline."""

        total_start = perf_counter()

        preprocessing_start = perf_counter()

        rgb, image_info = load_and_validate_image(
            source,
            max_pixels=(
                self.config.image.max_pixels
            ),
        )

        luminance = rgb_to_luminance(rgb)

        preprocessing_seconds = (
            perf_counter()
            - preprocessing_start
        )

        hed_start = perf_counter()

        raw_hed = self.hed_predictor.predict(
            rgb
        )

        hed_seconds = (
            perf_counter() - hed_start
        )

        entropy_start = perf_counter()

        raw_entropy = calculate_local_entropy(
            luminance,
            window_size=(
                self.config
                .texture
                .entropy_window
            ),
            bins=(
                self.config
                .texture
                .entropy_bins
            ),
        )

        entropy_seconds = (
            perf_counter() - entropy_start
        )

        variance_start = perf_counter()

        raw_variance = calculate_local_variance(
            luminance,
            window_size=(
                self.config
                .texture
                .variance_window
            ),
        )

        variance_seconds = (
            perf_counter() - variance_start
        )

        fusion_start = perf_counter()

        feature_maps = fuse_feature_maps(
            hed_map=raw_hed,
            entropy_map=raw_entropy,
            variance_map=raw_variance,
            hed_weight=(
                self.config
                .fusion
                .hed_weight
            ),
            entropy_weight=(
                self.config
                .fusion
                .entropy_weight
            ),
            variance_weight=(
                self.config
                .fusion
                .variance_weight
            ),
            median_kernel=(
                self.config
                .fusion
                .median_kernel
            ),
        )

        fusion_seconds = (
            perf_counter() - fusion_start
        )

        block_start = perf_counter()

        block_map = (
            generate_capacity_aware_block_map(
                heatmap=(
                    feature_maps
                    .fused_heatmap
                ),
                required_payload_bits=(
                    required_payload_bits
                ),
                block_size=(
                    self.config
                    .block_map
                    .block_size
                ),
                channels_per_selected_pixel=(
                    channels_per_selected_pixel
                ),
                reserved_position_count=(
                    reserved_position_count
                ),
                safety_margin=(
                    self.config
                    .block_map
                    .safety_margin
                ),
                score_method=(
                    self.config
                    .block_map
                    .score_method
                ),
            )
        )

        block_seconds = (
            perf_counter() - block_start
        )

        encoding_start = perf_counter()

        map_encoding = encode_block_map(
            block_map.selected_blocks
        )

        encoding_seconds = (
            perf_counter() - encoding_start
        )

        total_seconds = (
            perf_counter() - total_start
        )

        tile_count = getattr(
            self.hed_predictor,
            "last_tile_count",
            1,
        )

        timings = {
            "preprocessing": (
                preprocessing_seconds
            ),
            "hed": hed_seconds,
            "entropy": entropy_seconds,
            "variance": variance_seconds,
            "fusion": fusion_seconds,
            "block_selection": block_seconds,
            "map_encoding": encoding_seconds,
            "total": total_seconds,
        }

        config_used = {
            "hed_framework": (
                self.config.hed.framework
            ),
            "hed_device": (
                self.config
                .hed
                .resolved_device
            ),
            "hed_tile_size": (
                self.config.hed.tile_size
            ),
            "hed_overlap": (
                self.config.hed.overlap
            ),
            "hed_tile_count": tile_count,
            "entropy_window": (
                self.config
                .texture
                .entropy_window
            ),
            "entropy_bins": (
                self.config
                .texture
                .entropy_bins
            ),
            "variance_window": (
                self.config
                .texture
                .variance_window
            ),
            "fusion_weights": {
                "hed": (
                    self.config
                    .fusion
                    .hed_weight
                ),
                "entropy": (
                    self.config
                    .fusion
                    .entropy_weight
                ),
                "variance": (
                    self.config
                    .fusion
                    .variance_weight
                ),
            },
            "block_size": (
                self.config
                .block_map
                .block_size
            ),
            "safety_margin": (
                self.config
                .block_map
                .safety_margin
            ),
        }

        return VisionAnalysisResult(
            image_info=image_info,
            feature_maps=feature_maps,
            block_map=block_map,
            map_encoding=map_encoding,
            timings=timings,
            config_used=config_used,
        )


def analyze_cover(
    source: ImageSource,
    required_payload_bits: int,
    config_path: str | Path = (
        "configs/vision.yaml"
    ),
    channels_per_selected_pixel: int = 1,
    reserved_position_count: int = 0,
) -> VisionAnalysisResult:
    """
    Convenience function for one-off analysis.

    Flask should eventually create one service object and reuse it
    instead of loading HED for every request.
    """

    config = load_vision_config(
        config_path
    )

    service = GuardianPixelVisionService(
        config=config
    )

    return service.analyze_cover(
        source=source,
        required_payload_bits=(
            required_payload_bits
        ),
        channels_per_selected_pixel=(
            channels_per_selected_pixel
        ),
        reserved_position_count=(
            reserved_position_count
        ),
    )