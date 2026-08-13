"""Cover-image feature extraction for compatibility analysis."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from backend.vision.schemas import (
    VisionAnalysisResult,
)


class CoverFeatureError(ValueError):
    """Raised when cover features cannot be extracted."""


@dataclass(frozen=True, slots=True)
class CoverFeatures:
    """Numerical characteristics of one cover image."""

    image_id: str

    width: int
    height: int
    total_pixels: int
    channels: int
    original_format: str

    raw_hed_mean: float
    raw_hed_std: float

    raw_entropy_mean: float
    raw_entropy_std: float

    raw_variance_mean: float
    raw_variance_std: float

    hed_mean: float
    hed_std: float
    hed_p25: float
    hed_p50: float
    hed_p75: float
    edge_density: float

    entropy_mean: float
    entropy_std: float
    entropy_p25: float
    entropy_p50: float
    entropy_p75: float

    variance_mean: float
    variance_std: float
    variance_p25: float
    variance_p50: float
    variance_p75: float

    fused_mean: float
    fused_std: float
    fused_p25: float
    fused_p50: float
    fused_p75: float

    smooth_pixel_percentage: float
    high_score_pixel_percentage: float

    block_size: int
    total_blocks: int
    selected_blocks: int
    selected_percentage: float

    encoded_map_bytes: int
    map_compression_ratio: float

    texture_score: float
    analysis_time_seconds: float


def _validate_map(
    feature_map: np.ndarray,
    name: str,
) -> None:
    """Validate one two-dimensional feature map."""

    if not isinstance(
        feature_map,
        np.ndarray,
    ):
        raise CoverFeatureError(
            f"{name} must be a NumPy array."
        )

    if feature_map.ndim != 2:
        raise CoverFeatureError(
            f"{name} must have shape H×W."
        )

    if feature_map.size == 0:
        raise CoverFeatureError(
            f"{name} cannot be empty."
        )

    if not np.all(np.isfinite(feature_map)):
        raise CoverFeatureError(
            f"{name} contains invalid values."
        )


def _map_statistics(
    feature_map: np.ndarray,
) -> dict[str, float]:
    """Calculate common feature-map statistics."""

    return {
        "mean": float(
            np.mean(feature_map)
        ),
        "std": float(
            np.std(feature_map)
        ),
        "p25": float(
            np.percentile(
                feature_map,
                25,
            )
        ),
        "p50": float(
            np.percentile(
                feature_map,
                50,
            )
        ),
        "p75": float(
            np.percentile(
                feature_map,
                75,
            )
        ),
    }


def extract_cover_features(
    image_id: str,
    vision_result: VisionAnalysisResult,
    edge_threshold: float = 0.50,
    smooth_threshold: float = 0.25,
    high_score_threshold: float = 0.75,
) -> CoverFeatures:
    """Convert one vision result into compatibility features."""

    if not isinstance(image_id, str):
        raise CoverFeatureError(
            "image_id must be a string."
        )

    if not image_id.strip():
        raise CoverFeatureError(
            "image_id cannot be empty."
        )

    if not isinstance(
        vision_result,
        VisionAnalysisResult,
    ):
        raise CoverFeatureError(
            "vision_result must be VisionAnalysisResult."
        )

    for threshold, threshold_name in [
        (edge_threshold, "edge_threshold"),
        (
            smooth_threshold,
            "smooth_threshold",
        ),
        (
            high_score_threshold,
            "high_score_threshold",
        ),
    ]:
        if not 0 <= threshold <= 1:
            raise CoverFeatureError(
                f"{threshold_name} must be "
                "between 0 and 1."
            )

    maps = vision_result.feature_maps
    raw_maps = vision_result.raw_feature_maps

    if raw_maps is None:
        raise CoverFeatureError(
            "Raw feature maps are required "
            "for cross-cover comparison."
        )

    normalized_maps = {
        "hed_map": maps.hed_map,
        "entropy_map": maps.entropy_map,
        "variance_map": maps.variance_map,
        "fused_heatmap": (
            maps.fused_heatmap
        ),
    }

    raw_feature_maps = {
        "raw_hed_map": raw_maps.hed_map,
        "raw_entropy_map": (
            raw_maps.entropy_map
        ),
        "raw_variance_map": (
            raw_maps.variance_map
        ),
    }

    for name, feature_map in {
        **normalized_maps,
        **raw_feature_maps,
    }.items():
        _validate_map(
            feature_map,
            name,
        )

    expected_shape = (
        vision_result.image_info.height,
        vision_result.image_info.width,
    )

    for name, feature_map in {
        **normalized_maps,
        **raw_feature_maps,
    }.items():
        if feature_map.shape != expected_shape:
            raise CoverFeatureError(
                f"{name} dimensions do not "
                "match image information."
            )

    hed_stats = _map_statistics(
        maps.hed_map
    )

    entropy_stats = _map_statistics(
        maps.entropy_map
    )

    variance_stats = _map_statistics(
        maps.variance_map
    )

    fused_stats = _map_statistics(
        maps.fused_heatmap
    )

    raw_hed_mean = float(
        np.mean(raw_maps.hed_map)
    )

    raw_hed_std = float(
        np.std(raw_maps.hed_map)
    )

    raw_entropy_mean = float(
        np.mean(raw_maps.entropy_map)
    )

    raw_entropy_std = float(
        np.std(raw_maps.entropy_map)
    )

    raw_variance_mean = float(
        np.mean(raw_maps.variance_map)
    )

    raw_variance_std = float(
        np.std(raw_maps.variance_map)
    )

    # Maximum Shannon entropy with 32 bins:
    # log2(32) = 5.
    entropy_strength = float(
        np.clip(
            raw_entropy_mean / 5.0,
            0.0,
            1.0,
        )
    )

    # Maximum theoretical variance for values from 0 to 255:
    # (255 - 0)^2 / 4 = 16256.25.
    variance_strength = float(
        np.clip(
            raw_variance_mean / 16256.25,
            0.0,
            1.0,
        )
    )

    hed_strength = float(
        np.clip(
            raw_hed_mean,
            0.0,
            1.0,
        )
    )

    texture_score = float(
        0.30 * hed_strength
        + 0.40 * entropy_strength
        + 0.30 * variance_strength
    )

    edge_density = float(
        np.mean(
            raw_maps.hed_map
            >= edge_threshold
        )
    )

    smooth_percentage = float(
        np.mean(
            maps.fused_heatmap
            <= smooth_threshold
        )
        * 100.0
    )

    high_score_percentage = float(
        np.mean(
            maps.fused_heatmap
            >= high_score_threshold
        )
        * 100.0
    )

    return CoverFeatures(
        image_id=image_id.strip(),

        width=vision_result.image_info.width,
        height=vision_result.image_info.height,
        total_pixels=(
            vision_result.image_info
            .total_pixels
        ),
        channels=(
            vision_result.image_info.channels
        ),
        original_format=(
            vision_result.image_info
            .original_format
        ),

        raw_hed_mean=raw_hed_mean,
        raw_hed_std=raw_hed_std,

        raw_entropy_mean=(
            raw_entropy_mean
        ),
        raw_entropy_std=(
            raw_entropy_std
        ),

        raw_variance_mean=(
            raw_variance_mean
        ),
        raw_variance_std=(
            raw_variance_std
        ),

        hed_mean=hed_stats["mean"],
        hed_std=hed_stats["std"],
        hed_p25=hed_stats["p25"],
        hed_p50=hed_stats["p50"],
        hed_p75=hed_stats["p75"],
        edge_density=edge_density,

        entropy_mean=entropy_stats["mean"],
        entropy_std=entropy_stats["std"],
        entropy_p25=entropy_stats["p25"],
        entropy_p50=entropy_stats["p50"],
        entropy_p75=entropy_stats["p75"],

        variance_mean=variance_stats["mean"],
        variance_std=variance_stats["std"],
        variance_p25=variance_stats["p25"],
        variance_p50=variance_stats["p50"],
        variance_p75=variance_stats["p75"],

        fused_mean=fused_stats["mean"],
        fused_std=fused_stats["std"],
        fused_p25=fused_stats["p25"],
        fused_p50=fused_stats["p50"],
        fused_p75=fused_stats["p75"],

        smooth_pixel_percentage=(
            smooth_percentage
        ),
        high_score_pixel_percentage=(
            high_score_percentage
        ),

        block_size=(
            vision_result.block_map
            .block_size
        ),
        total_blocks=(
            vision_result.block_map
            .total_block_count
        ),
        selected_blocks=(
            vision_result.block_map
            .selected_block_count
        ),
        selected_percentage=(
            vision_result.block_map
            .selected_percentage
        ),

        encoded_map_bytes=(
            vision_result.map_encoding
            .encoded_byte_count
        ),
        map_compression_ratio=(
            vision_result.map_encoding
            .compression_ratio
        ),

        texture_score=texture_score,

        analysis_time_seconds=float(
            vision_result.timings.get(
                "total",
                0.0,
            )
        ),
    )


def write_cover_features_csv(
    features: list[CoverFeatures],
    output_path: str | Path,
) -> Path:
    """Write cover characteristics to CSV."""

    if not features:
        raise CoverFeatureError(
            "At least one cover feature row is required."
        )

    if not all(
        isinstance(item, CoverFeatures)
        for item in features
    ):
        raise CoverFeatureError(
            "Every CSV item must be CoverFeatures."
        )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        asdict(item)
        for item in features
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return path