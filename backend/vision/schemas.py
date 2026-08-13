"""Data structures shared by GuardianPixel vision modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Information about a validated cover image."""

    width: int
    height: int
    channels: int
    original_format: str
    original_mode: str

    @property
    def total_pixels(self) -> int:
        return self.width * self.height


@dataclass(slots=True)
class FeatureMaps:
    """Normalized feature maps used for fusion and selection."""

    hed_map: np.ndarray
    entropy_map: np.ndarray
    variance_map: np.ndarray
    fused_heatmap: np.ndarray


@dataclass(slots=True)
class RawFeatureMaps:
    """Feature maps before per-image normalization."""

    hed_map: np.ndarray
    entropy_map: np.ndarray
    variance_map: np.ndarray


@dataclass(slots=True)
class BlockMapResult:
    """Result of capacity-aware heatmap block selection."""

    block_size: int
    block_rows: int
    block_columns: int

    selected_blocks: np.ndarray
    block_scores: np.ndarray

    selected_block_count: int
    total_block_count: int

    selected_position_count: int
    reserved_position_count: int
    usable_position_count: int

    required_payload_bits: int
    target_position_count: int

    @property
    def selected_percentage(self) -> float:
        if self.total_block_count == 0:
            return 0.0

        return (
            self.selected_block_count
            / self.total_block_count
            * 100.0
        )


@dataclass(frozen=True, slots=True)
class MapEncodingResult:
    """Compressed representation of a binary block map."""

    encoding_type: str
    encoded_data: bytes
    original_bit_count: int
    raw_byte_count: int
    encoded_byte_count: int

    @property
    def compression_ratio(self) -> float:
        if self.raw_byte_count == 0:
            return 1.0

        return (
            self.encoded_byte_count
            / self.raw_byte_count
        )


@dataclass(slots=True)
class VisionAnalysisResult:
    """Complete result returned by the vision service."""

    image_info: ImageInfo
    feature_maps: FeatureMaps
    block_map: BlockMapResult
    map_encoding: MapEncodingResult

    raw_feature_maps: RawFeatureMaps | None = None

    timings: dict[str, float] = field(
        default_factory=dict
    )

    config_used: dict = field(
        default_factory=dict
    )