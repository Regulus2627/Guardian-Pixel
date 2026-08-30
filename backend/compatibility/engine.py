"""GuardianPixel cover-payload compatibility engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.compatibility.categories import (
    CategoryThresholds,
    categorize_texture_score,
)
from backend.compatibility.features import (
    extract_cover_features,
)
from backend.core.positions import (
    BOOTSTRAP_BIT_COUNT,
)
from backend.core.protocol import (
    EMBEDDING_POLICY_ADAPTIVE_LSB,
    LocationMetadata,
    serialize_metadata,
)
from backend.crypto.encryption import (
    GCM_TAG_SIZE,
)
from backend.crypto.envelope import (
    build_envelope,
)
from backend.vision.block_map import (
    BlockMapError,
    generate_capacity_aware_block_map,
)
from backend.vision.map_encoding import (
    encode_block_map,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)


class CompatibilityEngineError(ValueError):
    """Raised when compatibility cannot be assessed."""


@dataclass(frozen=True, slots=True)
class CompatibilityAssessment:
    """Pre-embedding cover-payload compatibility result."""

    compatible: bool
    recommended: bool
    classification: str
    reason: str
    is_provisional: bool

    cover_category: str
    texture_score: float

    image_width: int
    image_height: int
    total_pixels: int

    payload_type: str
    original_payload_bytes: int
    envelope_bytes: int
    encrypted_packet_bits: int

    bootstrap_bits: int
    metadata_bits: int
    total_required_bits: int
    required_positions_with_safety: int

    maximum_positions: int
    remaining_positions: int
    headroom_ratio: float

    selected_blocks: int
    total_blocks: int
    selected_percentage: float

    map_encoding: str
    encoded_map_bytes: int


def _build_payload_envelope(
    payload: bytes,
    payload_type: str,
    filename: str,
    mime_type: str,
) -> bytes:
    """Build the correct envelope for text/file/image input."""

    if payload_type == "text":
        return build_envelope(
            payload=payload,
            payload_type="text",
            filename="",
            mime_type="text/plain",
            enable_compression=True,
        )

    if payload_type in {
        "file",
        "image",
    }:
        if not filename:
            raise CompatibilityEngineError(
                "File and image payloads require a filename."
            )

        return build_envelope(
            payload=payload,
            payload_type="file",
            filename=filename,
            mime_type=mime_type,
            enable_compression=True,
        )

    raise CompatibilityEngineError(
        "payload_type must be text, file or image."
    )


def assess_compatibility(
    rgb: np.ndarray,
    payload: bytes,
    payload_type: str,
    vision_service: GuardianPixelVisionService,
    thresholds: CategoryThresholds,
    filename: str = "",
    mime_type: str = "application/octet-stream",
) -> CompatibilityAssessment:
    """
    Assess whether a payload can fit inside a cover.

    This is a pre-embedding assessment. Security-safe category limits
    are provisional until baseline/steganalysis evaluation is complete.
    """

    if not isinstance(rgb, np.ndarray):
        raise CompatibilityEngineError(
            "Cover must be a NumPy RGB array."
        )

    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
    ):
        raise CompatibilityEngineError(
            "Cover must be uint8 H×W×3 RGB."
        )

    if not isinstance(payload, bytes):
        raise CompatibilityEngineError(
            "Payload must be bytes."
        )

    envelope = _build_payload_envelope(
        payload=payload,
        payload_type=payload_type,
        filename=filename,
        mime_type=mime_type,
    )

    encrypted_packet_bits = (
        len(envelope) + GCM_TAG_SIZE
    ) * 8

    # Analyse cover characteristics once without a fake payload.
    cover_analysis = (
        vision_service.analyze_cover(
            source=rgb,
            required_payload_bits=0,
            channels_per_selected_pixel=1,
            reserved_position_count=0,
        )
    )

    cover_features = (
        extract_cover_features(
            image_id="runtime-cover",
            vision_result=cover_analysis,
        )
    )

    cover_category = (
        categorize_texture_score(
            cover_features.texture_score,
            thresholds,
        )
    )

    height, width = rgb.shape[:2]

    block_size = (
        vision_service
        .config
        .block_map
        .block_size
    )

    block_rows = math.ceil(
        height / block_size
    )

    block_columns = math.ceil(
        width / block_size
    )

    total_blocks = (
        block_rows * block_columns
    )

    # Map encoding is always no larger than its packed RAW bitmap.
    maximum_raw_map_bytes = math.ceil(
        total_blocks / 8
    )

    maximum_metadata_bits = (
        24 + maximum_raw_map_bytes
    ) * 8

    maximum_positions = (
        height * width
    )

    provisional_required_positions = (
        math.ceil(
            encrypted_packet_bits
            * (
                1.0
                + vision_service
                .config
                .block_map
                .safety_margin
            )
        )
        + BOOTSTRAP_BIT_COUNT
        + maximum_metadata_bits
    )

    try:
        block_map = (
            generate_capacity_aware_block_map(
                heatmap=(
                    cover_analysis
                    .feature_maps
                    .fused_heatmap
                ),
                required_payload_bits=(
                    encrypted_packet_bits
                ),
                block_size=block_size,
                channels_per_selected_pixel=1,
                reserved_position_count=(
                    BOOTSTRAP_BIT_COUNT
                    + maximum_metadata_bits
                ),
                safety_margin=(
                    vision_service
                    .config
                    .block_map
                    .safety_margin
                ),
                score_method=(
                    vision_service
                    .config
                    .block_map
                    .score_method
                ),
            )
        )

    except BlockMapError:
        remaining_positions = max(
            0,
            maximum_positions
            - provisional_required_positions,
        )

        return CompatibilityAssessment(
            compatible=False,
            recommended=False,
            classification="Poor",
            reason=(
                "The complete encrypted payload and protocol "
                "overhead exceed this cover's one-channel capacity."
            ),
            is_provisional=True,
            cover_category=cover_category,
            texture_score=(
                cover_features.texture_score
            ),
            image_width=width,
            image_height=height,
            total_pixels=maximum_positions,
            payload_type=payload_type,
            original_payload_bytes=len(payload),
            envelope_bytes=len(envelope),
            encrypted_packet_bits=(
                encrypted_packet_bits
            ),
            bootstrap_bits=(
                BOOTSTRAP_BIT_COUNT
            ),
            metadata_bits=(
                maximum_metadata_bits
            ),
            total_required_bits=(
                encrypted_packet_bits
                + BOOTSTRAP_BIT_COUNT
                + maximum_metadata_bits
            ),
            required_positions_with_safety=(
                provisional_required_positions
            ),
            maximum_positions=maximum_positions,
            remaining_positions=(
                remaining_positions
            ),
            headroom_ratio=0.0,
            selected_blocks=0,
            total_blocks=total_blocks,
            selected_percentage=0.0,
            map_encoding="UNAVAILABLE",
            encoded_map_bytes=0,
        )

    map_encoding = encode_block_map(
        block_map.selected_blocks
    )

    metadata = LocationMetadata(
        encoding_type=(
            map_encoding.encoding_type
        ),
        embedding_policy=(
            EMBEDDING_POLICY_ADAPTIVE_LSB
        ),
        channels_per_selected_pixel=1,
        block_rows=(
            block_map.block_rows
        ),
        block_columns=(
            block_map.block_columns
        ),
        original_map_bit_count=(
            map_encoding.original_bit_count
        ),
        encoded_map_data=(
            map_encoding.encoded_data
        ),
    )

    metadata_bytes = serialize_metadata(
        metadata
    )

    metadata_bits = (
        len(metadata_bytes) * 8
    )

    total_required_bits = (
        BOOTSTRAP_BIT_COUNT
        + metadata_bits
        + encrypted_packet_bits
    )

    required_positions_with_safety = (
        math.ceil(
            encrypted_packet_bits
            * (
                1.0
                + vision_service
                .config
                .block_map
                .safety_margin
            )
        )
        + BOOTSTRAP_BIT_COUNT
        + metadata_bits
    )

    remaining_positions = max(
        0,
        maximum_positions
        - required_positions_with_safety,
    )

    headroom_ratio = (
        remaining_positions
        / maximum_positions
    )

    selected_percentage = (
        block_map.selected_percentage
    )

    # Provisional recommendation tiers.
    # These are replaced later using baseline/steganalysis results.
    if (
        selected_percentage <= 20
        and headroom_ratio >= 0.50
    ):
        classification = "Excellent"
        recommended = True
        reason = (
            "The payload fits with high capacity headroom "
            "and low selected-block usage."
        )

    elif (
        selected_percentage <= 40
        and headroom_ratio >= 0.25
    ):
        classification = "Good"
        recommended = True
        reason = (
            "The payload fits with acceptable provisional "
            "capacity headroom."
        )

    elif selected_percentage <= 70:
        classification = "Moderate"
        recommended = False
        reason = (
            "The payload fits technically, but it uses a large "
            "part of the cover and should be treated as risky."
        )

    else:
        classification = "Poor"
        recommended = False
        reason = (
            "The payload fits technically but requires excessive "
            "block usage. Use a larger cover or smaller payload."
        )

    return CompatibilityAssessment(
        compatible=True,
        recommended=recommended,
        classification=classification,
        reason=reason,
        is_provisional=True,
        cover_category=cover_category,
        texture_score=(
            cover_features.texture_score
        ),
        image_width=width,
        image_height=height,
        total_pixels=maximum_positions,
        payload_type=payload_type,
        original_payload_bytes=len(payload),
        envelope_bytes=len(envelope),
        encrypted_packet_bits=(
            encrypted_packet_bits
        ),
        bootstrap_bits=BOOTSTRAP_BIT_COUNT,
        metadata_bits=metadata_bits,
        total_required_bits=(
            total_required_bits
        ),
        required_positions_with_safety=(
            required_positions_with_safety
        ),
        maximum_positions=(
            maximum_positions
        ),
        remaining_positions=(
            remaining_positions
        ),
        headroom_ratio=float(
            headroom_ratio
        ),
        selected_blocks=(
            block_map.selected_block_count
        ),
        total_blocks=(
            block_map.total_block_count
        ),
        selected_percentage=(
            selected_percentage
        ),
        map_encoding=(
            map_encoding.encoding_type
        ),
        encoded_map_bytes=(
            map_encoding.encoded_byte_count
        ),
    )