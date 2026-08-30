"""Complete GuardianPixel sender-side embedding service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from backend.core.bitstream import (
    bytes_to_bits,
)
from backend.core.positions import (
    BOOTSTRAP_BIT_COUNT,
    generate_metadata_positions,
    generate_payload_positions,
    generate_public_bootstrap_positions,
)
from backend.core.protocol import (
    CHANNEL_MODE_KEYED_SINGLE,
    EMBEDDING_POLICY_ADAPTIVE_LSB,
    BootstrapHeader,
    LocationMetadata,
    serialize_bootstrap,
    serialize_metadata,
)
from backend.crypto.encryption import (
    DEFAULT_AAD,
    GCM_TAG_SIZE,
    encrypt_envelope,
    recover_key_material,
)
from backend.crypto.envelope import (
    build_envelope,
)
from backend.stego.lsb_matching import (
    embed_bits_at_positions,
)
from backend.vision.block_map import (
    generate_capacity_aware_block_map,
)
from backend.vision.map_encoding import (
    encode_block_map,
)
from backend.vision.schemas import (
    BlockMapResult,
    MapEncodingResult,
    VisionAnalysisResult,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)


class EmbedderError(RuntimeError):
    """Raised when complete GuardianPixel embedding fails."""


PUBLIC_DIRECTION_KEY = hashlib.sha256(
    b"GuardianPixel-GPX1-public-bootstrap-direction"
).digest()


@dataclass(frozen=True, slots=True)
class EmbedResult:
    """Complete sender-side embedding result."""

    stego_image: np.ndarray
    bootstrap: BootstrapHeader
    metadata: LocationMetadata

    metadata_bytes: bytes
    encrypted_packet_bytes: bytes

    block_map: BlockMapResult
    map_encoding: MapEncodingResult
    vision_result: VisionAnalysisResult

    bootstrap_positions: np.ndarray
    metadata_positions: np.ndarray
    payload_positions: np.ndarray

    total_embedded_bits: int
    modified_channel_count: int
    maximum_absolute_change: int


def _select_final_block_map(
    fused_heatmap: np.ndarray,
    packet_bit_count: int,
    block_size: int,
    channels_per_selected_pixel: int,
    safety_margin: float,
    score_method: str,
    maximum_iterations: int = 12,
) -> tuple[
    BlockMapResult,
    MapEncodingResult,
    LocationMetadata,
    bytes,
]:
    """
    Select a valid block map without requiring exact size equality.

    A result is accepted as soon as the selected capacity can hold
    the payload, bootstrap and actual serialized metadata.
    """

    metadata_bit_estimate = 24 * 8

    for _ in range(maximum_iterations):
        reserved_positions = (
            BOOTSTRAP_BIT_COUNT
            + metadata_bit_estimate
        )

        block_map = (
            generate_capacity_aware_block_map(
                heatmap=fused_heatmap,
                required_payload_bits=(
                    packet_bit_count
                ),
                block_size=block_size,
                channels_per_selected_pixel=(
                    channels_per_selected_pixel
                ),
                reserved_position_count=(
                    reserved_positions
                ),
                safety_margin=safety_margin,
                score_method=score_method,
            )
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
            channels_per_selected_pixel=(
                channels_per_selected_pixel
            ),
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

        metadata_bytes = (
            serialize_metadata(metadata)
        )

        actual_metadata_bits = (
            len(metadata_bytes) * 8
        )

        actual_required_positions = (
            block_map.target_position_count
            - metadata_bit_estimate
            + actual_metadata_bits
        )

        if (
            block_map.selected_position_count
            >= actual_required_positions
        ):
            return (
                block_map,
                map_encoding,
                metadata,
                metadata_bytes,
            )

        metadata_bit_estimate = max(
            actual_metadata_bits,
            metadata_bit_estimate + 8,
        )

    # Guaranteed conservative fallback:
    # map encoding cannot be larger than the raw packed bitmap.
    block_rows = (
        fused_heatmap.shape[0]
        + block_size
        - 1
    ) // block_size

    block_columns = (
        fused_heatmap.shape[1]
        + block_size
        - 1
    ) // block_size

    raw_map_bytes = (
        block_rows * block_columns
        + 7
    ) // 8

    maximum_metadata_bits = (
        24 + raw_map_bytes
    ) * 8

    block_map = (
        generate_capacity_aware_block_map(
            heatmap=fused_heatmap,
            required_payload_bits=(
                packet_bit_count
            ),
            block_size=block_size,
            channels_per_selected_pixel=(
                channels_per_selected_pixel
            ),
            reserved_position_count=(
                BOOTSTRAP_BIT_COUNT
                + maximum_metadata_bits
            ),
            safety_margin=safety_margin,
            score_method=score_method,
        )
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
        channels_per_selected_pixel=(
            channels_per_selected_pixel
        ),
        block_rows=block_map.block_rows,
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

    return (
        block_map,
        map_encoding,
        metadata,
        metadata_bytes,
    )

def embed_secret(
    rgb: np.ndarray,
    payload: bytes,
    payload_type: str,
    passphrase: str,
    vision_service: GuardianPixelVisionService,
    filename: str = "",
    mime_type: str = "application/octet-stream",
    enable_compression: bool = True,
    channels_per_selected_pixel: int = 1,
    precomputed_vision_result: (
        VisionAnalysisResult | None
    ) = None,
) -> EmbedResult:
    """Encrypt and embed a secret into an RGB cover image."""

    if not isinstance(
        vision_service,
        GuardianPixelVisionService,
    ):
        raise EmbedderError(
            "vision_service must be GuardianPixelVisionService."
        )

    envelope = build_envelope(
        payload=payload,
        payload_type=payload_type,
        filename=filename,
        mime_type=mime_type,
        enable_compression=enable_compression,
    )

    # AES-GCM adds a fixed 16-byte authentication tag.
    expected_ciphertext_bytes = (
        len(envelope)
        + GCM_TAG_SIZE
    )

    packet_bit_count = (
        expected_ciphertext_bytes * 8
    )

    # One full AI/CV pass. The initial block result is replaced after
    # exact metadata-size stabilization below.
    if precomputed_vision_result is None:
        initial_vision_result = (
            vision_service.analyze_cover(
                source=rgb,
                required_payload_bits=(
                    packet_bit_count
                ),
                channels_per_selected_pixel=(
                    channels_per_selected_pixel
                ),
                reserved_position_count=(
                    BOOTSTRAP_BIT_COUNT
                    + 24 * 8
                ),
            )
        )

    else:
        if not isinstance(
            precomputed_vision_result,
            VisionAnalysisResult,
        ):
            raise EmbedderError(
                "precomputed_vision_result must be "
                "VisionAnalysisResult or None."
            )

        if (
            precomputed_vision_result
            .image_info
            .height
            != rgb.shape[0]
            or precomputed_vision_result
            .image_info
            .width
            != rgb.shape[1]
        ):
            raise EmbedderError(
                "Precomputed vision result dimensions "
                "do not match the cover image."
            )

        expected_shape = rgb.shape[:2]

        if (
            precomputed_vision_result
            .feature_maps
            .fused_heatmap
            .shape
            != expected_shape
        ):
            raise EmbedderError(
                "Precomputed fused heatmap dimensions "
                "do not match the cover image."
            )

        initial_vision_result = (
            precomputed_vision_result
        )

    config = vision_service.config

    (
        final_block_map,
        final_map_encoding,
        metadata,
        metadata_bytes,
    ) = _select_final_block_map(
        fused_heatmap=(
            initial_vision_result
            .feature_maps
            .fused_heatmap
        ),
        packet_bit_count=packet_bit_count,
        block_size=(
            config.block_map.block_size
        ),
        channels_per_selected_pixel=(
            channels_per_selected_pixel
        ),
        safety_margin=(
            config.block_map.safety_margin
        ),
        score_method=(
            config.block_map.score_method
        ),
    )

    authenticated_metadata = (
        DEFAULT_AAD
        + metadata_bytes
    )

    encrypted_payload = (
        encrypt_envelope(
            envelope=envelope,
            passphrase=passphrase,
            aad=authenticated_metadata,
        )
    )

    encrypted_packet_bytes = (
        encrypted_payload.ciphertext
    )

    if (
        len(encrypted_packet_bytes) * 8
        != packet_bit_count
    ):
        raise EmbedderError(
            "Encrypted packet size changed unexpectedly."
        )

    bootstrap = BootstrapHeader(
        flags=0,
        block_size=(
            final_block_map.block_size
        ),
        channel_mode=(
            CHANNEL_MODE_KEYED_SINGLE
        ),
        image_width=rgb.shape[1],
        image_height=rgb.shape[0],
        salt=encrypted_payload.salt,
        nonce=encrypted_payload.nonce,
        metadata_bit_length=(
            len(metadata_bytes) * 8
        ),
        packet_bit_length=(
            packet_bit_count
        ),
    )

    bootstrap_bytes = (
        serialize_bootstrap(bootstrap)
    )

    bootstrap_bits = bytes_to_bits(
        bootstrap_bytes
    )

    metadata_bits = bytes_to_bits(
        metadata_bytes
    )

    packet_bits = bytes_to_bits(
        encrypted_packet_bytes
    )

    bootstrap_positions = (
        generate_public_bootstrap_positions(
            image_height=rgb.shape[0],
            image_width=rgb.shape[1],
            count=bootstrap_bits.size,
        )
    )

    key_material = recover_key_material(
        passphrase=passphrase,
        salt=encrypted_payload.salt,
    )

    metadata_positions = (
        generate_metadata_positions(
            image_height=rgb.shape[0],
            image_width=rgb.shape[1],
            metadata_bit_count=(
                metadata_bits.size
            ),
            position_key=(
                key_material.position_key
            ),
            bootstrap_positions=(
                bootstrap_positions
            ),
        )
    )

    excluded_positions = np.concatenate(
        [
            bootstrap_positions,
            metadata_positions,
        ]
    )

    payload_positions = (
        generate_payload_positions(
            selected_blocks=(
                final_block_map
                .selected_blocks
            ),
            image_height=rgb.shape[0],
            image_width=rgb.shape[1],
            block_size=(
                final_block_map.block_size
            ),
            payload_bit_count=(
                packet_bits.size
            ),
            position_key=(
                key_material.position_key
            ),
            excluded_positions=(
                excluded_positions
            ),
        )
    )

    bootstrap_result = (
        embed_bits_at_positions(
            rgb=rgb,
            bits=bootstrap_bits,
            positions=(
                bootstrap_positions
            ),
            direction_key=(
                PUBLIC_DIRECTION_KEY
            ),
        )
    )

    metadata_result = (
        embed_bits_at_positions(
            rgb=(
                bootstrap_result
                .stego_image
            ),
            bits=metadata_bits,
            positions=(
                metadata_positions
            ),
            direction_key=(
                key_material.position_key
            ),
        )
    )

    payload_result = (
        embed_bits_at_positions(
            rgb=(
                metadata_result
                .stego_image
            ),
            bits=packet_bits,
            positions=(
                payload_positions
            ),
            direction_key=(
                key_material.position_key
            ),
        )
    )

    stego_image = (
        payload_result.stego_image
    )

    difference = np.abs(
        stego_image.astype(np.int16)
        - rgb.astype(np.int16)
    )

    final_modified_count = int(
        np.count_nonzero(difference)
    )

    maximum_change = int(
        difference.max()
    )

    final_vision_result = VisionAnalysisResult(
        image_info=(
            initial_vision_result.image_info
        ),
        feature_maps=(
            initial_vision_result.feature_maps
        ),
        block_map=final_block_map,
        map_encoding=final_map_encoding,
        raw_feature_maps=(
            initial_vision_result
            .raw_feature_maps
        ),
        timings=dict(
            initial_vision_result.timings
        ),
        config_used=dict(
            initial_vision_result.config_used
        ),
    )

    return EmbedResult(
        stego_image=stego_image,
        bootstrap=bootstrap,
        metadata=metadata,
        metadata_bytes=metadata_bytes,
        encrypted_packet_bytes=(
            encrypted_packet_bytes
        ),
        block_map=final_block_map,
        map_encoding=final_map_encoding,
        vision_result=(
            final_vision_result
        ),
        bootstrap_positions=(
            bootstrap_positions
        ),
        metadata_positions=(
            metadata_positions
        ),
        payload_positions=(
            payload_positions
        ),
        total_embedded_bits=int(
            bootstrap_bits.size
            + metadata_bits.size
            + packet_bits.size
        ),
        modified_channel_count=(
            final_modified_count
        ),
        maximum_absolute_change=(
            maximum_change
        ),
    )