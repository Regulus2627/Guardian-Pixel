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
    maximum_iterations: int = 10,
) -> tuple[
    BlockMapResult,
    MapEncodingResult,
    LocationMetadata,
    bytes,
]:
    """
    Recalculate block selection until metadata size becomes stable.

    Metadata size depends on map compression, while map selection also
    depends on reserved metadata capacity.
    """

    # Minimum metadata:
    # 20-byte metadata header + 4-byte CRC.
    metadata_bit_estimate = 24 * 8

    previous_state = None

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
                map_encoding
                .original_bit_count
            ),
            encoded_map_data=(
                map_encoding.encoded_data
            ),
        )

        metadata_bytes = (
            serialize_metadata(metadata)
        )

        new_metadata_bits = (
            len(metadata_bytes) * 8
        )

        current_state = (
            new_metadata_bits,
            block_map.selected_block_count,
            map_encoding.encoding_type,
            map_encoding.encoded_byte_count,
        )

        if (
            new_metadata_bits
            == metadata_bit_estimate
            and current_state
            == previous_state
        ):
            return (
                block_map,
                map_encoding,
                metadata,
                metadata_bytes,
            )

        previous_state = current_state
        metadata_bit_estimate = (
            new_metadata_bits
        )

    raise EmbedderError(
        "Block-map metadata size did not stabilize."
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

    initial_vision_result.block_map = (
        final_block_map
    )

    initial_vision_result.map_encoding = (
        final_map_encoding
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
            initial_vision_result
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