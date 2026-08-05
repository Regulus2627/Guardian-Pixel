"""Complete GuardianPixel receiver-side extraction service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.bitstream import (
    BitstreamError,
    bits_to_bytes,
)
from backend.core.positions import (
    BOOTSTRAP_BIT_COUNT,
    PositionError,
    generate_metadata_positions,
    generate_payload_positions,
    generate_public_bootstrap_positions,
)
from backend.core.protocol import (
    BootstrapHeader,
    LocationMetadata,
    ProtocolError,
    parse_bootstrap,
    parse_metadata,
)
from backend.crypto.encryption import (
    DEFAULT_AAD,
    EncryptedPayload,
    EncryptionError,
    decrypt_envelope,
    recover_key_material,
)
from backend.crypto.envelope import (
    EnvelopeError,
    ParsedEnvelope,
    parse_envelope,
)
from backend.crypto.kdf import (
    KeyDerivationError,
)
from backend.stego.lsb_matching import (
    LSBMatchingError,
    extract_bits_at_positions,
)
from backend.vision.map_encoding import (
    MapEncodingError,
    decode_block_map,
)


class ExtractorError(RuntimeError):
    """Raised when GuardianPixel extraction fails."""


@dataclass(frozen=True, slots=True)
class ExtractResult:
    """Complete receiver-side extraction result."""

    parsed_envelope: ParsedEnvelope
    bootstrap: BootstrapHeader
    metadata: LocationMetadata

    metadata_bytes: bytes
    encrypted_packet_bytes: bytes

    bootstrap_positions: np.ndarray
    metadata_positions: np.ndarray
    payload_positions: np.ndarray


def _extract_secret_internal(
    stego_rgb: np.ndarray,
    passphrase: str,
) -> ExtractResult:
    """Internal extraction implementation."""

    if not isinstance(
        stego_rgb,
        np.ndarray,
    ):
        raise ExtractorError(
            "Stego image must be a NumPy array."
        )

    if (
        stego_rgb.ndim != 3
        or stego_rgb.shape[2] != 3
    ):
        raise ExtractorError(
            "Stego image must have shape H×W×3."
        )

    if stego_rgb.size == 0:
        raise ExtractorError(
            "Stego image cannot be empty."
        )

    if stego_rgb.dtype != np.uint8:
        raise ExtractorError(
            "Stego image must use uint8 values."
        )

    image_height, image_width = (
        stego_rgb.shape[:2]
    )

    # Bootstrap positions are public and depend only on image size.
    bootstrap_positions = (
        generate_public_bootstrap_positions(
            image_height=image_height,
            image_width=image_width,
            count=BOOTSTRAP_BIT_COUNT,
        )
    )

    bootstrap_bits = (
        extract_bits_at_positions(
            stego_rgb,
            bootstrap_positions,
        )
    )

    bootstrap_bytes = bits_to_bytes(
        bootstrap_bits
    )

    bootstrap = parse_bootstrap(
        bootstrap_bytes
    )

    if (
        bootstrap.image_width
        != image_width
        or bootstrap.image_height
        != image_height
    ):
        raise ExtractorError(
            "Stego image dimensions do not match "
            "the embedded bootstrap."
        )

    # The passphrase and embedded salt recreate the same position key.
    key_material = recover_key_material(
        passphrase=passphrase,
        salt=bootstrap.salt,
    )

    metadata_positions = (
        generate_metadata_positions(
            image_height=image_height,
            image_width=image_width,
            metadata_bit_count=(
                bootstrap.metadata_bit_length
            ),
            position_key=(
                key_material.position_key
            ),
            bootstrap_positions=(
                bootstrap_positions
            ),
        )
    )

    metadata_bits = (
        extract_bits_at_positions(
            stego_rgb,
            metadata_positions,
        )
    )

    metadata_bytes = bits_to_bytes(
        metadata_bits
    )

    metadata = parse_metadata(
        metadata_bytes
    )

    selected_blocks = decode_block_map(
        encoding_type=(
            metadata.encoding_type
        ),
        encoded_data=(
            metadata.encoded_map_data
        ),
        block_rows=(
            metadata.block_rows
        ),
        block_columns=(
            metadata.block_columns
        ),
    )

    if (
        selected_blocks.size
        != metadata.original_map_bit_count
    ):
        raise ExtractorError(
            "Decoded block-map size is incorrect."
        )

    expected_rows = (
        image_height
        + bootstrap.block_size
        - 1
    ) // bootstrap.block_size

    expected_columns = (
        image_width
        + bootstrap.block_size
        - 1
    ) // bootstrap.block_size

    if selected_blocks.shape != (
        expected_rows,
        expected_columns,
    ):
        raise ExtractorError(
            "Decoded block-map dimensions do not "
            "match the stego image."
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
                selected_blocks
            ),
            image_height=image_height,
            image_width=image_width,
            block_size=(
                bootstrap.block_size
            ),
            payload_bit_count=(
                bootstrap.packet_bit_length
            ),
            position_key=(
                key_material.position_key
            ),
            excluded_positions=(
                excluded_positions
            ),
        )
    )

    packet_bits = (
        extract_bits_at_positions(
            stego_rgb,
            payload_positions,
        )
    )

    encrypted_packet_bytes = (
        bits_to_bytes(packet_bits)
    )

    encrypted_payload = EncryptedPayload(
        salt=bootstrap.salt,
        nonce=bootstrap.nonce,
        ciphertext=(
            encrypted_packet_bytes
        ),
    )

    authenticated_metadata = (
        DEFAULT_AAD
        + metadata_bytes
    )

    recovered_envelope = (
        decrypt_envelope(
            encrypted_payload=(
                encrypted_payload
            ),
            passphrase=passphrase,
            aad=authenticated_metadata,
        )
    )

    parsed_envelope = parse_envelope(
        recovered_envelope
    )

    return ExtractResult(
        parsed_envelope=parsed_envelope,
        bootstrap=bootstrap,
        metadata=metadata,
        metadata_bytes=metadata_bytes,
        encrypted_packet_bytes=(
            encrypted_packet_bytes
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
    )


def extract_secret(
    stego_rgb: np.ndarray,
    passphrase: str,
) -> ExtractResult:
    """
    Safely extract a GuardianPixel secret.

    Wrong passwords, modified metadata and damaged ciphertext return
    one common error. This avoids revealing which verification stage
    failed.
    """

    try:
        return _extract_secret_internal(
            stego_rgb=stego_rgb,
            passphrase=passphrase,
        )

    except ExtractorError:
        raise

    except (
        BitstreamError,
        PositionError,
        ProtocolError,
        EncryptionError,
        EnvelopeError,
        KeyDerivationError,
        LSBMatchingError,
        MapEncodingError,
    ) as error:
        raise ExtractorError(
            "Extraction failed: wrong passphrase "
            "or modified image."
        ) from error