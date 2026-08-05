"""GuardianPixel GPX1 bootstrap and metadata protocol."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from backend.crypto.encryption import (
    NONCE_SIZE,
)
from backend.crypto.kdf import (
    SALT_SIZE,
)


class ProtocolError(ValueError):
    """Raised when GuardianPixel protocol data is invalid."""


BOOTSTRAP_MAGIC = b"GPX1"
BOOTSTRAP_VERSION = 1

METADATA_MAGIC = b"GPM1"
METADATA_VERSION = 1

CHANNEL_MODE_KEYED_SINGLE = 1
EMBEDDING_POLICY_ADAPTIVE_LSB = 1

ENCODING_TO_CODE = {
    "RAW": 1,
    "RLE": 2,
    "ZLIB": 3,
    "ALL_SELECTED": 4,
    "NONE_SELECTED": 5,
}

CODE_TO_ENCODING = {
    value: key
    for key, value in ENCODING_TO_CODE.items()
}

# Without CRC:
# magic, version, flags, block size, channel mode,
# width, height, salt, nonce,
# metadata bit length, packet bit length
BOOTSTRAP_BODY = struct.Struct(
    ">4sBBBBII16s12sIQ"
)

CRC_STRUCT = struct.Struct(">I")

BOOTSTRAP_SIZE = (
    BOOTSTRAP_BODY.size
    + CRC_STRUCT.size
)

# Without encoded map and CRC:
# magic, version, encoding code,
# embedding policy, channels per pixel,
# block rows, block columns,
# original map bit count,
# encoded-map byte count
METADATA_HEADER = struct.Struct(
    ">4sBBBBHHII"
)


@dataclass(frozen=True, slots=True)
class BootstrapHeader:
    """Fixed information required to begin extraction."""

    flags: int
    block_size: int
    channel_mode: int

    image_width: int
    image_height: int

    salt: bytes
    nonce: bytes

    metadata_bit_length: int
    packet_bit_length: int


@dataclass(frozen=True, slots=True)
class LocationMetadata:
    """Variable metadata containing the encoded block map."""

    encoding_type: str
    embedding_policy: int
    channels_per_selected_pixel: int

    block_rows: int
    block_columns: int
    original_map_bit_count: int

    encoded_map_data: bytes


def _crc32(data: bytes) -> int:
    """Calculate unsigned CRC32."""

    return zlib.crc32(data) & 0xFFFFFFFF


def serialize_bootstrap(
    bootstrap: BootstrapHeader,
) -> bytes:
    """Serialize a bootstrap into exactly 60 bytes."""

    if not isinstance(
        bootstrap,
        BootstrapHeader,
    ):
        raise ProtocolError(
            "bootstrap must be a BootstrapHeader."
        )

    if not 0 <= bootstrap.flags <= 255:
        raise ProtocolError(
            "Bootstrap flags must fit inside one byte."
        )

    if bootstrap.block_size not in {
        4,
        8,
        16,
        32,
    }:
        raise ProtocolError(
            "Unsupported bootstrap block size."
        )

    if (
        bootstrap.channel_mode
        != CHANNEL_MODE_KEYED_SINGLE
    ):
        raise ProtocolError(
            "Unsupported channel mode."
        )

    if bootstrap.image_width <= 0:
        raise ProtocolError(
            "Image width must be positive."
        )

    if bootstrap.image_height <= 0:
        raise ProtocolError(
            "Image height must be positive."
        )

    if len(bootstrap.salt) != SALT_SIZE:
        raise ProtocolError(
            f"Salt must contain {SALT_SIZE} bytes."
        )

    if len(bootstrap.nonce) != NONCE_SIZE:
        raise ProtocolError(
            f"Nonce must contain {NONCE_SIZE} bytes."
        )

    if bootstrap.metadata_bit_length <= 0:
        raise ProtocolError(
            "Metadata bit length must be positive."
        )

    if bootstrap.metadata_bit_length % 8 != 0:
        raise ProtocolError(
            "Metadata bit length must be byte-aligned."
        )

    if bootstrap.packet_bit_length <= 0:
        raise ProtocolError(
            "Packet bit length must be positive."
        )

    if bootstrap.packet_bit_length % 8 != 0:
        raise ProtocolError(
            "Packet bit length must be byte-aligned."
        )

    body = BOOTSTRAP_BODY.pack(
        BOOTSTRAP_MAGIC,
        BOOTSTRAP_VERSION,
        bootstrap.flags,
        bootstrap.block_size,
        bootstrap.channel_mode,
        bootstrap.image_width,
        bootstrap.image_height,
        bootstrap.salt,
        bootstrap.nonce,
        bootstrap.metadata_bit_length,
        bootstrap.packet_bit_length,
    )

    checksum = CRC_STRUCT.pack(
        _crc32(body)
    )

    serialized = body + checksum

    if len(serialized) != BOOTSTRAP_SIZE:
        raise ProtocolError(
            "Serialized bootstrap has an invalid size."
        )

    return serialized


def parse_bootstrap(
    data: bytes | bytearray,
) -> BootstrapHeader:
    """Parse and validate a 60-byte bootstrap."""

    if not isinstance(
        data,
        (bytes, bytearray),
    ):
        raise ProtocolError(
            "Bootstrap data must be bytes."
        )

    bootstrap_bytes = bytes(data)

    if len(bootstrap_bytes) != BOOTSTRAP_SIZE:
        raise ProtocolError(
            f"Bootstrap must contain exactly "
            f"{BOOTSTRAP_SIZE} bytes."
        )

    body = bootstrap_bytes[
        :BOOTSTRAP_BODY.size
    ]

    stored_crc = CRC_STRUCT.unpack(
        bootstrap_bytes[
            BOOTSTRAP_BODY.size:
        ]
    )[0]

    calculated_crc = _crc32(body)

    if stored_crc != calculated_crc:
        raise ProtocolError(
            "Bootstrap CRC32 verification failed."
        )

    (
        magic,
        version,
        flags,
        block_size,
        channel_mode,
        image_width,
        image_height,
        salt,
        nonce,
        metadata_bit_length,
        packet_bit_length,
    ) = BOOTSTRAP_BODY.unpack(body)

    if magic != BOOTSTRAP_MAGIC:
        raise ProtocolError(
            "Invalid bootstrap magic."
        )

    if version != BOOTSTRAP_VERSION:
        raise ProtocolError(
            f"Unsupported bootstrap version: {version}"
        )

    bootstrap = BootstrapHeader(
        flags=flags,
        block_size=block_size,
        channel_mode=channel_mode,
        image_width=image_width,
        image_height=image_height,
        salt=salt,
        nonce=nonce,
        metadata_bit_length=(
            metadata_bit_length
        ),
        packet_bit_length=(
            packet_bit_length
        ),
    )

    # Reuse serialization validation.
    serialize_bootstrap(bootstrap)

    return bootstrap


def serialize_metadata(
    metadata: LocationMetadata,
) -> bytes:
    """Serialize location metadata and append CRC32."""

    if not isinstance(
        metadata,
        LocationMetadata,
    ):
        raise ProtocolError(
            "metadata must be LocationMetadata."
        )

    if (
        metadata.encoding_type
        not in ENCODING_TO_CODE
    ):
        raise ProtocolError(
            "Unsupported block-map encoding."
        )

    if (
        metadata.embedding_policy
        != EMBEDDING_POLICY_ADAPTIVE_LSB
    ):
        raise ProtocolError(
            "Unsupported embedding policy."
        )

    if (
        metadata.channels_per_selected_pixel
        != 1
    ):
        raise ProtocolError(
            "The current protocol supports one "
            "channel per selected pixel."
        )

    if not 1 <= metadata.block_rows <= 65535:
        raise ProtocolError(
            "Block rows must fit inside two bytes."
        )

    if not 1 <= metadata.block_columns <= 65535:
        raise ProtocolError(
            "Block columns must fit inside two bytes."
        )

    expected_map_bits = (
        metadata.block_rows
        * metadata.block_columns
    )

    if (
        metadata.original_map_bit_count
        != expected_map_bits
    ):
        raise ProtocolError(
            "Original map bit count does not match "
            "block-map dimensions."
        )

    if not isinstance(
        metadata.encoded_map_data,
        bytes,
    ):
        raise ProtocolError(
            "Encoded map data must be bytes."
        )

    if (
        metadata.encoding_type
        in {"ALL_SELECTED", "NONE_SELECTED"}
        and metadata.encoded_map_data
    ):
        raise ProtocolError(
            "Special map encoding must not "
            "contain encoded bytes."
        )

    header = METADATA_HEADER.pack(
        METADATA_MAGIC,
        METADATA_VERSION,
        ENCODING_TO_CODE[
            metadata.encoding_type
        ],
        metadata.embedding_policy,
        metadata.channels_per_selected_pixel,
        metadata.block_rows,
        metadata.block_columns,
        metadata.original_map_bit_count,
        len(metadata.encoded_map_data),
    )

    body = (
        header
        + metadata.encoded_map_data
    )

    return (
        body
        + CRC_STRUCT.pack(
            _crc32(body)
        )
    )


def parse_metadata(
    data: bytes | bytearray,
) -> LocationMetadata:
    """Parse and validate location metadata."""

    if not isinstance(
        data,
        (bytes, bytearray),
    ):
        raise ProtocolError(
            "Metadata data must be bytes."
        )

    metadata_bytes = bytes(data)

    minimum_length = (
        METADATA_HEADER.size
        + CRC_STRUCT.size
    )

    if len(metadata_bytes) < minimum_length:
        raise ProtocolError(
            "Metadata is smaller than its header."
        )

    body = metadata_bytes[
        :-CRC_STRUCT.size
    ]

    stored_crc = CRC_STRUCT.unpack(
        metadata_bytes[
            -CRC_STRUCT.size:
        ]
    )[0]

    if stored_crc != _crc32(body):
        raise ProtocolError(
            "Metadata CRC32 verification failed."
        )

    (
        magic,
        version,
        encoding_code,
        embedding_policy,
        channels_per_selected_pixel,
        block_rows,
        block_columns,
        original_map_bit_count,
        encoded_map_byte_count,
    ) = METADATA_HEADER.unpack_from(
        body,
        0,
    )

    if magic != METADATA_MAGIC:
        raise ProtocolError(
            "Invalid metadata magic."
        )

    if version != METADATA_VERSION:
        raise ProtocolError(
            f"Unsupported metadata version: {version}"
        )

    if encoding_code not in CODE_TO_ENCODING:
        raise ProtocolError(
            "Unknown metadata encoding code."
        )

    expected_length = (
        METADATA_HEADER.size
        + encoded_map_byte_count
    )

    if len(body) != expected_length:
        raise ProtocolError(
            "Metadata length does not match its header."
        )

    encoded_map_data = body[
        METADATA_HEADER.size:
    ]

    metadata = LocationMetadata(
        encoding_type=(
            CODE_TO_ENCODING[
                encoding_code
            ]
        ),
        embedding_policy=(
            embedding_policy
        ),
        channels_per_selected_pixel=(
            channels_per_selected_pixel
        ),
        block_rows=block_rows,
        block_columns=block_columns,
        original_map_bit_count=(
            original_map_bit_count
        ),
        encoded_map_data=(
            encoded_map_data
        ),
    )

    # Reuse serializer validation.
    serialize_metadata(metadata)

    return metadata