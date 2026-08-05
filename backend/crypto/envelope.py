"""Structured plaintext envelope for GuardianPixel secrets."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from backend.crypto.hashing import (
    sha256_digest,
    verify_sha256,
)


class EnvelopeError(ValueError):
    """Raised when a GuardianPixel envelope is invalid."""


ENVELOPE_MAGIC = b"GPE1"
ENVELOPE_VERSION = 1

PAYLOAD_TYPE_TEXT = 1
PAYLOAD_TYPE_FILE = 2

FLAG_COMPRESSED = 0x01

MAX_FILENAME_BYTES = 255
MAX_MIME_BYTES = 127
DEFAULT_MAX_PAYLOAD_BYTES = 20 * 1024 * 1024

# Fields:
# magic, version, type, flags,
# filename length, MIME length,
# original payload length, stored payload length,
# SHA-256 digest
ENVELOPE_HEADER = struct.Struct(
    ">4sBBBHHQQ32s"
)


@dataclass(frozen=True, slots=True)
class ParsedEnvelope:
    """Recovered and verified secret envelope."""

    payload: bytes
    payload_type: str
    filename: str
    mime_type: str
    original_length: int
    sha256_hex: str
    compressed: bool


def _payload_type_to_code(
    payload_type: str,
) -> int:
    if payload_type == "text":
        return PAYLOAD_TYPE_TEXT

    if payload_type == "file":
        return PAYLOAD_TYPE_FILE

    raise EnvelopeError(
        "payload_type must be text or file."
    )


def _payload_code_to_type(
    payload_code: int,
) -> str:
    if payload_code == PAYLOAD_TYPE_TEXT:
        return "text"

    if payload_code == PAYLOAD_TYPE_FILE:
        return "file"

    raise EnvelopeError(
        f"Unsupported payload type code: {payload_code}"
    )


def _validate_filename(
    filename: str,
) -> bytes:
    if not isinstance(filename, str):
        raise EnvelopeError(
            "filename must be a string."
        )

    if "\x00" in filename:
        raise EnvelopeError(
            "filename cannot contain null characters."
        )

    if filename and Path(filename).name != filename:
        raise EnvelopeError(
            "filename must not contain directory paths."
        )

    encoded = filename.encode("utf-8")

    if len(encoded) > MAX_FILENAME_BYTES:
        raise EnvelopeError(
            "filename is too long."
        )

    return encoded


def _validate_mime_type(
    mime_type: str,
) -> bytes:
    if not isinstance(mime_type, str):
        raise EnvelopeError(
            "mime_type must be a string."
        )

    if "\x00" in mime_type:
        raise EnvelopeError(
            "mime_type cannot contain null characters."
        )

    encoded = mime_type.encode("utf-8")

    if len(encoded) > MAX_MIME_BYTES:
        raise EnvelopeError(
            "mime_type is too long."
        )

    return encoded


def build_envelope(
    payload: bytes | bytearray,
    payload_type: str,
    filename: str = "",
    mime_type: str = "application/octet-stream",
    enable_compression: bool = True,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> bytes:
    """Create a structured envelope before encryption."""

    if not isinstance(
        payload,
        (bytes, bytearray),
    ):
        raise EnvelopeError(
            "payload must be bytes or bytearray."
        )

    payload_bytes = bytes(payload)

    if max_payload_bytes <= 0:
        raise EnvelopeError(
            "max_payload_bytes must be positive."
        )

    if len(payload_bytes) > max_payload_bytes:
        raise EnvelopeError(
            "payload exceeds the configured size limit."
        )

    payload_code = _payload_type_to_code(
        payload_type
    )

    if payload_type == "text" and filename:
        raise EnvelopeError(
            "Text payloads must not include a filename."
        )

    if payload_type == "file" and not filename:
        raise EnvelopeError(
            "File payloads require a filename."
        )

    filename_bytes = _validate_filename(
        filename
    )

    mime_bytes = _validate_mime_type(
        mime_type
    )

    original_digest = sha256_digest(
        payload_bytes
    )

    stored_payload = payload_bytes
    flags = 0

    if enable_compression and payload_bytes:
        compressed_payload = zlib.compress(
            payload_bytes,
            level=9,
        )

        # Compression is used only when it genuinely saves space.
        if len(compressed_payload) < len(
            payload_bytes
        ):
            stored_payload = compressed_payload
            flags |= FLAG_COMPRESSED

    header = ENVELOPE_HEADER.pack(
        ENVELOPE_MAGIC,
        ENVELOPE_VERSION,
        payload_code,
        flags,
        len(filename_bytes),
        len(mime_bytes),
        len(payload_bytes),
        len(stored_payload),
        original_digest,
    )

    return (
        header
        + filename_bytes
        + mime_bytes
        + stored_payload
    )


def parse_envelope(
    envelope: bytes | bytearray,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> ParsedEnvelope:
    """Parse, decompress and verify a GuardianPixel envelope."""

    if not isinstance(
        envelope,
        (bytes, bytearray),
    ):
        raise EnvelopeError(
            "envelope must be bytes or bytearray."
        )

    envelope_bytes = bytes(envelope)

    if len(envelope_bytes) < ENVELOPE_HEADER.size:
        raise EnvelopeError(
            "Envelope is smaller than its header."
        )

    if max_payload_bytes <= 0:
        raise EnvelopeError(
            "max_payload_bytes must be positive."
        )

    (
        magic,
        version,
        payload_code,
        flags,
        filename_length,
        mime_length,
        original_length,
        stored_length,
        expected_digest,
    ) = ENVELOPE_HEADER.unpack_from(
        envelope_bytes,
        0,
    )

    if magic != ENVELOPE_MAGIC:
        raise EnvelopeError(
            "Invalid envelope magic."
        )

    if version != ENVELOPE_VERSION:
        raise EnvelopeError(
            f"Unsupported envelope version: {version}"
        )

    if flags & ~FLAG_COMPRESSED:
        raise EnvelopeError(
            "Envelope contains unsupported flags."
        )

    payload_type = _payload_code_to_type(
        payload_code
    )

    if original_length > max_payload_bytes:
        raise EnvelopeError(
            "Original payload exceeds the size limit."
        )

    expected_total_length = (
        ENVELOPE_HEADER.size
        + filename_length
        + mime_length
        + stored_length
    )

    if len(envelope_bytes) != expected_total_length:
        raise EnvelopeError(
            "Envelope length does not match its header."
        )

    offset = ENVELOPE_HEADER.size

    filename_bytes = envelope_bytes[
        offset:offset + filename_length
    ]

    offset += filename_length

    mime_bytes = envelope_bytes[
        offset:offset + mime_length
    ]

    offset += mime_length

    stored_payload = envelope_bytes[
        offset:offset + stored_length
    ]

    try:
        filename = filename_bytes.decode(
            "utf-8"
        )

        mime_type = mime_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError as error:
        raise EnvelopeError(
            "Filename or MIME type is not valid UTF-8."
        ) from error

    _validate_filename(filename)
    _validate_mime_type(mime_type)

    if payload_type == "text" and filename:
        raise EnvelopeError(
            "Text envelope must not contain a filename."
        )

    if payload_type == "file" and not filename:
        raise EnvelopeError(
            "File envelope requires a filename."
        )

    compressed = bool(
        flags & FLAG_COMPRESSED
    )

    if compressed:
        try:
            decompressor = zlib.decompressobj()

            payload = decompressor.decompress(
                stored_payload,
                original_length + 1,
            )

            payload += decompressor.flush()

        except zlib.error as error:
            raise EnvelopeError(
                f"Invalid compressed payload: {error}"
            ) from error

        if decompressor.unused_data:
            raise EnvelopeError(
                "Compressed payload contains trailing data."
            )

    else:
        payload = stored_payload

    if len(payload) != original_length:
        raise EnvelopeError(
            "Recovered payload length is incorrect."
        )

    if len(payload) > max_payload_bytes:
        raise EnvelopeError(
            "Recovered payload exceeds the size limit."
        )

    if not verify_sha256(
        payload,
        expected_digest,
    ):
        raise EnvelopeError(
            "SHA-256 payload verification failed."
        )

    return ParsedEnvelope(
        payload=payload,
        payload_type=payload_type,
        filename=filename,
        mime_type=mime_type,
        original_length=original_length,
        sha256_hex=expected_digest.hex(),
        compressed=compressed,
    )