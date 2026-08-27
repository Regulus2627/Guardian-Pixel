"""Payload characteristics for GuardianPixel compatibility analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.crypto.encryption import (
    GCM_TAG_SIZE,
)
from backend.crypto.envelope import (
    ENVELOPE_HEADER,
    build_envelope,
    parse_envelope,
)


class PayloadFeatureError(ValueError):
    """Raised when payload characteristics cannot be calculated."""


@dataclass(frozen=True, slots=True)
class PayloadFeatures:
    """Characteristics of a text payload before embedding."""

    profile: str
    character_count: int
    utf8_bytes: int

    compressed: bool
    stored_payload_bytes: int

    envelope_bytes: int
    expected_ciphertext_bytes: int
    expected_ciphertext_bits: int

    compression_ratio: float


ENGLISH_CORPUS = (
    "GuardianPixel securely hides encrypted information inside "
    "carefully selected textured regions of a digital image. "
    "The receiver uses the correct passphrase to recover the "
    "original information without running the neural network. "
)


def generate_text_payload(
    requested_bytes: int,
    profile: str = "english",
    seed: int = 42,
) -> bytes:
    """
    Generate deterministic UTF-8-compatible text of an exact byte size.

    Profiles:
        english: repeated normal English text;
        repetitive: repeated letter A;
        random_ascii: low-compressibility printable ASCII.
    """

    if not isinstance(requested_bytes, int):
        raise PayloadFeatureError(
            "requested_bytes must be an integer."
        )

    if requested_bytes < 0:
        raise PayloadFeatureError(
            "requested_bytes cannot be negative."
        )

    if profile == "repetitive":
        return b"A" * requested_bytes

    if profile == "english":
        corpus_bytes = ENGLISH_CORPUS.encode(
            "utf-8"
        )

        if requested_bytes == 0:
            return b""

        repetitions = (
            requested_bytes
            // len(corpus_bytes)
            + 1
        )

        return (
            corpus_bytes * repetitions
        )[:requested_bytes]

    if profile == "random_ascii":
        generator = np.random.default_rng(
            seed + requested_bytes
        )

        # Printable ASCII values from space (32) to tilde (126).
        values = generator.integers(
            32,
            127,
            size=requested_bytes,
            dtype=np.uint8,
        )

        return values.tobytes()

    raise PayloadFeatureError(
        "profile must be english, repetitive "
        "or random_ascii."
    )


def analyse_text_payload(
    payload: bytes,
    profile: str,
) -> tuple[PayloadFeatures, bytes]:
    """
    Build a text envelope and return its exact size characteristics.

    Returns:
        PayloadFeatures
        Serialized plaintext envelope
    """

    if not isinstance(payload, bytes):
        raise PayloadFeatureError(
            "payload must be bytes."
        )

    try:
        decoded_text = payload.decode(
            "utf-8"
        )
    except UnicodeDecodeError as error:
        raise PayloadFeatureError(
            "Text payload must contain valid UTF-8."
        ) from error

    envelope = build_envelope(
        payload=payload,
        payload_type="text",
        filename="",
        mime_type="text/plain",
        enable_compression=True,
    )

    (
        _magic,
        _version,
        _payload_code,
        _flags,
        _filename_length,
        _mime_length,
        original_length,
        stored_length,
        _digest,
    ) = ENVELOPE_HEADER.unpack_from(
        envelope,
        0,
    )

    parsed = parse_envelope(
        envelope
    )

    if original_length != len(payload):
        raise PayloadFeatureError(
            "Envelope original length is incorrect."
        )

    if len(payload) == 0:
        compression_ratio = 1.0
    else:
        compression_ratio = (
            stored_length
            / len(payload)
        )

    expected_ciphertext_bytes = (
        len(envelope)
        + GCM_TAG_SIZE
    )

    features = PayloadFeatures(
        profile=profile,
        character_count=len(
            decoded_text
        ),
        utf8_bytes=len(payload),
        compressed=parsed.compressed,
        stored_payload_bytes=(
            stored_length
        ),
        envelope_bytes=len(envelope),
        expected_ciphertext_bytes=(
            expected_ciphertext_bytes
        ),
        expected_ciphertext_bits=(
            expected_ciphertext_bytes
            * 8
        ),
        compression_ratio=float(
            compression_ratio
        ),
    )

    return features, envelope