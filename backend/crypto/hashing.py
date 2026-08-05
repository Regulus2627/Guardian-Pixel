"""SHA-256 hashing utilities for GuardianPixel."""

from __future__ import annotations

import hashlib
import hmac


class HashingError(ValueError):
    """Raised when hashing input is invalid."""


def sha256_digest(
    data: bytes | bytearray,
) -> bytes:
    """Return the raw 32-byte SHA-256 digest."""

    if not isinstance(
        data,
        (bytes, bytearray),
    ):
        raise HashingError(
            "Hashing input must be bytes or bytearray."
        )

    return hashlib.sha256(
        bytes(data)
    ).digest()


def sha256_hex(
    data: bytes | bytearray,
) -> str:
    """Return a hexadecimal SHA-256 digest."""

    return sha256_digest(data).hex()


def verify_sha256(
    data: bytes | bytearray,
    expected_digest: bytes,
) -> bool:
    """Compare a payload with an expected SHA-256 digest."""

    if not isinstance(
        expected_digest,
        bytes,
    ):
        raise HashingError(
            "Expected digest must be bytes."
        )

    if len(expected_digest) != 32:
        raise HashingError(
            "SHA-256 digest must contain 32 bytes."
        )

    calculated_digest = sha256_digest(data)

    return hmac.compare_digest(
        calculated_digest,
        expected_digest,
    )