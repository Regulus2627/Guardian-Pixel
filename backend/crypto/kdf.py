"""Passphrase-based key derivation for GuardianPixel."""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import (
    hashes,
)
from cryptography.hazmat.primitives.kdf.hkdf import (
    HKDF,
)
from cryptography.hazmat.primitives.kdf.scrypt import (
    Scrypt,
)


class KeyDerivationError(ValueError):
    """Raised when GuardianPixel keys cannot be derived."""


SALT_SIZE = 16
KEY_SIZE = 32

SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1

HKDF_OUTPUT_SIZE = 64
HKDF_INFO = b"GuardianPixel-GPX1-key-separation"


@dataclass(frozen=True, slots=True)
class KeyMaterial:
    """Separate keys derived from one passphrase."""

    encryption_key: bytes
    position_key: bytes


def generate_salt() -> bytes:
    """Generate a cryptographically secure random salt."""

    return os.urandom(SALT_SIZE)


def _validate_passphrase(
    passphrase: str,
) -> bytes:
    if not isinstance(passphrase, str):
        raise KeyDerivationError(
            "Passphrase must be a string."
        )

    if len(passphrase) < 8:
        raise KeyDerivationError(
            "Passphrase must contain at least 8 characters."
        )

    encoded = passphrase.encode("utf-8")

    if len(encoded) > 1024:
        raise KeyDerivationError(
            "Passphrase is too long."
        )

    return encoded


def derive_keys(
    passphrase: str,
    salt: bytes,
) -> KeyMaterial:
    """
    Derive independent encryption and position keys.

    Scrypt slows password guessing. HKDF separates the resulting
    master secret into keys for different purposes.
    """

    passphrase_bytes = _validate_passphrase(
        passphrase
    )

    if not isinstance(salt, bytes):
        raise KeyDerivationError(
            "Salt must be bytes."
        )

    if len(salt) != SALT_SIZE:
        raise KeyDerivationError(
            f"Salt must contain {SALT_SIZE} bytes."
        )

    scrypt = Scrypt(
        salt=salt,
        length=KEY_SIZE,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
    )

    master_key = scrypt.derive(
        passphrase_bytes
    )

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=HKDF_OUTPUT_SIZE,
        salt=salt,
        info=HKDF_INFO,
    )

    separated_keys = hkdf.derive(
        master_key
    )

    encryption_key = separated_keys[
        :KEY_SIZE
    ]

    position_key = separated_keys[
        KEY_SIZE:
    ]

    return KeyMaterial(
        encryption_key=encryption_key,
        position_key=position_key,
    )