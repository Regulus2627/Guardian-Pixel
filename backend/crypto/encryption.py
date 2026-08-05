"""AES-256-GCM authenticated encryption for GuardianPixel."""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import (
    AESGCM,
)

from backend.crypto.kdf import (
    KeyMaterial,
    SALT_SIZE,
    derive_keys,
    generate_salt,
)


class EncryptionError(ValueError):
    """Raised when encryption or authenticated decryption fails."""


NONCE_SIZE = 12
GCM_TAG_SIZE = 16

DEFAULT_AAD = b"GuardianPixel/GPX1"


@dataclass(frozen=True, slots=True)
class EncryptedPayload:
    """Values required to store and later decrypt ciphertext."""

    salt: bytes
    nonce: bytes
    ciphertext: bytes

    @property
    def ciphertext_length(self) -> int:
        return len(self.ciphertext)


def generate_nonce() -> bytes:
    """Generate a random 96-bit AES-GCM nonce."""

    return os.urandom(NONCE_SIZE)


def _validate_bytes(
    value,
    name: str,
) -> bytes:
    if not isinstance(
        value,
        (bytes, bytearray),
    ):
        raise EncryptionError(
            f"{name} must be bytes or bytearray."
        )

    return bytes(value)


def encrypt_envelope(
    envelope: bytes | bytearray,
    passphrase: str,
    aad: bytes | bytearray = DEFAULT_AAD,
) -> EncryptedPayload:
    """Encrypt an envelope using a new salt and nonce."""

    envelope_bytes = _validate_bytes(
        envelope,
        "Envelope",
    )

    aad_bytes = _validate_bytes(
        aad,
        "AAD",
    )

    salt = generate_salt()
    nonce = generate_nonce()

    keys = derive_keys(
        passphrase,
        salt,
    )

    aes_gcm = AESGCM(
        keys.encryption_key
    )

    ciphertext = aes_gcm.encrypt(
        nonce,
        envelope_bytes,
        aad_bytes,
    )

    return EncryptedPayload(
        salt=salt,
        nonce=nonce,
        ciphertext=ciphertext,
    )


def decrypt_envelope(
    encrypted_payload: EncryptedPayload,
    passphrase: str,
    aad: bytes | bytearray = DEFAULT_AAD,
) -> bytes:
    """Authenticate and decrypt an encrypted envelope."""

    if not isinstance(
        encrypted_payload,
        EncryptedPayload,
    ):
        raise EncryptionError(
            "encrypted_payload must be an EncryptedPayload."
        )

    aad_bytes = _validate_bytes(
        aad,
        "AAD",
    )

    if len(encrypted_payload.salt) != SALT_SIZE:
        raise EncryptionError(
            f"Salt must contain {SALT_SIZE} bytes."
        )

    if len(encrypted_payload.nonce) != NONCE_SIZE:
        raise EncryptionError(
            f"Nonce must contain {NONCE_SIZE} bytes."
        )

    if len(
        encrypted_payload.ciphertext
    ) < GCM_TAG_SIZE:
        raise EncryptionError(
            "Ciphertext is too short to contain a GCM tag."
        )

    keys = derive_keys(
        passphrase,
        encrypted_payload.salt,
    )

    aes_gcm = AESGCM(
        keys.encryption_key
    )

    try:
        return aes_gcm.decrypt(
            encrypted_payload.nonce,
            encrypted_payload.ciphertext,
            aad_bytes,
        )

    except InvalidTag as error:
        raise EncryptionError(
            "Authentication failed: wrong passphrase "
            "or modified encrypted data."
        ) from error


def recover_key_material(
    passphrase: str,
    salt: bytes,
) -> KeyMaterial:
    """
    Recreate both keys during extraction.

    The position key will later regenerate metadata and payload
    locations.
    """

    return derive_keys(
        passphrase,
        salt,
    )