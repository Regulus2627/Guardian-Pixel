import pytest

from backend.crypto.encryption import (
    DEFAULT_AAD,
    EncryptedPayload,
    EncryptionError,
    decrypt_envelope,
    encrypt_envelope,
    recover_key_material,
)
from backend.crypto.envelope import (
    build_envelope,
    parse_envelope,
)


PASSPHRASE = "GuardianPixel-Strong-Test-Password"


def create_test_envelope() -> bytes:
    return build_envelope(
        payload=(
            b"GuardianPixel encrypted message"
        ),
        payload_type="text",
        filename="",
        mime_type="text/plain",
    )


def test_encryption_round_trip():
    original_envelope = (
        create_test_envelope()
    )

    encrypted = encrypt_envelope(
        original_envelope,
        PASSPHRASE,
    )

    recovered_envelope = decrypt_envelope(
        encrypted,
        PASSPHRASE,
    )

    assert (
        recovered_envelope
        == original_envelope
    )

    parsed = parse_envelope(
        recovered_envelope
    )

    assert parsed.payload == (
        b"GuardianPixel encrypted message"
    )


def test_same_plaintext_produces_different_ciphertext():
    envelope = create_test_envelope()

    first = encrypt_envelope(
        envelope,
        PASSPHRASE,
    )

    second = encrypt_envelope(
        envelope,
        PASSPHRASE,
    )

    assert first.salt != second.salt
    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext


def test_wrong_passphrase_is_rejected():
    encrypted = encrypt_envelope(
        create_test_envelope(),
        PASSPHRASE,
    )

    with pytest.raises(
        EncryptionError,
        match="Authentication failed",
    ):
        decrypt_envelope(
            encrypted,
            "Incorrect-Password-123",
        )


def test_modified_ciphertext_is_rejected():
    encrypted = encrypt_envelope(
        create_test_envelope(),
        PASSPHRASE,
    )

    modified_ciphertext = bytearray(
        encrypted.ciphertext
    )

    modified_ciphertext[0] ^= 0x01

    modified = EncryptedPayload(
        salt=encrypted.salt,
        nonce=encrypted.nonce,
        ciphertext=bytes(
            modified_ciphertext
        ),
    )

    with pytest.raises(
        EncryptionError,
        match="Authentication failed",
    ):
        decrypt_envelope(
            modified,
            PASSPHRASE,
        )


def test_modified_aad_is_rejected():
    encrypted = encrypt_envelope(
        create_test_envelope(),
        PASSPHRASE,
        aad=DEFAULT_AAD,
    )

    with pytest.raises(
        EncryptionError,
        match="Authentication failed",
    ):
        decrypt_envelope(
            encrypted,
            PASSPHRASE,
            aad=b"Different-AAD",
        )


def test_position_key_can_be_recovered():
    encrypted = encrypt_envelope(
        create_test_envelope(),
        PASSPHRASE,
    )

    first_keys = recover_key_material(
        PASSPHRASE,
        encrypted.salt,
    )

    second_keys = recover_key_material(
        PASSPHRASE,
        encrypted.salt,
    )

    assert (
        first_keys.position_key
        == second_keys.position_key
    )


def test_invalid_nonce_size_is_rejected():
    encrypted = EncryptedPayload(
        salt=b"\x01" * 16,
        nonce=b"short",
        ciphertext=b"A" * 32,
    )

    with pytest.raises(
        EncryptionError,
        match="Nonce",
    ):
        decrypt_envelope(
            encrypted,
            PASSPHRASE,
        )