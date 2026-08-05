import pytest

from backend.crypto.kdf import (
    KEY_SIZE,
    SALT_SIZE,
    KeyDerivationError,
    derive_keys,
    generate_salt,
)


PASSPHRASE = "GuardianPixel-Test-Password"


def test_generated_salt_size():
    salt = generate_salt()

    assert isinstance(salt, bytes)
    assert len(salt) == SALT_SIZE


def test_salts_are_random():
    first = generate_salt()
    second = generate_salt()

    assert first != second


def test_derived_key_lengths():
    keys = derive_keys(
        PASSPHRASE,
        generate_salt(),
    )

    assert len(keys.encryption_key) == KEY_SIZE
    assert len(keys.position_key) == KEY_SIZE


def test_encryption_and_position_keys_differ():
    keys = derive_keys(
        PASSPHRASE,
        generate_salt(),
    )

    assert (
        keys.encryption_key
        != keys.position_key
    )


def test_same_inputs_produce_same_keys():
    salt = bytes(range(SALT_SIZE))

    first = derive_keys(
        PASSPHRASE,
        salt,
    )

    second = derive_keys(
        PASSPHRASE,
        salt,
    )

    assert (
        first.encryption_key
        == second.encryption_key
    )

    assert (
        first.position_key
        == second.position_key
    )


def test_different_salts_produce_different_keys():
    first = derive_keys(
        PASSPHRASE,
        b"\x01" * SALT_SIZE,
    )

    second = derive_keys(
        PASSPHRASE,
        b"\x02" * SALT_SIZE,
    )

    assert (
        first.encryption_key
        != second.encryption_key
    )


def test_short_passphrase_is_rejected():
    with pytest.raises(
        KeyDerivationError,
        match="at least 8",
    ):
        derive_keys(
            "short",
            generate_salt(),
        )


def test_invalid_salt_size_is_rejected():
    with pytest.raises(
        KeyDerivationError,
        match="16 bytes",
    ):
        derive_keys(
            PASSPHRASE,
            b"small",
        )