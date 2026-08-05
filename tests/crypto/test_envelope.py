import os

import pytest

from backend.crypto.envelope import (
    EnvelopeError,
    build_envelope,
    parse_envelope,
)
from backend.crypto.hashing import (
    sha256_digest,
    sha256_hex,
    verify_sha256,
)


def test_sha256_known_value():
    digest = sha256_hex(b"abc")

    assert digest == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_verification():
    payload = b"GuardianPixel"

    digest = sha256_digest(payload)

    assert verify_sha256(
        payload,
        digest,
    )

    assert not verify_sha256(
        b"Changed",
        digest,
    )


def test_text_envelope_round_trip():
    original = (
        "GuardianPixel secret message"
    ).encode("utf-8")

    envelope = build_envelope(
        payload=original,
        payload_type="text",
        filename="",
        mime_type="text/plain",
    )

    parsed = parse_envelope(envelope)

    assert parsed.payload == original
    assert parsed.payload_type == "text"
    assert parsed.filename == ""
    assert parsed.mime_type == "text/plain"


def test_file_envelope_round_trip():
    original = bytes(
        range(256)
    )

    envelope = build_envelope(
        payload=original,
        payload_type="file",
        filename="sample.bin",
        mime_type="application/octet-stream",
    )

    parsed = parse_envelope(envelope)

    assert parsed.payload == original
    assert parsed.payload_type == "file"
    assert parsed.filename == "sample.bin"
    assert (
        parsed.mime_type
        == "application/octet-stream"
    )


def test_repetitive_payload_is_compressed():
    original = b"A" * 10_000

    envelope = build_envelope(
        payload=original,
        payload_type="file",
        filename="repetitive.txt",
        mime_type="text/plain",
    )

    parsed = parse_envelope(envelope)

    assert parsed.payload == original
    assert parsed.compressed is True
    assert len(envelope) < len(original)


def test_random_payload_is_recovered():
    original = os.urandom(4096)

    envelope = build_envelope(
        payload=original,
        payload_type="file",
        filename="random.bin",
    )

    parsed = parse_envelope(envelope)

    assert parsed.payload == original


def test_unicode_filename_round_trip():
    original = b"example"

    envelope = build_envelope(
        payload=original,
        payload_type="file",
        filename="प्रोजेक्ट.txt",
        mime_type="text/plain",
    )

    parsed = parse_envelope(envelope)

    assert parsed.filename == "प्रोजेक्ट.txt"
    assert parsed.payload == original


def test_file_requires_filename():
    with pytest.raises(
        EnvelopeError,
        match="require a filename",
    ):
        build_envelope(
            payload=b"data",
            payload_type="file",
            filename="",
        )


def test_directory_filename_is_rejected():
    with pytest.raises(
        EnvelopeError,
        match="directory paths",
    ):
        build_envelope(
            payload=b"data",
            payload_type="file",
            filename="../secret.txt",
        )


def test_invalid_magic_is_rejected():
    envelope = bytearray(
        build_envelope(
            payload=b"message",
            payload_type="text",
            mime_type="text/plain",
        )
    )

    envelope[0:4] = b"BAD!"

    with pytest.raises(
        EnvelopeError,
        match="magic",
    ):
        parse_envelope(envelope)


def test_modified_payload_fails_hash():
    envelope = bytearray(
        build_envelope(
            payload=os.urandom(1024),
            payload_type="file",
            filename="sample.bin",
            enable_compression=False,
        )
    )

    envelope[-1] ^= 0x01

    with pytest.raises(
        EnvelopeError,
        match="SHA-256",
    ):
        parse_envelope(envelope)


def test_payload_size_limit():
    with pytest.raises(
        EnvelopeError,
        match="size limit",
    ):
        build_envelope(
            payload=b"A" * 100,
            payload_type="text",
            max_payload_bytes=50,
        )