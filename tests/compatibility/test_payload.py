import pytest

from backend.compatibility.payload import (
    PayloadFeatureError,
    analyse_text_payload,
    generate_text_payload,
)


@pytest.mark.parametrize(
    "size",
    [
        0,
        1,
        50,
        100,
        1000,
        5000,
    ],
)
def test_generated_payload_has_exact_size(size):
    payload = generate_text_payload(
        requested_bytes=size,
        profile="english",
    )

    assert len(payload) == size

    # All generated text profiles must be valid UTF-8.
    payload.decode("utf-8")


@pytest.mark.parametrize(
    "profile",
    [
        "english",
        "repetitive",
        "random_ascii",
    ],
)
def test_all_profiles_generate_valid_text(
    profile,
):
    payload = generate_text_payload(
        requested_bytes=1000,
        profile=profile,
    )

    assert len(payload) == 1000
    payload.decode("utf-8")


def test_payload_generation_is_deterministic():
    first = generate_text_payload(
        requested_bytes=1000,
        profile="random_ascii",
        seed=42,
    )

    second = generate_text_payload(
        requested_bytes=1000,
        profile="random_ascii",
        seed=42,
    )

    assert first == second


def test_payload_analysis():
    payload = generate_text_payload(
        requested_bytes=500,
        profile="english",
    )

    features, envelope = (
        analyse_text_payload(
            payload=payload,
            profile="english",
        )
    )

    assert features.utf8_bytes == 500
    assert features.character_count == 500
    assert features.envelope_bytes == len(
        envelope
    )

    assert (
        features.expected_ciphertext_bytes
        == len(envelope) + 16
    )

    assert (
        features.expected_ciphertext_bits
        == features.expected_ciphertext_bytes
        * 8
    )


def test_repetitive_text_compresses():
    payload = generate_text_payload(
        requested_bytes=10_000,
        profile="repetitive",
    )

    features, _ = analyse_text_payload(
        payload=payload,
        profile="repetitive",
    )

    assert features.compressed is True
    assert features.compression_ratio < 1


def test_random_ascii_is_larger_than_repetitive():
    repetitive = generate_text_payload(
        requested_bytes=10_000,
        profile="repetitive",
    )

    random_ascii = generate_text_payload(
        requested_bytes=10_000,
        profile="random_ascii",
    )

    repetitive_features, _ = (
        analyse_text_payload(
            repetitive,
            profile="repetitive",
        )
    )

    random_features, _ = (
        analyse_text_payload(
            random_ascii,
            profile="random_ascii",
        )
    )

    assert (
        random_features
        .stored_payload_bytes
        >
        repetitive_features
        .stored_payload_bytes
    )


def test_invalid_profile_is_rejected():
    with pytest.raises(
        PayloadFeatureError,
        match="profile must be",
    ):
        generate_text_payload(
            requested_bytes=100,
            profile="unknown",
        )


def test_negative_size_is_rejected():
    with pytest.raises(
        PayloadFeatureError,
        match="cannot be negative",
    ):
        generate_text_payload(
            requested_bytes=-1,
            profile="english",
        )


def test_invalid_utf8_is_rejected():
    with pytest.raises(
        PayloadFeatureError,
        match="valid UTF-8",
    ):
        analyse_text_payload(
            payload=b"\xff\xfe",
            profile="invalid",
        )