import pytest

from backend.core.protocol import (
    BOOTSTRAP_SIZE,
    CHANNEL_MODE_KEYED_SINGLE,
    EMBEDDING_POLICY_ADAPTIVE_LSB,
    BootstrapHeader,
    LocationMetadata,
    ProtocolError,
    parse_bootstrap,
    parse_metadata,
    serialize_bootstrap,
    serialize_metadata,
)


def create_bootstrap() -> BootstrapHeader:
    return BootstrapHeader(
        flags=0,
        block_size=8,
        channel_mode=(
            CHANNEL_MODE_KEYED_SINGLE
        ),
        image_width=800,
        image_height=533,
        salt=b"\x01" * 16,
        nonce=b"\x02" * 12,
        metadata_bit_length=4000,
        packet_bit_length=50000,
    )


def test_bootstrap_is_exactly_60_bytes():
    serialized = serialize_bootstrap(
        create_bootstrap()
    )

    assert len(serialized) == BOOTSTRAP_SIZE
    assert BOOTSTRAP_SIZE == 60


def test_bootstrap_round_trip():
    original = create_bootstrap()

    serialized = serialize_bootstrap(
        original
    )

    recovered = parse_bootstrap(
        serialized
    )

    assert recovered == original


def test_bootstrap_crc_detects_changes():
    serialized = bytearray(
        serialize_bootstrap(
            create_bootstrap()
        )
    )

    serialized[10] ^= 0x01

    with pytest.raises(
        ProtocolError,
        match="CRC32",
    ):
        parse_bootstrap(serialized)


def test_invalid_bootstrap_size():
    with pytest.raises(
        ProtocolError,
        match="exactly",
    ):
        parse_bootstrap(b"too small")


def test_invalid_salt_size():
    invalid = BootstrapHeader(
        flags=0,
        block_size=8,
        channel_mode=(
            CHANNEL_MODE_KEYED_SINGLE
        ),
        image_width=800,
        image_height=533,
        salt=b"short",
        nonce=b"\x02" * 12,
        metadata_bit_length=4000,
        packet_bit_length=50000,
    )

    with pytest.raises(
        ProtocolError,
        match="Salt",
    ):
        serialize_bootstrap(invalid)


def create_metadata() -> LocationMetadata:
    return LocationMetadata(
        encoding_type="ZLIB",
        embedding_policy=(
            EMBEDDING_POLICY_ADAPTIVE_LSB
        ),
        channels_per_selected_pixel=1,
        block_rows=67,
        block_columns=100,
        original_map_bit_count=6700,
        encoded_map_data=b"compressed-map-data",
    )


def test_metadata_round_trip():
    original = create_metadata()

    serialized = serialize_metadata(
        original
    )

    recovered = parse_metadata(
        serialized
    )

    assert recovered == original


def test_metadata_crc_detects_changes():
    serialized = bytearray(
        serialize_metadata(
            create_metadata()
        )
    )

    serialized[-5] ^= 0x01

    with pytest.raises(
        ProtocolError,
        match="CRC32",
    ):
        parse_metadata(serialized)


def test_metadata_map_size_must_match_dimensions():
    invalid = LocationMetadata(
        encoding_type="RAW",
        embedding_policy=(
            EMBEDDING_POLICY_ADAPTIVE_LSB
        ),
        channels_per_selected_pixel=1,
        block_rows=10,
        block_columns=10,
        original_map_bit_count=99,
        encoded_map_data=b"map",
    )

    with pytest.raises(
        ProtocolError,
        match="does not match",
    ):
        serialize_metadata(invalid)


def test_special_encoding_has_no_data():
    invalid = LocationMetadata(
        encoding_type="ALL_SELECTED",
        embedding_policy=(
            EMBEDDING_POLICY_ADAPTIVE_LSB
        ),
        channels_per_selected_pixel=1,
        block_rows=10,
        block_columns=10,
        original_map_bit_count=100,
        encoded_map_data=b"unexpected",
    )

    with pytest.raises(
        ProtocolError,
        match="must not contain",
    ):
        serialize_metadata(invalid)