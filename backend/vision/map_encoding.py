"""Binary block-map encoding and decoding."""

from __future__ import annotations

import zlib

import numpy as np

from backend.vision.schemas import (
    MapEncodingResult,
)


class MapEncodingError(ValueError):
    """Raised when a binary map cannot be encoded or decoded."""


def _encode_unsigned_varint(
    value: int,
) -> bytes:
    """Encode a non-negative integer using unsigned LEB128."""

    if value < 0:
        raise MapEncodingError(
            "Varint value cannot be negative."
        )

    encoded = bytearray()

    while True:
        current_byte = value & 0x7F
        value >>= 7

        if value:
            encoded.append(
                current_byte | 0x80
            )
        else:
            encoded.append(current_byte)
            break

    return bytes(encoded)


def _decode_unsigned_varint(
    data: bytes,
    offset: int,
) -> tuple[int, int]:
    """Decode one unsigned LEB128 integer."""

    value = 0
    shift = 0

    while offset < len(data):
        current_byte = data[offset]
        offset += 1

        value |= (
            current_byte & 0x7F
        ) << shift

        if not (
            current_byte & 0x80
        ):
            return value, offset

        shift += 7

        if shift > 63:
            raise MapEncodingError(
                "Varint is too large."
            )

    raise MapEncodingError(
        "Unexpected end of varint data."
    )


def _pack_raw(
    flat_map: np.ndarray,
) -> bytes:
    packed = np.packbits(
        flat_map.astype(np.uint8),
        bitorder="big",
    )

    return packed.tobytes()


def _encode_rle(
    flat_map: np.ndarray,
) -> bytes:
    """Encode Boolean runs using first bit plus varint lengths."""

    if flat_map.size == 0:
        return b""

    encoded = bytearray()

    current_value = int(flat_map[0])
    encoded.append(current_value)

    run_length = 1

    for value in flat_map[1:]:
        integer_value = int(value)

        if integer_value == current_value:
            run_length += 1
        else:
            encoded.extend(
                _encode_unsigned_varint(
                    run_length
                )
            )

            current_value = integer_value
            run_length = 1

    encoded.extend(
        _encode_unsigned_varint(run_length)
    )

    return bytes(encoded)


def _decode_rle(
    data: bytes,
    bit_count: int,
) -> np.ndarray:
    if bit_count == 0:
        return np.empty(
            0,
            dtype=bool,
        )

    if len(data) < 2:
        raise MapEncodingError(
            "RLE data is incomplete."
        )

    current_value = data[0]

    if current_value not in {0, 1}:
        raise MapEncodingError(
            "RLE first value must be 0 or 1."
        )

    offset = 1
    decoded = []

    while offset < len(data):
        run_length, offset = (
            _decode_unsigned_varint(
                data,
                offset,
            )
        )

        if run_length <= 0:
            raise MapEncodingError(
                "RLE run length must be positive."
            )

        decoded.extend(
            [bool(current_value)]
            * run_length
        )

        if len(decoded) > bit_count:
            raise MapEncodingError(
                "RLE data exceeds expected bit count."
            )

        current_value = 1 - current_value

    if len(decoded) != bit_count:
        raise MapEncodingError(
            "RLE data does not match expected bit count."
        )

    return np.asarray(
        decoded,
        dtype=bool,
    )


def encode_block_map(
    selected_blocks: np.ndarray,
) -> MapEncodingResult:
    """Encode a 2D Boolean block map using its smallest format."""

    if not isinstance(
        selected_blocks,
        np.ndarray,
    ):
        raise MapEncodingError(
            "Block map must be a NumPy array."
        )

    if selected_blocks.ndim != 2:
        raise MapEncodingError(
            "Block map must have shape rows×columns."
        )

    if selected_blocks.size == 0:
        raise MapEncodingError(
            "Block map cannot be empty."
        )

    flat_map = selected_blocks.astype(
        bool,
        copy=False,
    ).ravel(order="C")

    bit_count = int(flat_map.size)
    raw_data = _pack_raw(flat_map)

    if np.all(flat_map):
        return MapEncodingResult(
            encoding_type="ALL_SELECTED",
            encoded_data=b"",
            original_bit_count=bit_count,
            raw_byte_count=len(raw_data),
            encoded_byte_count=0,
        )

    if not np.any(flat_map):
        return MapEncodingResult(
            encoding_type="NONE_SELECTED",
            encoded_data=b"",
            original_bit_count=bit_count,
            raw_byte_count=len(raw_data),
            encoded_byte_count=0,
        )

    rle_data = _encode_rle(flat_map)

    zlib_data = zlib.compress(
        raw_data,
        level=9,
    )

    candidates = {
        "RAW": raw_data,
        "RLE": rle_data,
        "ZLIB": zlib_data,
    }

    encoding_type, encoded_data = min(
        candidates.items(),
        key=lambda item: len(item[1]),
    )

    return MapEncodingResult(
        encoding_type=encoding_type,
        encoded_data=encoded_data,
        original_bit_count=bit_count,
        raw_byte_count=len(raw_data),
        encoded_byte_count=len(
            encoded_data
        ),
    )


def decode_block_map(
    encoding_type: str,
    encoded_data: bytes,
    block_rows: int,
    block_columns: int,
) -> np.ndarray:
    """Decode a previously encoded block map."""

    if block_rows <= 0 or block_columns <= 0:
        raise MapEncodingError(
            "Block-map dimensions must be positive."
        )

    bit_count = (
        block_rows * block_columns
    )

    expected_raw_bytes = (
        bit_count + 7
    ) // 8

    if encoding_type == "ALL_SELECTED":
        flat_map = np.ones(
            bit_count,
            dtype=bool,
        )

    elif encoding_type == "NONE_SELECTED":
        flat_map = np.zeros(
            bit_count,
            dtype=bool,
        )

    elif encoding_type == "RAW":
        if len(encoded_data) != expected_raw_bytes:
            raise MapEncodingError(
                "RAW data has an incorrect length."
            )

        unpacked = np.unpackbits(
            np.frombuffer(
                encoded_data,
                dtype=np.uint8,
            ),
            bitorder="big",
        )

        flat_map = unpacked[
            :bit_count
        ].astype(bool)

    elif encoding_type == "ZLIB":
        try:
            raw_data = zlib.decompress(
                encoded_data
            )
        except zlib.error as error:
            raise MapEncodingError(
                f"Invalid zlib block map: {error}"
            ) from error

        if len(raw_data) != expected_raw_bytes:
            raise MapEncodingError(
                "Decoded zlib map has "
                "an incorrect length."
            )

        unpacked = np.unpackbits(
            np.frombuffer(
                raw_data,
                dtype=np.uint8,
            ),
            bitorder="big",
        )

        flat_map = unpacked[
            :bit_count
        ].astype(bool)

    elif encoding_type == "RLE":
        flat_map = _decode_rle(
            encoded_data,
            bit_count,
        )

    else:
        raise MapEncodingError(
            f"Unsupported map encoding: "
            f"{encoding_type}"
        )

    return flat_map.reshape(
        block_rows,
        block_columns,
        order="C",
    )