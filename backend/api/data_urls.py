"""Safe Base64 data-URL handling for the Flask API."""

from __future__ import annotations

import base64
import binascii
import re


class DataUrlError(ValueError):
    """Raised when a data URL is invalid."""


DATA_URL_PATTERN = re.compile(
    r"^data:"
    r"(?P<mime>[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+)"
    r";base64,"
    r"(?P<data>[A-Za-z0-9+/=\r\n]+)$"
)


def decode_data_url(
    value: str,
    maximum_bytes: int,
) -> tuple[str, bytes]:
    """Decode and validate one Base64 data URL."""

    if not isinstance(value, str):
        raise DataUrlError(
            "Data URL must be a string."
        )

    if maximum_bytes <= 0:
        raise DataUrlError(
            "maximum_bytes must be positive."
        )

    match = DATA_URL_PATTERN.fullmatch(
        value.strip()
    )

    if match is None:
        raise DataUrlError(
            "Invalid Base64 data URL."
        )

    mime_type = match.group("mime")
    encoded_data = match.group("data")

    maximum_encoded_length = (
        (maximum_bytes * 4 // 3)
        + 8
    )

    if len(encoded_data) > maximum_encoded_length:
        raise DataUrlError(
            "Encoded data exceeds the size limit."
        )

    try:
        decoded = base64.b64decode(
            encoded_data,
            validate=True,
        )

    except (
        binascii.Error,
        ValueError,
    ) as error:
        raise DataUrlError(
            "Invalid Base64 data."
        ) from error

    if len(decoded) > maximum_bytes:
        raise DataUrlError(
            "Decoded data exceeds the size limit."
        )

    return mime_type, decoded


def encode_data_url(
    mime_type: str,
    data: bytes,
) -> str:
    """Encode bytes as a Base64 data URL."""

    if not isinstance(mime_type, str):
        raise DataUrlError(
            "MIME type must be a string."
        )

    if not isinstance(data, bytes):
        raise DataUrlError(
            "Data must be bytes."
        )

    encoded = base64.b64encode(
        data
    ).decode("ascii")

    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )