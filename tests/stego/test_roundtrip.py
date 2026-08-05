import io

import numpy as np
import pytest
from PIL import Image


from backend.stego.embedder import (
    embed_secret,
)
from backend.stego.extractor import (
    ExtractorError,
    extract_secret,
)
from backend.vision.config import (
    load_vision_config,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)


class FakeHEDPredictor:
    """Fast deterministic HED replacement for round-trip tests."""

    def __init__(self):
        self.last_tile_count = 1

    def predict(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:
        luminance = np.mean(
            rgb.astype(np.float32),
            axis=2,
        )

        minimum = float(
            luminance.min()
        )

        maximum = float(
            luminance.max()
        )

        value_range = maximum - minimum

        if value_range <= 0:
            return np.zeros(
                rgb.shape[:2],
                dtype=np.float32,
            )

        return (
            (
                luminance - minimum
            )
            / value_range
        ).astype(np.float32)


PASSPHRASE = (
    "GuardianPixel-RoundTrip-Password"
)


def create_cover(
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    generator = np.random.default_rng(
        42
    )

    return generator.integers(
        0,
        256,
        size=(height, width, 3),
        dtype=np.uint8,
    )


def create_service():
    return GuardianPixelVisionService(
        config=load_vision_config(),
        hed_predictor=FakeHEDPredictor(),
    )


def test_text_round_trip():
    cover = create_cover()

    original = (
        "GuardianPixel exact secret text"
    ).encode("utf-8")

    embedded = embed_secret(
        rgb=cover,
        payload=original,
        payload_type="text",
        passphrase=PASSPHRASE,
        vision_service=create_service(),
        mime_type="text/plain",
    )

    extracted = extract_secret(
        embedded.stego_image,
        PASSPHRASE,
    )

    assert (
        extracted.parsed_envelope.payload
        == original
    )

    assert (
        extracted
        .parsed_envelope
        .payload_type
        == "text"
    )


def test_binary_file_round_trip():
    cover = create_cover(
        width=384,
        height=384,
    )

    original = bytes(
        range(256)
    ) * 4

    embedded = embed_secret(
        rgb=cover,
        payload=original,
        payload_type="file",
        filename="sample.bin",
        mime_type=(
            "application/octet-stream"
        ),
        passphrase=PASSPHRASE,
        vision_service=create_service(),
    )

    extracted = extract_secret(
        embedded.stego_image,
        PASSPHRASE,
    )

    assert (
        extracted.parsed_envelope.payload
        == original
    )

    assert (
        extracted.parsed_envelope.filename
        == "sample.bin"
    )


def test_stego_image_remains_rgb_and_same_size():
    cover = create_cover()

    embedded = embed_secret(
        rgb=cover,
        payload=b"small message",
        payload_type="text",
        passphrase=PASSPHRASE,
        vision_service=create_service(),
        mime_type="text/plain",
    )

    assert (
        embedded.stego_image.shape
        == cover.shape
    )

    assert (
        embedded.stego_image.dtype
        == np.uint8
    )


def test_maximum_channel_change_is_one():
    cover = create_cover()

    embedded = embed_secret(
        rgb=cover,
        payload=b"test message",
        payload_type="text",
        passphrase=PASSPHRASE,
        vision_service=create_service(),
        mime_type="text/plain",
    )

    difference = np.abs(
        embedded.stego_image.astype(
            np.int16
        )
        - cover.astype(np.int16)
    )

    assert difference.max() <= 1

    assert (
        embedded.maximum_absolute_change
        <= 1
    )


def test_wrong_password_is_rejected():
    cover = create_cover()

    embedded = embed_secret(
        rgb=cover,
        payload=b"secret",
        payload_type="text",
        passphrase=PASSPHRASE,
        vision_service=create_service(),
        mime_type="text/plain",
    )

    with pytest.raises(
        ExtractorError,
        match="Extraction failed",
    ):
        extract_secret(
            embedded.stego_image,
            "Incorrect-Password-123",
        )


def test_payload_tampering_is_rejected():
    cover = create_cover()

    embedded = embed_secret(
        rgb=cover,
        payload=b"tamper protected secret",
        payload_type="text",
        passphrase=PASSPHRASE,
        vision_service=create_service(),
        mime_type="text/plain",
    )

    tampered = (
        embedded.stego_image.copy()
    )

    tampered_flat = tampered.reshape(-1)

    first_payload_position = int(
        embedded.payload_positions[0]
    )

    # Flips the embedded LSB at one payload position.
    tampered_flat[
        first_payload_position
    ] ^= 0x01

    with pytest.raises(
        ExtractorError,
        match="Extraction failed",
    ):
        extract_secret(
            tampered,
            PASSPHRASE,
        )

def test_position_groups_do_not_overlap():
    cover = create_cover()

    embedded = embed_secret(
        rgb=cover,
        payload=b"position test",
        payload_type="text",
        passphrase=PASSPHRASE,
        vision_service=create_service(),
        mime_type="text/plain",
    )

    bootstrap_set = set(
        embedded.bootstrap_positions.tolist()
    )

    metadata_set = set(
        embedded.metadata_positions.tolist()
    )

    payload_set = set(
        embedded.payload_positions.tolist()
    )

    assert bootstrap_set.isdisjoint(
        metadata_set
    )

    assert bootstrap_set.isdisjoint(
        payload_set
    )

    assert metadata_set.isdisjoint(
        payload_set
    )


def test_saved_png_round_trip(tmp_path):
    cover = create_cover()

    embedded = embed_secret(
        rgb=cover,
        payload=b"saved PNG test",
        payload_type="text",
        passphrase=PASSPHRASE,
        vision_service=create_service(),
        mime_type="text/plain",
    )

    output_path = (
        tmp_path / "stego.png"
    )

    Image.fromarray(
        embedded.stego_image
    ).save(output_path)

    reloaded = np.asarray(
        Image.open(
            output_path
        ).convert("RGB"),
        dtype=np.uint8,
    ).copy()

    extracted = extract_secret(
        reloaded,
        PASSPHRASE,
    )

    assert (
        extracted.parsed_envelope.payload
        == b"saved PNG test"
    )