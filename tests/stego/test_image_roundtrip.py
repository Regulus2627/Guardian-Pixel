"""
Image-on-image payload round-trip tests.

Mirrors tests/stego/test_roundtrip.py's conventions exactly (same
FakeHEDPredictor, same create_cover/create_service helpers) but covers
payload_type="image" specifically, which test_roundtrip.py does not
touch (it only covers "text" and "file").

Covers spec tasks 2-3 under "Image-on-image tasks":
  - secret-image categories: smooth logo/graphic, natural photograph,
    textured/noisy image, PNG vs JPEG file bytes
  - secret dimensions: 32x32, 64x64, 128x128, 256x256
  - always records actual encoded bytes
  - the "does not fit" path (task 4) without silently resizing (task 5)
"""

import io
import itertools

import numpy as np
import pytest
from PIL import Image

from backend.stego.embedder import embed_secret
from backend.stego.extractor import ExtractorError, extract_secret
from backend.vision.config import load_vision_config
from backend.vision.service import GuardianPixelVisionService


class FakeHEDPredictor:
    """Fast deterministic HED replacement for round-trip tests.
    (Identical to tests/stego/test_roundtrip.py's version.)"""

    def __init__(self):
        self.last_tile_count = 1
        self.call_count = 0

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        luminance = np.mean(rgb.astype(np.float32), axis=2)
        self.call_count += 1
        minimum = float(luminance.min())
        maximum = float(luminance.max())
        value_range = maximum - minimum
        if value_range <= 0:
            return np.zeros(rgb.shape[:2], dtype=np.float32)
        return ((luminance - minimum) / value_range).astype(np.float32)


PASSPHRASE = "GuardianPixel-ImageRoundTrip-Password"

SIZES = [32, 64, 128, 256]
FORMATS = ["PNG", "JPEG"]

# Large enough that every secret in the grid fits comfortably; isolates
# "does exact image-in-image round-trip correctly" from "does it fit"
# (the latter is tested separately below).
COVER_SIDE = 1400


def create_cover(width: int = 256, height: int = 256) -> np.ndarray:
    generator = np.random.default_rng(42)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def create_service():
    return GuardianPixelVisionService(
        config=load_vision_config(),
        hed_predictor=FakeHEDPredictor(),
    )


# --------------------------------------------------------------------------
# Secret-image category generators
# --------------------------------------------------------------------------

def _make_smooth(size: int) -> np.ndarray:
    """Smooth logo/graphic: flat color blocks, low entropy."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[: size // 2, : size // 2] = (240, 240, 245)
    arr[size // 2:, size // 2:] = (20, 20, 30)
    arr[: size // 2, size // 2:] = (200, 30, 30)
    arr[size // 2:, : size // 2] = (30, 120, 200)
    return arr


def _make_photo(size: int) -> np.ndarray:
    """Natural photograph stand-in: smooth gradient + soft noise."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    base = (np.stack([xx, yy, (xx + yy) / 2], axis=-1) * 255).astype(np.float64)
    noise = np.random.default_rng(0).normal(0, 8, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _make_textured(size: int) -> np.ndarray:
    """Textured/noisy image: high-frequency random noise (worst case for capacity)."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


SECRET_GENERATORS = {
    "smooth": _make_smooth,
    "photo": _make_photo,
    "textured": _make_textured,
}


def _encode_secret_bytes(category: str, size: int, fmt: str) -> tuple[bytes, str, str]:
    """Returns (raw_file_bytes, filename, mime_type) for a secret image."""
    arr = SECRET_GENERATORS[category](size)
    buf = io.BytesIO()
    im = Image.fromarray(arr)
    if fmt == "JPEG":
        im.save(buf, format="JPEG", quality=90)
        filename, mime = f"secret_{category}_{size}.jpg", "image/jpeg"
    else:
        im.save(buf, format="PNG")
        filename, mime = f"secret_{category}_{size}.png", "image/png"
    return buf.getvalue(), filename, mime


# --------------------------------------------------------------------------
# Full grid: category x size x format -> exact round trip via embed_secret/
# extract_secret directly (no HTTP layer, matching test_roundtrip.py's style)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "category,size,fmt",
    list(itertools.product(SECRET_GENERATORS.keys(), SIZES, FORMATS)),
)
def test_image_payload_round_trip(category, size, fmt):
    cover = create_cover(width=COVER_SIDE, height=COVER_SIDE)
    original_bytes, filename, mime_type = _encode_secret_bytes(category, size, fmt)

    embedded = embed_secret(
        rgb=cover,
        payload=original_bytes,
        payload_type="image",
        filename=filename,
        mime_type=mime_type,
        passphrase=PASSPHRASE,
        vision_service=create_service(),
    )

    extracted = extract_secret(embedded.stego_image, PASSPHRASE)
    parsed = extracted.parsed_envelope

    assert parsed.payload_type == "image"
    assert parsed.payload == original_bytes, "recovered bytes are not byte-exact"
    assert parsed.filename == filename
    assert len(parsed.payload) == len(original_bytes)  # actual encoded bytes recorded


def test_image_payload_survives_png_save_reload(tmp_path):
    """Confirms the exact PNG round-trip (not just in-memory) for an image secret."""
    cover = create_cover(width=COVER_SIDE, height=COVER_SIDE)
    original_bytes, filename, mime_type = _encode_secret_bytes("photo", 128, "PNG")

    embedded = embed_secret(
        rgb=cover,
        payload=original_bytes,
        payload_type="image",
        filename=filename,
        mime_type=mime_type,
        passphrase=PASSPHRASE,
        vision_service=create_service(),
    )

    output_path = tmp_path / "stego.png"
    Image.fromarray(embedded.stego_image).save(output_path)

    reloaded = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8).copy()
    extracted = extract_secret(reloaded, PASSPHRASE)

    assert extracted.parsed_envelope.payload == original_bytes


def test_wrong_passphrase_rejected_for_image_payload():
    cover = create_cover(width=COVER_SIDE, height=COVER_SIDE)
    original_bytes, filename, mime_type = _encode_secret_bytes("smooth", 64, "PNG")

    embedded = embed_secret(
        rgb=cover,
        payload=original_bytes,
        payload_type="image",
        filename=filename,
        mime_type=mime_type,
        passphrase=PASSPHRASE,
        vision_service=create_service(),
    )

    with pytest.raises(ExtractorError, match="Extraction failed"):
        extract_secret(embedded.stego_image, "Wrong-Password-123")


# --------------------------------------------------------------------------
# Task 4/5: secret image does not fit a small cover -> must fail loudly,
# never silently resize or truncate.
#
# NOTE: the exact exception type raised by embed_secret()'s capacity
# selection (via backend.vision.block_map.generate_capacity_aware_block_map)
# wasn't visible to me when writing this — if it raises something more
# specific than a bare Exception (e.g. a BlockMapError/CapacityError),
# narrow the `pytest.raises` below to that type.
# --------------------------------------------------------------------------

def test_oversized_image_payload_does_not_silently_fit():
    tiny_cover = create_cover(width=24, height=24)
    original_bytes, filename, mime_type = _encode_secret_bytes("textured", 256, "PNG")

    with pytest.raises(Exception):
        embed_secret(
            rgb=tiny_cover,
            payload=original_bytes,
            payload_type="image",
            filename=filename,
            mime_type=mime_type,
            passphrase=PASSPHRASE,
            vision_service=create_service(),
        )