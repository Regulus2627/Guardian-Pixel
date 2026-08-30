import numpy as np

from backend.compatibility.categories import (
    CategoryThresholds,
)
from backend.compatibility.engine import (
    assess_compatibility,
)
from backend.vision.config import (
    load_vision_config,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)


class FakeHEDPredictor:
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

        if maximum <= minimum:
            return np.zeros(
                rgb.shape[:2],
                dtype=np.float32,
            )

        return (
            (
                luminance - minimum
            )
            / (maximum - minimum)
        ).astype(np.float32)


def create_service():
    return GuardianPixelVisionService(
        config=load_vision_config(),
        hed_predictor=FakeHEDPredictor(),
    )


def create_cover(
    width: int,
    height: int,
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


THRESHOLDS = CategoryThresholds(
    smooth_upper=0.10,
    textured_lower=0.30,
)


def test_small_payload_is_compatible():
    result = assess_compatibility(
        rgb=create_cover(
            width=256,
            height=256,
        ),
        payload=b"Small payload",
        payload_type="text",
        vision_service=create_service(),
        thresholds=THRESHOLDS,
    )

    assert result.compatible is True
    assert result.total_required_bits > 0
    assert result.maximum_positions == (
        256 * 256
    )

    assert result.classification in {
        "Excellent",
        "Good",
        "Moderate",
        "Poor",
    }


def test_large_payload_is_incompatible():
    result = assess_compatibility(
        rgb=create_cover(
            width=64,
            height=64,
        ),
        payload=b"A" * 100_000,
        payload_type="file",
        filename="random.bin",
        mime_type=(
            "application/octet-stream"
        ),
        vision_service=create_service(),
        thresholds=THRESHOLDS,
    )

    # Repetitive A data compresses strongly, so use a random payload
    # if this unexpectedly fits.
    if result.compatible:
        random_payload = (
            np.random.default_rng(100)
            .integers(
                0,
                256,
                size=100_000,
                dtype=np.uint8,
            )
            .tobytes()
        )

        result = assess_compatibility(
            rgb=create_cover(
                width=64,
                height=64,
            ),
            payload=random_payload,
            payload_type="file",
            filename="random.bin",
            vision_service=create_service(),
            thresholds=THRESHOLDS,
        )

    assert result.compatible is False
    assert result.recommended is False
    assert result.classification == "Poor"


def test_image_payload_requires_filename():
    try:
        assess_compatibility(
            rgb=create_cover(
                width=256,
                height=256,
            ),
            payload=b"image data",
            payload_type="image",
            filename="",
            vision_service=create_service(),
            thresholds=THRESHOLDS,
        )

    except ValueError as error:
        assert "filename" in str(error)
    else:
        raise AssertionError(
            "Missing filename was not rejected."
        )