import base64
import io

import numpy as np
from PIL import Image

import backend.api.compatibility as compatibility_api
from backend.app import create_app
from backend.compatibility.categories import (
    CategoryThresholds,
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


def create_image_data_url() -> str:
    generator = np.random.default_rng(
        42
    )

    rgb = generator.integers(
        0,
        256,
        size=(256, 256, 3),
        dtype=np.uint8,
    )

    buffer = io.BytesIO()

    Image.fromarray(rgb).save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return (
        "data:image/png;base64,"
        + encoded
    )


def create_test_service():
    return GuardianPixelVisionService(
        config=load_vision_config(),
        hed_predictor=FakeHEDPredictor(),
    )


def test_text_compatibility_endpoint(
    monkeypatch,
):
    monkeypatch.setattr(
        compatibility_api,
        "get_vision_service",
        create_test_service,
    )

    monkeypatch.setattr(
        compatibility_api,
        "load_category_thresholds",
        lambda _path: CategoryThresholds(
            smooth_upper=0.10,
            textured_lower=0.30,
        ),
    )

    app = create_app(
        {
            "TESTING": True,
        }
    )

    client = app.test_client()

    response = client.post(
        "/api/v1/compatibility",
        json={
            "coverDataUrl": (
                create_image_data_url()
            ),
            "payloadKind": "text",
            "rawPayload": (
                "GuardianPixel API test"
            ),
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["isSimulated"] is False
    assert "compatible" in data
    assert "classification" in data
    assert "textureScore" in data
    assert "capacity" in data
    assert "locationMap" in data

    assert (
        data["payload"]["originalBytes"]
        > 0
    )


def test_missing_cover_is_rejected():
    app = create_app(
        {
            "TESTING": True,
        }
    )

    client = app.test_client()

    response = client.post(
        "/api/v1/compatibility",
        json={
            "payloadKind": "text",
            "rawPayload": "secret",
        },
    )

    assert response.status_code == 400
    assert (
        response.get_json()["code"]
        == "MISSING_COVER"
    )


def test_non_json_request_is_rejected():
    app = create_app(
        {
            "TESTING": True,
        }
    )

    client = app.test_client()

    response = client.post(
        "/api/v1/compatibility",
        data="not-json",
        content_type="text/plain",
    )

    assert response.status_code == 415