import base64
import io

import numpy as np
from PIL import Image

import backend.api.embed as embed_api
from backend.app import create_app
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
        return np.mean(
            rgb.astype(np.float32),
            axis=2,
        ) / 255.0


def create_cover_data_url() -> str:
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


def test_real_embed_endpoint(monkeypatch):
    monkeypatch.setattr(
        embed_api,
        "get_vision_service",
        create_test_service,
    )

    app = create_app(
        {
            "TESTING": True,
        }
    )

    client = app.test_client()

    response = client.post(
        "/api/v1/embed",
        json={
            "coverDataUrl": (
                create_cover_data_url()
            ),
            "payloadKind": "text",
            "rawPayload": (
                "GuardianPixel API secret"
            ),
            "passphrase": (
                "GuardianPixel-Test-Password"
            ),
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["isSimulated"] is False

    assert data[
        "stegoImageDataUrl"
    ].startswith(
        "data:image/png;base64,"
    )

    assert data["metrics"][
        "maximumChange"
    ] <= 1

    assert data["payloadInfo"][
        "rawBytes"
    ] > 0