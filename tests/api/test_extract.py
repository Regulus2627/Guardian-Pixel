import backend.api.embed as embed_api
from backend.app import create_app
from tests.api.test_embed import (
    create_cover_data_url,
    create_test_service,
)


PASSPHRASE = (
    "GuardianPixel-Test-Password"
)


def test_api_embed_extract_round_trip(
    monkeypatch,
):
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

    original_text = (
        "GuardianPixel real API round trip"
    )

    embed_response = client.post(
        "/api/v1/embed",
        json={
            "coverDataUrl": (
                create_cover_data_url()
            ),
            "payloadKind": "text",
            "rawPayload": original_text,
            "passphrase": PASSPHRASE,
        },
    )

    assert embed_response.status_code == 200, (
        embed_response.get_data(
            as_text=True
        )
    )

    embed_data = (
        embed_response.get_json()
    )

    assert embed_data is not None

    stego_data_url = (
        embed_data[
            "stegoImageDataUrl"
        ]
    )

    assert embed_response.status_code == 200, (
        embed_response.get_data(
            as_text=True
        )
    )

    embed_data = (
        embed_response.get_json()
    )

    assert embed_data is not None

    stego_data_url = (
        embed_data[
            "stegoImageDataUrl"
        ]
    )

    extract_response = client.post(
        "/api/v1/extract",
        json={
            "stegoImageDataUrl": (
                stego_data_url
            ),
            "passphrase": PASSPHRASE,
        },
    )

    assert extract_response.status_code == 200

    result = extract_response.get_json()

    assert result["isSimulated"] is False
    assert result["integrityVerified"] is True
    assert (
        result["textContent"]
        == original_text
    )


def test_api_wrong_password_is_rejected(
    monkeypatch,
):
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

    embed_response = client.post(
        "/api/v1/embed",
        json={
            "coverDataUrl": (
                create_cover_data_url()
            ),
            "payloadKind": "text",
            "rawPayload": "Protected secret",
            "passphrase": PASSPHRASE,
        },
    )

    stego_data_url = (
        embed_response.get_json()[
            "stegoImageDataUrl"
        ]
    )

    extract_response = client.post(
        "/api/v1/extract",
        json={
            "stegoImageDataUrl": (
                stego_data_url
            ),
            "passphrase": (
                "Incorrect-Password-123"
            ),
        },
    )

    assert extract_response.status_code == 401

    result = extract_response.get_json()

    assert (
        result["code"]
        == "AUTH_OR_DATA_FAILED"
    )