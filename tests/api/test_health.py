from backend.app import create_app


def test_health_endpoint():
    app = create_app(
        {
            "TESTING": True,
        }
    )

    client = app.test_client()

    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code in {
        200,
        503,
    }

    data = response.get_json()

    assert data["service"] == "GuardianPixel"
    assert data["protocol"] == "GPX1"

    assert "status" in data
    assert "hed_model_available" in data