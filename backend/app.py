"""GuardianPixel Flask application factory."""

from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from backend.api.health import (
    health_blueprint,
)
from backend.web_config import (
    WebConfig,
)


def create_app(
    test_config: dict | None = None,
) -> Flask:
    """Create and configure the GuardianPixel Flask app."""

    app = Flask(__name__)

    app.config.from_object(
        WebConfig
    )

    if test_config:
        app.config.update(
            test_config
        )

    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": app.config[
                    "CORS_ORIGINS"
                ]
            }
        },
    )

    app.register_blueprint(
        health_blueprint
    )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )