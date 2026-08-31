"""GuardianPixel Flask application factory."""

from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS

from backend.api.compatibility import (
    compatibility_blueprint,
)
from backend.api.embed import (
    embed_blueprint,
)
from backend.api.extract import (
    extract_blueprint,
)
from backend.api.health import (
    health_blueprint,
)
from backend.web_config import (
    WebConfig,
)


def create_app(
    test_config: dict | None = None,
) -> Flask:
    """Create and configure the GuardianPixel Flask application."""

    app = Flask(__name__)

    app.config.from_object(
        WebConfig
    )

    if test_config is not None:
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

    # Register every API blueprint.
    app.register_blueprint(
        health_blueprint
    )

    app.register_blueprint(
        compatibility_blueprint
    )

    app.register_blueprint(
        embed_blueprint
    )

    app.register_blueprint(
        extract_blueprint
    )

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "GuardianPixel",
                "status": "running",
                "apiVersion": "v1",
            }
        ), 200

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify(
            {
                "code": "REQUEST_TOO_LARGE",
                "message": (
                    "The uploaded request exceeds "
                    "the configured size limit."
                ),
            }
        ), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )