"""GuardianPixel health endpoint."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify

from backend.core.protocol import (
    BOOTSTRAP_VERSION,
)
from backend.vision.config import (
    load_vision_config,
)


health_blueprint = Blueprint(
    "health",
    __name__,
    url_prefix="/api/v1",
)


@health_blueprint.get("/health")
def health():
    """Return lightweight application readiness information."""

    try:
        vision_config = load_vision_config()

        prototxt_available = Path(
            vision_config.hed.prototxt_path
        ).is_file()

        weights_available = Path(
            vision_config.hed.weights_path
        ).is_file()

        hed_model_available = (
            prototxt_available
            and weights_available
        )

        response = {
            "status": (
                "ok"
                if hed_model_available
                else "degraded"
            ),
            "service": "GuardianPixel",
            "protocol": (
                f"GPX{BOOTSTRAP_VERSION}"
            ),
            "hed_model_available": (
                hed_model_available
            ),
            "prototxt_available": (
                prototxt_available
            ),
            "weights_available": (
                weights_available
            ),
        }

        status_code = (
            200
            if hed_model_available
            else 503
        )

        return jsonify(response), status_code

    except Exception:
        return jsonify(
            {
                "status": "error",
                "service": "GuardianPixel",
                "protocol": (
                    f"GPX{BOOTSTRAP_VERSION}"
                ),
                "hed_model_available": False,
            }
        ), 500