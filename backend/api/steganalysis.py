"""Serve real multi-image steganalysis validation results."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify


steganalysis_blueprint = Blueprint(
    "steganalysis",
    __name__,
    url_prefix="/api/v1",
)


@steganalysis_blueprint.get("/steganalysis")
def get_steganalysis_results():
    """Return the precomputed 20-image steganalysis validation results."""

    result_path = Path(
        current_app.config["STEGANALYSIS_RESULT_JSON"]
    )

    if not result_path.is_file():
        return jsonify(
            {
                "code": "STEGANALYSIS_RESULT_NOT_FOUND",
                "message": (
                    f"Required steganalysis result is missing: "
                    f"{result_path}"
                ),
            }
        ), 404

    try:
        result = json.loads(
            result_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        current_app.logger.exception(
            "Could not load steganalysis validation results."
        )
        return jsonify(
            {
                "code": "INVALID_STEGANALYSIS_RESULT",
                "message": (
                    "The steganalysis validation result file is invalid."
                ),
            }
        ), 500

    return jsonify(
        {
            "isSimulated": False,
            "validation": result.get("validation", {}),
            "metrics": result.get("metrics", {}),
            "perImage": result.get("per_image", []),
            "limitations": (
                "These classical steganalysis metrics are indicators "
                "and do not by themselves prove undetectability or "
                "security. The validation uses 20 DIV2K validation "
                "images with a 5000-byte payload and lossless PNG output."
            ),
        }
    ), 200
