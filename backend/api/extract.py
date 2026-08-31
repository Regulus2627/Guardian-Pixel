"""Real GuardianPixel extraction API."""

from __future__ import annotations

import time

from flask import (
    Blueprint,
    current_app,
    jsonify,
    request,
)

from backend.api.data_urls import (
    DataUrlError,
    decode_data_url,
    encode_data_url,
)
from backend.stego.extractor import (
    ExtractorError,
    extract_secret,
)
from backend.vision.config import (
    load_vision_config,
)
from backend.vision.preprocessing import (
    ImagePreprocessingError,
    load_and_validate_image,
)


extract_blueprint = Blueprint(
    "extract",
    __name__,
    url_prefix="/api/v1",
)


@extract_blueprint.post("/extract")
def extract():
    """Extract and authenticate a GuardianPixel payload."""

    if not request.is_json:
        return jsonify(
            {
                "code": "INVALID_REQUEST",
                "message": "Request must contain JSON.",
            }
        ), 415

    body = request.get_json(
        silent=True
    )

    if not isinstance(body, dict):
        return jsonify(
            {
                "code": "INVALID_JSON",
                "message": "Request JSON is invalid.",
            }
        ), 400

    stego_data_url = body.get(
        "stegoImageDataUrl"
    )

    passphrase = body.get(
        "passphrase"
    )

    if not stego_data_url:
        return jsonify(
            {
                "code": "MISSING_STEGO_IMAGE",
                "message": (
                    "A stego PNG is required."
                ),
            }
        ), 400

    if not isinstance(
        passphrase,
        str,
    ) or len(passphrase) < 8:
        return jsonify(
            {
                "code": "INVALID_PASSPHRASE",
                "message": (
                    "Passphrase must contain "
                    "at least 8 characters."
                ),
            }
        ), 400

    maximum_bytes = current_app.config[
        "MAX_DECODED_FILE_BYTES"
    ]

    try:
        mime_type, stego_bytes = (
            decode_data_url(
                stego_data_url,
                maximum_bytes=maximum_bytes,
            )
        )

        if mime_type != "image/png":
            return jsonify(
                {
                    "code": "LOSSY_STEGO_NOT_ALLOWED",
                    "message": (
                        "Stego input must be a "
                        "lossless PNG."
                    ),
                }
            ), 415

        vision_config = load_vision_config()

        stego_rgb, _ = (
            load_and_validate_image(
                stego_bytes,
                max_pixels=(
                    vision_config
                    .image
                    .max_pixels
                ),
            )
        )

        start_time = time.perf_counter()

        result = extract_secret(
            stego_rgb=stego_rgb,
            passphrase=passphrase,
        )

        extraction_seconds = (
            time.perf_counter()
            - start_time
        )

        parsed = result.parsed_envelope

        response = {
            "kind": (
                parsed.payload_type
            ),
            "extractTimeMs": (
                extraction_seconds * 1000.0
            ),
            "integrityVerified": True,
            "method": (
                "GuardianPixel-HED-Adaptive-LSB"
            ),
            "isSimulated": False,
            "filename": (
                parsed.filename or None
            ),
            "mimeType": (
                parsed.mime_type
            ),
            "fileSizeBytes": (
                parsed.original_length
            ),
            "sha256": (
                parsed.sha256_hex
            ),
        }

        if parsed.payload_type == "text":
            try:
                response["textContent"] = (
                    parsed.payload.decode(
                        "utf-8"
                    )
                )

            except UnicodeDecodeError as error:
                raise ExtractorError(
                    "Recovered text is not valid UTF-8."
                ) from error

        else:
            response["fileDataUrl"] = (
                encode_data_url(
                    parsed.mime_type,
                    parsed.payload,
                )
            )

        return jsonify(response), 200

    except ExtractorError as error:
        return jsonify(
            {
                "code": "AUTH_OR_DATA_FAILED",
                "message": str(error),
            }
        ), 401

    except (
        DataUrlError,
        ImagePreprocessingError,
        ValueError,
    ) as error:
        return jsonify(
            {
                "code": "EXTRACTION_INPUT_ERROR",
                "message": str(error),
            }
        ), 422

    except Exception:
        current_app.logger.exception(
            "Extraction failed."
        )

        return jsonify(
            {
                "code": "INTERNAL_ERROR",
                "message": "Extraction failed.",
            }
        ), 500