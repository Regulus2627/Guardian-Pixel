"""Real cover-payload compatibility API."""

from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    jsonify,
    request,
)

from backend.api.data_urls import (
    DataUrlError,
    decode_data_url,
)
from backend.compatibility.categories import (
    load_category_thresholds,
)
from backend.compatibility.engine import (
    CompatibilityEngineError,
    assess_compatibility,
)
from backend.runtime.services import (
    get_vision_service,
)
from backend.vision.preprocessing import (
    ImagePreprocessingError,
    load_and_validate_image,
)


compatibility_blueprint = Blueprint(
    "compatibility",
    __name__,
    url_prefix="/api/v1",
)


@compatibility_blueprint.post(
    "/compatibility"
)
def compatibility():
    """Assess one cover and payload before embedding."""

    if not request.is_json:
        return jsonify(
            {
                "code": "INVALID_REQUEST",
                "message": (
                    "Request must contain JSON."
                ),
            }
        ), 415

    body = request.get_json(
        silent=True
    )

    if not isinstance(body, dict):
        return jsonify(
            {
                "code": "INVALID_JSON",
                "message": (
                    "Request JSON is invalid."
                ),
            }
        ), 400

    cover_data_url = body.get(
        "coverDataUrl"
    )

    payload_kind = body.get(
        "payloadKind",
        "text",
    )

    raw_payload = body.get(
        "rawPayload"
    )

    payload_filename = body.get(
        "payloadFilename",
        "",
    )

    payload_mime_type = body.get(
        "payloadMimeType",
        "application/octet-stream",
    )

    if not cover_data_url:
        return jsonify(
            {
                "code": "MISSING_COVER",
                "message": (
                    "A cover image is required."
                ),
            }
        ), 400

    if payload_kind not in {
        "text",
        "file",
        "image",
    }:
        return jsonify(
            {
                "code": "INVALID_PAYLOAD_TYPE",
                "message": (
                    "Payload type must be "
                    "text, file or image."
                ),
            }
        ), 400

    if raw_payload is None:
        return jsonify(
            {
                "code": "MISSING_PAYLOAD",
                "message": (
                    "A secret payload is required."
                ),
            }
        ), 400

    maximum_bytes = current_app.config[
        "MAX_DECODED_FILE_BYTES"
    ]

    try:
        _cover_mime, cover_bytes = (
            decode_data_url(
                cover_data_url,
                maximum_bytes=maximum_bytes,
            )
        )

        vision_service = (
            get_vision_service()
        )

        cover_rgb, _ = (
            load_and_validate_image(
                cover_bytes,
                max_pixels=(
                    vision_service
                    .config
                    .image
                    .max_pixels
                ),
            )
        )

        if payload_kind == "text":
            if not isinstance(
                raw_payload,
                str,
            ):
                raise DataUrlError(
                    "Text payload must be a string."
                )

            payload_bytes = (
                raw_payload.encode("utf-8")
            )

            filename = ""
            mime_type = "text/plain"

        else:
            if not isinstance(
                raw_payload,
                str,
            ):
                raise DataUrlError(
                    "File payload must be a data URL."
                )

            decoded_mime, payload_bytes = (
                decode_data_url(
                    raw_payload,
                    maximum_bytes=(
                        maximum_bytes
                    ),
                )
            )

            filename = (
                payload_filename
                or (
                    "secret-image.png"
                    if payload_kind == "image"
                    else "secret-file.bin"
                )
            )

            mime_type = (
                payload_mime_type
                or decoded_mime
            )

        thresholds = (
            load_category_thresholds(
                current_app.config[
                    "CATEGORY_THRESHOLDS_PATH"
                ]
            )
        )

        result = assess_compatibility(
            rgb=cover_rgb,
            payload=payload_bytes,
            payload_type=payload_kind,
            filename=filename,
            mime_type=mime_type,
            vision_service=vision_service,
            thresholds=thresholds,
        )

        response = {
            # Existing frontend-compatible fields
            "textureScore": (
                result.texture_score
            ),
            "headroomRatio": (
                result.headroom_ratio
            ),
            "classification": (
                result.classification
            ),
            "recommended": (
                result.recommended
            ),
            "reason": result.reason,

            # Extended real-backend fields
            "compatible": (
                result.compatible
            ),
            "isProvisional": (
                result.is_provisional
            ),
            "isSimulated": False,

            "cover": {
                "category": (
                    result.cover_category
                ),
                "width": (
                    result.image_width
                ),
                "height": (
                    result.image_height
                ),
                "totalPixels": (
                    result.total_pixels
                ),
            },

            "payload": {
                "kind": (
                    result.payload_type
                ),
                "originalBytes": (
                    result
                    .original_payload_bytes
                ),
                "envelopeBytes": (
                    result.envelope_bytes
                ),
                "encryptedPacketBits": (
                    result
                    .encrypted_packet_bits
                ),
            },

            "capacity": {
                "bootstrapBits": (
                    result.bootstrap_bits
                ),
                "metadataBits": (
                    result.metadata_bits
                ),
                "totalRequiredBits": (
                    result
                    .total_required_bits
                ),
                "requiredWithSafety": (
                    result
                    .required_positions_with_safety
                ),
                "maximumPositions": (
                    result.maximum_positions
                ),
                "remainingPositions": (
                    result.remaining_positions
                ),
            },

            "locationMap": {
                "selectedBlocks": (
                    result.selected_blocks
                ),
                "totalBlocks": (
                    result.total_blocks
                ),
                "selectedPercentage": (
                    result
                    .selected_percentage
                ),
                "encoding": (
                    result.map_encoding
                ),
                "encodedBytes": (
                    result
                    .encoded_map_bytes
                ),
            },

            "details": {
                "bppTarget": (
                    result.total_required_bits
                    / result.total_pixels
                ),
                "isSupportedFormat": True,
                "hasMinimumResolution": (
                    result.total_pixels
                    >= 65_536
                ),
            },
        }

        return jsonify(response), 200

    except (
        DataUrlError,
        ImagePreprocessingError,
        CompatibilityEngineError,
        ValueError,
    ) as error:
        return jsonify(
            {
                "code": (
                    "COMPATIBILITY_INPUT_ERROR"
                ),
                "message": str(error),
            }
        ), 422

    except Exception:
        current_app.logger.exception(
            "Compatibility analysis failed."
        )

        return jsonify(
            {
                "code": "INTERNAL_ERROR",
                "message": (
                    "Compatibility analysis failed."
                ),
            }
        ), 500