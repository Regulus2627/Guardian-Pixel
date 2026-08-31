"""Real GuardianPixel embedding API."""

from __future__ import annotations

import io
import time

import numpy as np
from flask import (
    Blueprint,
    current_app,
    jsonify,
    request,
)
from PIL import Image

from backend.api.data_urls import (
    DataUrlError,
    decode_data_url,
    encode_data_url,
)
from backend.metrics.image_quality import (
    calculate_image_quality,
)
from backend.runtime.services import (
    get_vision_service,
)
from backend.stego.embedder import (
    EmbedderError,
    embed_secret,
)
from backend.vision.preprocessing import (
    ImagePreprocessingError,
    load_and_validate_image,
)


embed_blueprint = Blueprint(
    "embed",
    __name__,
    url_prefix="/api/v1",
)


@embed_blueprint.post("/embed")
def embed():
    """Encrypt and hide a real payload inside a cover image."""

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

    passphrase = body.get(
        "passphrase"
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
                "message": "A cover image is required.",
            }
        ), 400

    if raw_payload is None:
        return jsonify(
            {
                "code": "MISSING_PAYLOAD",
                "message": "A secret payload is required.",
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
                    "Payload type must be text, "
                    "file or image."
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
        _cover_mime, cover_bytes = (
            decode_data_url(
                cover_data_url,
                maximum_bytes=maximum_bytes,
            )
        )

        vision_service = (
            get_vision_service()
        )

        cover_rgb, image_info = (
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
                    maximum_bytes=maximum_bytes,
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

        start_time = time.perf_counter()

        # Run HED/entropy/variance only once.
        cached_analysis = (
            vision_service.analyze_cover(
                source=cover_rgb,
                required_payload_bits=0,
                channels_per_selected_pixel=1,
                reserved_position_count=0,
            )
        )

        embedded = embed_secret(
            rgb=cover_rgb,
            payload=payload_bytes,
            payload_type=payload_kind,
            passphrase=passphrase,
            vision_service=vision_service,
            filename=filename,
            mime_type=mime_type,
            precomputed_vision_result=(
                cached_analysis
            ),
        )

        embedding_seconds = (
            time.perf_counter()
            - start_time
        )

        output_buffer = io.BytesIO()

        Image.fromarray(
            embedded.stego_image
        ).save(
            output_buffer,
            format="PNG",
        )

        stego_bytes = (
            output_buffer.getvalue()
        )

        # Verify the actual encoded PNG pixels.
        with Image.open(
            io.BytesIO(stego_bytes)
        ) as stego_image:
            reloaded_stego = np.asarray(
                stego_image.convert("RGB"),
                dtype=np.uint8,
            ).copy()

        metrics = calculate_image_quality(
            cover=cover_rgb,
            stego=reloaded_stego,
            payload_bits=(
                len(
                    embedded
                    .encrypted_packet_bytes
                )
                * 8
            ),
            total_embedded_bits=(
                embedded.total_embedded_bits
            ),
        )

        response = {
            "stegoImageDataUrl": (
                encode_data_url(
                    "image/png",
                    stego_bytes,
                )
            ),
            "embedTimeMs": (
                embedding_seconds * 1000.0
            ),
            "method": (
                "GuardianPixel-HED-Adaptive-LSB"
            ),
            "bpp": metrics.total_bpp,
            "isSimulated": False,

            "payloadInfo": {
                "kind": payload_kind,
                "rawBytes": len(
                    payload_bytes
                ),
                "encryptedBytes": len(
                    embedded
                    .encrypted_packet_bytes
                ),
                "isSimulated": False,
                "filename": (
                    filename or None
                ),
                "mimeType": mime_type,
            },

            "metrics": {
                "mse": metrics.mse,
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "payloadBpp": (
                    metrics.payload_bpp
                ),
                "totalBpp": (
                    metrics.total_bpp
                ),
                "modifiedChannels": (
                    metrics
                    .modified_channel_count
                ),
                "modifiedPixels": (
                    metrics
                    .modified_pixel_count
                ),
                "maximumChange": (
                    metrics
                    .maximum_absolute_change
                ),
            },

            "locationMap": {
                "selectedBlocks": (
                    embedded
                    .block_map
                    .selected_block_count
                ),
                "totalBlocks": (
                    embedded
                    .block_map
                    .total_block_count
                ),
                "selectedPercentage": (
                    embedded
                    .block_map
                    .selected_percentage
                ),
                "encoding": (
                    embedded
                    .map_encoding
                    .encoding_type
                ),
                "encodedBytes": (
                    embedded
                    .map_encoding
                    .encoded_byte_count
                ),
            },

            "image": {
                "width": image_info.width,
                "height": image_info.height,
                "format": "PNG",
            },
        }

        return jsonify(response), 200

    except (
        DataUrlError,
        ImagePreprocessingError,
        EmbedderError,
        ValueError,
    ) as error:
        return jsonify(
            {
                "code": "EMBEDDING_INPUT_ERROR",
                "message": str(error),
            }
        ), 422

    except Exception:
        current_app.logger.exception(
            "Embedding failed."
        )

        return jsonify(
            {
                "code": "INTERNAL_ERROR",
                "message": "Embedding failed.",
            }
        ), 500