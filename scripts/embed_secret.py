"""Create a real encrypted GuardianPixel stego PNG."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import mimetypes
from pathlib import Path

import numpy as np
from PIL import Image

from backend.metrics.image_quality import (
    calculate_image_quality,
)
from backend.stego.embedder import (
    embed_secret,
)
from backend.stego.extractor import (
    extract_secret,
)
from backend.vision.config import (
    load_vision_config,
)
from backend.vision.preprocessing import (
    load_and_validate_image,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encrypt and hide text or a file "
            "inside a lossless PNG."
        )
    )

    parser.add_argument(
        "--cover",
        required=True,
        help="Path to the cover image.",
    )

    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    input_group.add_argument(
        "--text-file",
        help=(
            "UTF-8 text file whose content "
            "will be hidden as text."
        ),
    )

    input_group.add_argument(
        "--secret-file",
        help="Binary file to hide.",
    )

    parser.add_argument(
        "--output",
        default="sample_data/outputs/stego.png",
        help="Output stego PNG path.",
    )

    parser.add_argument(
        "--metrics",
        default="sample_data/outputs/embed_metrics.json",
        help="Metrics JSON output path.",
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
    )

    arguments = parser.parse_args()

    passphrase = getpass.getpass(
        "Enter passphrase: "
    )

    confirmation = getpass.getpass(
        "Confirm passphrase: "
    )

    if passphrase != confirmation:
        raise SystemExit(
            "Passphrases do not match."
        )

    config = load_vision_config(
        arguments.config
    )

    cover_rgb, _ = load_and_validate_image(
        arguments.cover,
        max_pixels=config.image.max_pixels,
    )

    if arguments.text_file:
        secret_path = Path(
            arguments.text_file
        )

        payload = secret_path.read_text(
            encoding="utf-8"
        ).encode("utf-8")

        payload_type = "text"
        filename = ""
        mime_type = "text/plain"

    else:
        secret_path = Path(
            arguments.secret_file
        )

        payload = secret_path.read_bytes()
        payload_type = "file"
        filename = secret_path.name

        mime_type = (
            mimetypes.guess_type(
                filename
            )[0]
            or "application/octet-stream"
        )

    vision_service = (
        GuardianPixelVisionService(
            config=config
        )
    )

    embedded = embed_secret(
        rgb=cover_rgb,
        payload=payload,
        payload_type=payload_type,
        passphrase=passphrase,
        vision_service=vision_service,
        filename=filename,
        mime_type=mime_type,
    )

    output_path = Path(
        arguments.output
    )

    if output_path.suffix.lower() != ".png":
        raise SystemExit(
            "Stego output must use the .png extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        embedded.stego_image
    ).save(
        output_path,
        format="PNG",
    )

    # Reload the actual saved PNG and verify extraction.
    reloaded_stego = np.asarray(
        Image.open(
            output_path
        ).convert("RGB"),
        dtype=np.uint8,
    ).copy()

    verification = extract_secret(
        reloaded_stego,
        passphrase,
    )

    round_trip_verified = (
        verification
        .parsed_envelope
        .payload
        == payload
    )

    metrics = calculate_image_quality(
        cover=cover_rgb,
        stego=reloaded_stego,
        payload_bits=(
            len(embedded.encrypted_packet_bytes)
            * 8
        ),
        total_embedded_bits=(
            embedded.total_embedded_bits
        ),
    )

    metrics_data = {
        "output": str(output_path),
        "payload_type": payload_type,
        "payload_bytes": len(payload),
        "encrypted_packet_bytes": len(
            embedded.encrypted_packet_bytes
        ),
        "round_trip_verified": (
            round_trip_verified
        ),
        "image_width": cover_rgb.shape[1],
        "image_height": cover_rgb.shape[0],
        "mse": metrics.mse,
        "psnr": (
            "Infinity"
            if math.isinf(metrics.psnr)
            else metrics.psnr
        ),
        "ssim": metrics.ssim,
        "payload_bpp": metrics.payload_bpp,
        "total_bpp": metrics.total_bpp,
        "modified_channels": (
            metrics.modified_channel_count
        ),
        "modified_pixels": (
            metrics.modified_pixel_count
        ),
        "maximum_absolute_change": (
            metrics.maximum_absolute_change
        ),
        "selected_blocks": (
            embedded.block_map
            .selected_block_count
        ),
        "selected_percentage": (
            embedded.block_map
            .selected_percentage
        ),
        "map_encoding": (
            embedded.map_encoding
            .encoding_type
        ),
        "encoded_map_bytes": (
            embedded.map_encoding
            .encoded_byte_count
        ),
    }

    metrics_path = Path(
        arguments.metrics
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.write_text(
        json.dumps(
            metrics_data,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("GuardianPixel embedding completed.")
    print(f"Stego PNG: {output_path}")
    print(
        f"Round-trip verified: "
        f"{round_trip_verified}"
    )
    print(f"PSNR: {metrics.psnr:.4f} dB")
    print(f"SSIM: {metrics.ssim:.8f}")
    print(
        f"Maximum change: "
        f"{metrics.maximum_absolute_change}"
    )
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()