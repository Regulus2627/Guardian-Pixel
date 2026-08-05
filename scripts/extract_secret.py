"""Extract and authenticate a GuardianPixel secret."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from backend.stego.extractor import (
    extract_secret,
)
from backend.vision.config import (
    load_vision_config,
)
from backend.vision.preprocessing import (
    load_and_validate_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an encrypted GuardianPixel "
            "secret from a stego PNG."
        )
    )

    parser.add_argument(
        "--stego",
        required=True,
        help="Path to the stego PNG.",
    )

    parser.add_argument(
        "--output-dir",
        default="sample_data/outputs/recovered",
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
    )

    arguments = parser.parse_args()

    stego_path = Path(
        arguments.stego
    )

    if stego_path.suffix.lower() != ".png":
        raise SystemExit(
            "Stego input must be a PNG."
        )

    passphrase = getpass.getpass(
        "Enter passphrase: "
    )

    config = load_vision_config(
        arguments.config
    )

    stego_rgb, _ = load_and_validate_image(
        stego_path,
        max_pixels=config.image.max_pixels,
    )

    result = extract_secret(
        stego_rgb,
        passphrase,
    )

    output_directory = Path(
        arguments.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    parsed = result.parsed_envelope

    if parsed.payload_type == "text":
        output_path = (
            output_directory
            / "recovered_text.txt"
        )

    else:
        output_path = (
            output_directory
            / parsed.filename
        )

    output_path.write_bytes(
        parsed.payload
    )

    print("")
    print("GuardianPixel extraction completed.")
    print(
        f"Payload type: "
        f"{parsed.payload_type}"
    )
    print(
        f"Recovered bytes: "
        f"{parsed.original_length}"
    )
    print(
        f"SHA-256: "
        f"{parsed.sha256_hex}"
    )
    print(
        f"Integrity verified: True"
    )
    print(
        f"Recovered output: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()