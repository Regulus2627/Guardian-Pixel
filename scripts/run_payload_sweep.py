"""Run GuardianPixel text payload compatibility experiments."""

from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from backend.compatibility.payload import (
    analyse_text_payload,
    generate_text_payload,
)
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


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


def parse_integer_list(
    value: str,
) -> list[int]:
    """Parse comma-separated positive integers."""

    try:
        values = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Payload sizes must be integers."
        ) from error

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one payload size is required."
        )

    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "Payload sizes must be positive."
        )

    return values


def parse_profile_list(
    value: str,
) -> list[str]:
    """Parse and validate payload profiles."""

    profiles = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    allowed = {
        "english",
        "repetitive",
        "random_ascii",
    }

    if not profiles:
        raise argparse.ArgumentTypeError(
            "At least one payload profile is required."
        )

    invalid = [
        item
        for item in profiles
        if item not in allowed
    ]

    if invalid:
        raise argparse.ArgumentTypeError(
            "Profiles must be english, repetitive "
            "or random_ascii."
        )

    return profiles


def load_cover_feature_lookup(
    csv_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Load image categories and texture scores."""

    path = Path(csv_path)

    if not path.exists():
        raise SystemExit(
            f"Cover-feature CSV was not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    return {
        row["image_id"]: row
        for row in rows
    }


def png_round_trip(
    rgb: np.ndarray,
) -> np.ndarray:
    """Save and reload an RGB image as lossless PNG in memory."""

    buffer = io.BytesIO()

    Image.fromarray(rgb).save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    with Image.open(buffer) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        ).copy()


def calculate_compatibility_class(
    exact_recovery: bool,
    psnr: float,
    ssim: float,
    maximum_change: int,
    minimum_psnr: float,
    minimum_ssim: float,
) -> tuple[bool, str, str]:
    """Apply provisional recovery and quality rules."""

    if not exact_recovery:
        return (
            False,
            "Incompatible",
            "EXACT_RECOVERY_FAILED",
        )

    if maximum_change > 1:
        return (
            False,
            "Incompatible",
            "CHANNEL_CHANGE_EXCEEDED",
        )

    if (
        psnr < minimum_psnr
        or ssim < minimum_ssim
    ):
        return (
            False,
            "Risky",
            "QUALITY_BELOW_THRESHOLD",
        )

    if psnr >= 50 and ssim >= 0.995:
        return (
            True,
            "Excellent",
            "",
        )

    return (
        True,
        "Good",
        "",
    )


def write_csv(
    rows: list[dict],
    path: str | Path,
) -> Path:
    """Write experiment rows to CSV."""

    if not rows:
        raise SystemExit(
            "No experiment rows were generated."
        )

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def build_summary(
    rows: list[dict],
) -> list[dict]:
    """Find maximum compatible text size for each cover/profile."""

    grouped: dict[
        tuple[str, str],
        list[dict],
    ] = {}

    for row in rows:
        key = (
            row["image_id"],
            row["payload_profile"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    summary_rows = []

    for (
        image_id,
        profile,
    ), group_rows in sorted(
        grouped.items()
    ):
        compatible_rows = [
            row
            for row in group_rows
            if row["compatible"] is True
        ]

        if compatible_rows:
            best = max(
                compatible_rows,
                key=lambda row: (
                    row["payload_utf8_bytes"]
                ),
            )

            maximum_safe_bytes = (
                best["payload_utf8_bytes"]
            )

            maximum_safe_total_bpp = (
                best["total_bpp"]
            )

            psnr_at_maximum = best["psnr"]
            ssim_at_maximum = best["ssim"]

            selected_percentage = (
                best["selected_percentage"]
            )

            result_class = (
                best["compatibility_class"]
            )

        else:
            first = group_rows[0]

            maximum_safe_bytes = 0
            maximum_safe_total_bpp = 0.0
            psnr_at_maximum = ""
            ssim_at_maximum = ""
            selected_percentage = ""
            result_class = "Incompatible"

            best = first

        summary_rows.append(
            {
                "image_id": image_id,
                "cover_category": (
                    best["cover_category"]
                ),
                "texture_score": (
                    best["texture_score"]
                ),
                "width": best["width"],
                "height": best["height"],
                "payload_profile": profile,
                "maximum_tested_safe_bytes": (
                    maximum_safe_bytes
                ),
                "maximum_safe_total_bpp": (
                    maximum_safe_total_bpp
                ),
                "psnr_at_maximum": (
                    psnr_at_maximum
                ),
                "ssim_at_maximum": (
                    ssim_at_maximum
                ),
                "selected_percentage_at_maximum": (
                    selected_percentage
                ),
                "result_class": (
                    result_class
                ),
            }
        )

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run cached GuardianPixel payload-size "
            "experiments."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing cover images.",
    )

    parser.add_argument(
        "--cover-features-csv",
        required=True,
        help=(
            "CSV containing fixed cover categories "
            "and texture scores."
        ),
    )

    parser.add_argument(
        "--payload-sizes",
        type=parse_integer_list,
        default=parse_integer_list(
            "50,100,250,500,1000,2000,5000,10000,20000,50000"
        ),
    )

    parser.add_argument(
        "--profiles",
        type=parse_profile_list,
        default=parse_profile_list(
            "random_ascii"
        ),
    )

    parser.add_argument(
        "--max-covers",
        type=int,
        default=None,
        help=(
            "Limit number of covers for a pilot run."
        ),
    )

    parser.add_argument(
        "--minimum-psnr",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--minimum-ssim",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "experiments/results/"
            "payload_sweep.csv"
        ),
    )

    parser.add_argument(
        "--summary-csv",
        default=(
            "experiments/results/"
            "safe_capacity_summary.csv"
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
    )

    arguments = parser.parse_args()

    if (
        arguments.max_covers is not None
        and arguments.max_covers <= 0
    ):
        parser.error(
            "--max-covers must be positive."
        )

    experiment_passphrase = os.environ.get(
        "GUARDIANPIXEL_EXPERIMENT_PASSPHRASE"
    )

    if not experiment_passphrase:
        raise SystemExit(
            "Set the environment variable "
            "GUARDIANPIXEL_EXPERIMENT_PASSPHRASE "
            "before running the sweep."
        )

    input_directory = Path(
        arguments.input_dir
    )

    if not input_directory.is_dir():
        raise SystemExit(
            f"Input directory not found: "
            f"{input_directory}"
        )

    image_paths = sorted(
        path
        for path in input_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    if arguments.max_covers:
        image_paths = image_paths[
            :arguments.max_covers
        ]

    if not image_paths:
        raise SystemExit(
            "No supported cover images were found."
        )

    feature_lookup = (
        load_cover_feature_lookup(
            arguments.cover_features_csv
        )
    )

    config = load_vision_config(
        arguments.config
    )

    vision_service = (
        GuardianPixelVisionService(
            config=config
        )
    )

    rows = []

    total_experiments = (
        len(image_paths)
        * len(arguments.payload_sizes)
        * len(arguments.profiles)
    )

    experiment_number = 0

    for image_path in image_paths:
        if image_path.name not in feature_lookup:
            print(
                f"Skipping {image_path.name}: "
                "not present in cover-feature CSV."
            )
            continue

        feature_row = feature_lookup[
            image_path.name
        ]

        print("")
        print(
            f"Analysing cover once: "
            f"{image_path.name}"
        )

        cover_rgb, image_info = (
            load_and_validate_image(
                image_path,
                max_pixels=(
                    config.image.max_pixels
                ),
            )
        )

        cover_analysis = (
            vision_service.analyze_cover(
                source=cover_rgb,
                required_payload_bits=0,
                channels_per_selected_pixel=1,
                reserved_position_count=0,
            )
        )

        for profile in arguments.profiles:
            for payload_size in (
                arguments.payload_sizes
            ):
                experiment_number += 1

                print(
                    f"[{experiment_number}/"
                    f"{total_experiments}] "
                    f"{image_path.name} | "
                    f"{profile} | "
                    f"{payload_size} bytes"
                )

                payload = generate_text_payload(
                    requested_bytes=(
                        payload_size
                    ),
                    profile=profile,
                    seed=42,
                )

                payload_features, envelope = (
                    analyse_text_payload(
                        payload=payload,
                        profile=profile,
                    )
                )

                base_row = {
                    "image_id": image_path.name,
                    "cover_category": (
                        feature_row[
                            "cover_category"
                        ]
                    ),
                    "texture_score": float(
                        feature_row[
                            "texture_score"
                        ]
                    ),
                    "width": image_info.width,
                    "height": image_info.height,
                    "total_pixels": (
                        image_info.total_pixels
                    ),
                    "payload_profile": profile,
                    "text_character_count": (
                        payload_features
                        .character_count
                    ),
                    "payload_utf8_bytes": (
                        payload_features
                        .utf8_bytes
                    ),
                    "compressed": (
                        payload_features
                        .compressed
                    ),
                    "stored_payload_bytes": (
                        payload_features
                        .stored_payload_bytes
                    ),
                    "compression_ratio": (
                        payload_features
                        .compression_ratio
                    ),
                    "envelope_bytes": (
                        len(envelope)
                    ),
                    "expected_ciphertext_bits": (
                        payload_features
                        .expected_ciphertext_bits
                    ),
                }

                try:
                    embed_start = perf_counter()

                    embedded = embed_secret(
                        rgb=cover_rgb,
                        payload=payload,
                        payload_type="text",
                        passphrase=(
                            experiment_passphrase
                        ),
                        vision_service=(
                            vision_service
                        ),
                        mime_type="text/plain",
                        precomputed_vision_result=(
                            cover_analysis
                        ),
                    )

                    embedding_seconds = (
                        perf_counter()
                        - embed_start
                    )

                    reloaded_stego = (
                        png_round_trip(
                            embedded.stego_image
                        )
                    )

                    extract_start = perf_counter()

                    extracted = extract_secret(
                        reloaded_stego,
                        experiment_passphrase,
                    )

                    extraction_seconds = (
                        perf_counter()
                        - extract_start
                    )

                    exact_recovery = (
                        extracted
                        .parsed_envelope
                        .payload
                        == payload
                    )

                    packet_bits = (
                        len(
                            embedded
                            .encrypted_packet_bytes
                        )
                        * 8
                    )

                    metrics = (
                        calculate_image_quality(
                            cover=cover_rgb,
                            stego=reloaded_stego,
                            payload_bits=packet_bits,
                            total_embedded_bits=(
                                embedded
                                .total_embedded_bits
                            ),
                        )
                    )

                    (
                        compatible,
                        compatibility_class,
                        failure_reason,
                    ) = (
                        calculate_compatibility_class(
                            exact_recovery=(
                                exact_recovery
                            ),
                            psnr=metrics.psnr,
                            ssim=metrics.ssim,
                            maximum_change=(
                                metrics
                                .maximum_absolute_change
                            ),
                            minimum_psnr=(
                                arguments
                                .minimum_psnr
                            ),
                            minimum_ssim=(
                                arguments
                                .minimum_ssim
                            ),
                        )
                    )

                    row = {
                        **base_row,
                        "encrypted_packet_bits": (
                            packet_bits
                        ),
                        "bootstrap_bits": 480,
                        "metadata_bits": (
                            len(
                                embedded
                                .metadata_bytes
                            )
                            * 8
                        ),
                        "total_embedded_bits": (
                            embedded
                            .total_embedded_bits
                        ),
                        "payload_bpp": (
                            metrics.payload_bpp
                        ),
                        "total_bpp": (
                            metrics.total_bpp
                        ),
                        "selected_blocks": (
                            embedded
                            .block_map
                            .selected_block_count
                        ),
                        "selected_percentage": (
                            embedded
                            .block_map
                            .selected_percentage
                        ),
                        "map_encoding": (
                            embedded
                            .map_encoding
                            .encoding_type
                        ),
                        "map_bytes": (
                            embedded
                            .map_encoding
                            .encoded_byte_count
                        ),
                        "mse": metrics.mse,
                        "psnr": metrics.psnr,
                        "ssim": metrics.ssim,
                        "modified_channels": (
                            metrics
                            .modified_channel_count
                        ),
                        "maximum_change": (
                            metrics
                            .maximum_absolute_change
                        ),
                        "exact_recovery": (
                            exact_recovery
                        ),
                        "ber": (
                            0.0
                            if exact_recovery
                            else ""
                        ),
                        "compatible": compatible,
                        "compatibility_class": (
                            compatibility_class
                        ),
                        "failure_reason": (
                            failure_reason
                        ),
                        "cover_analysis_seconds": (
                            cover_analysis
                            .timings["total"]
                        ),
                        "embedding_seconds": (
                            embedding_seconds
                        ),
                        "extraction_seconds": (
                            extraction_seconds
                        ),
                    }

                except Exception as error:
                    row = {
                        **base_row,
                        "encrypted_packet_bits": (
                            payload_features
                            .expected_ciphertext_bits
                        ),
                        "bootstrap_bits": 480,
                        "metadata_bits": "",
                        "total_embedded_bits": "",
                        "payload_bpp": "",
                        "total_bpp": "",
                        "selected_blocks": "",
                        "selected_percentage": "",
                        "map_encoding": "",
                        "map_bytes": "",
                        "mse": "",
                        "psnr": "",
                        "ssim": "",
                        "modified_channels": "",
                        "maximum_change": "",
                        "exact_recovery": False,
                        "ber": "",
                        "compatible": False,
                        "compatibility_class": (
                            "Incompatible"
                        ),
                        "failure_reason": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                        "cover_analysis_seconds": (
                            cover_analysis
                            .timings["total"]
                        ),
                        "embedding_seconds": "",
                        "extraction_seconds": "",
                    }

                    print(
                        f"  Failed: "
                        f"{row['failure_reason']}"
                    )

                rows.append(row)

    sweep_path = write_csv(
        rows,
        arguments.output_csv,
    )

    summary_rows = build_summary(
        rows
    )

    summary_path = write_csv(
        summary_rows,
        arguments.summary_csv,
    )

    compatible_count = sum(
        row["compatible"] is True
        for row in rows
    )

    print("")
    print("Payload sweep completed.")
    print(
        f"Experiment rows: {len(rows)}"
    )
    print(
        f"Compatible rows: "
        f"{compatible_count}"
    )
    print(
        f"Incompatible/risky rows: "
        f"{len(rows) - compatible_count}"
    )
    print(f"Sweep CSV: {sweep_path}")
    print(
        f"Summary CSV: {summary_path}"
    )


if __name__ == "__main__":
    main()