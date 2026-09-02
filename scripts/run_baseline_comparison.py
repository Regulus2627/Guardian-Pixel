"""Compare GuardianPixel with four LSB baselines at the same total bit count."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from backend.compatibility.payload import generate_text_payload
from backend.core.bitstream import bytes_to_bits
from backend.core.protocol import serialize_bootstrap
from backend.metrics.image_quality import calculate_image_quality
from backend.stego.baselines.canny_edge import embed_canny_lsb, extract_canny_lsb
from backend.stego.baselines.keyed_random_matching import (
    embed_keyed_random_matching,
    extract_keyed_random_matching,
)
from backend.stego.baselines.sequential_matching import (
    embed_matching_lsb,
    extract_matching_lsb,
)
from backend.stego.baselines.sequential_replacement import (
    embed_sequential_lsb,
    extract_sequential_lsb,
)
from backend.stego.embedder import embed_secret
from backend.stego.extractor import extract_secret
from backend.vision.config import load_vision_config
from backend.vision.preprocessing import load_and_validate_image
from backend.vision.service import GuardianPixelVisionService


BASELINE_KEY = hashlib.sha256(
    b"GuardianPixel-fair-baseline-position-key"
).digest()


def save_difference_image(
    cover: np.ndarray,
    stego: np.ndarray,
    output_path: Path,
) -> None:
    """Save an amplified RGB absolute-difference image."""

    difference = np.abs(
        stego.astype(np.int16) - cover.astype(np.int16)
    )

    amplified = np.clip(
        difference * 255,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(amplified).save(output_path)


def calculate_ber(
    expected: np.ndarray,
    extracted: np.ndarray,
) -> float:
    """Calculate bit error rate."""

    if expected.size == 0:
        return 0.0

    if extracted.size != expected.size:
        return 1.0

    return float(
        np.mean(expected != extracted)
    )


def write_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def create_metric_charts(rows: list[dict], output_directory: Path) -> None:
    successful = [
        row
        for row in rows
        if row["embedding_success"] is True and row["psnr"] != ""
    ]

    if not successful:
        return

    methods = [row["method"] for row in successful]
    psnr_values = [float(row["psnr"]) for row in successful]
    ssim_values = [float(row["ssim"]) for row in successful]

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(methods, psnr_values, color="#14b8a6")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("PSNR at Equal Total Embedded Bits")
    axis.tick_params(axis="x", rotation=22)
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, psnr_values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(
        output_directory / "psnr_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(methods, ssim_values, color="#6366f1")
    axis.set_ylabel("SSIM")
    axis.set_ylim(min(ssim_values) - 0.0001, 1.00001)
    axis.set_title("SSIM at Equal Total Embedded Bits")
    axis.tick_params(axis="x", rotation=22)
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, ssim_values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(
        output_directory / "ssim_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GuardianPixel with four LSB baselines."
    )
    parser.add_argument("--cover", required=True)
    parser.add_argument("--payload-bytes", type=int, default=10_000)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/vision.yaml")
    parser.add_argument(
        "--output",
        default="experiments/results/baseline_comparison",
    )
    arguments = parser.parse_args()

    if arguments.payload_bytes <= 0:
        parser.error("--payload-bytes must be positive.")

    if arguments.channel not in {0, 1, 2}:
        parser.error("--channel must be 0, 1 or 2.")

    passphrase = os.environ.get("GUARDIANPIXEL_EXPERIMENT_PASSPHRASE")
    if not passphrase:
        raise SystemExit(
            "Set GUARDIANPIXEL_EXPERIMENT_PASSPHRASE before running."
        )

    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    config = load_vision_config(arguments.config)
    cover, image_info = load_and_validate_image(
        arguments.cover,
        max_pixels=config.image.max_pixels,
    )

    Image.fromarray(cover).save(output_directory / "original_cover.png")

    payload = generate_text_payload(
        requested_bytes=arguments.payload_bytes,
        profile="random_ascii",
        seed=arguments.seed,
    )

    vision_service = GuardianPixelVisionService(config=config)

    cover_analysis = vision_service.analyze_cover(
        source=cover,
        required_payload_bits=0,
        channels_per_selected_pixel=1,
        reserved_position_count=0,
    )

    proposed_start = perf_counter()
    proposed = embed_secret(
        rgb=cover,
        payload=payload,
        payload_type="text",
        passphrase=passphrase,
        vision_service=vision_service,
        mime_type="text/plain",
        precomputed_vision_result=cover_analysis,
    )
    proposed_embed_seconds = perf_counter() - proposed_start

    proposed_extract_start = perf_counter()
    proposed_extracted = extract_secret(
        proposed.stego_image,
        passphrase,
    )
    proposed_extract_seconds = perf_counter() - proposed_extract_start

    proposed_recovered = (
        proposed_extracted.parsed_envelope.payload == payload
    )

    # Use the exact complete GuardianPixel bit count for every baseline.
    # The stream contains bootstrap, metadata and encrypted ciphertext bytes.
    comparison_bytes = (
        serialize_bootstrap(proposed.bootstrap)
        + proposed.metadata_bytes
        + proposed.encrypted_packet_bytes
    )
    comparison_bits = bytes_to_bits(comparison_bytes)

    if comparison_bits.size != proposed.total_embedded_bits:
        raise RuntimeError(
            "Comparison bit count does not match GuardianPixel total bits."
        )

    total_pixels = image_info.total_pixels
    total_bpp = float(comparison_bits.size / total_pixels)

    if comparison_bits.size > total_pixels:
        raise RuntimeError(
            "The selected cover is too small for one-channel baseline comparison."
        )

    rows: list[dict] = []

    proposed_metrics = calculate_image_quality(
        cover=cover,
        stego=proposed.stego_image,
        payload_bits=len(proposed.encrypted_packet_bytes) * 8,
        total_embedded_bits=proposed.total_embedded_bits,
    )

    Image.fromarray(proposed.stego_image).save(
        output_directory / "guardianpixel_stego.png"
    )
    save_difference_image(
        cover,
        proposed.stego_image,
        output_directory / "guardianpixel_difference_x255.png",
    )

    rows.append(
        {
            "method": "GuardianPixel",
            "image_id": Path(arguments.cover).name,
            "width": image_info.width,
            "height": image_info.height,
            "payload_original_bytes": len(payload),
            "embedded_bits": int(comparison_bits.size),
            "total_bpp": total_bpp,
            "embedding_success": True,
            "extraction_success": proposed_recovered,
            "ber": 0.0 if proposed_recovered else 1.0,
            "mse": proposed_metrics.mse,
            "psnr": proposed_metrics.psnr,
            "ssim": proposed_metrics.ssim,
            "changed_pixels": proposed_metrics.modified_pixel_count,
            "changed_pixel_ratio": (
                proposed_metrics.modified_pixel_count / total_pixels * 100.0
            ),
            "maximum_change": proposed_metrics.maximum_absolute_change,
            "embedding_seconds": proposed_embed_seconds,
            "extraction_seconds": proposed_extract_seconds,
            "failure_reason": "",
        }
    )

    def run_baseline(
        method_name: str,
        embed_function,
        extract_function,
        output_stem: str,
    ) -> None:
        try:
            embed_start = perf_counter()
            stego = embed_function()
            embed_seconds = perf_counter() - embed_start

            extract_start = perf_counter()
            extracted = extract_function(stego)
            extract_seconds = perf_counter() - extract_start

            ber = calculate_ber(comparison_bits, extracted)
            extraction_success = bool(ber == 0.0)

            metrics = calculate_image_quality(
                cover=cover,
                stego=stego,
                payload_bits=int(comparison_bits.size),
                total_embedded_bits=int(comparison_bits.size),
            )

            Image.fromarray(stego).save(
                output_directory / f"{output_stem}_stego.png"
            )
            save_difference_image(
                cover,
                stego,
                output_directory / f"{output_stem}_difference_x255.png",
            )

            rows.append(
                {
                    "method": method_name,
                    "image_id": Path(arguments.cover).name,
                    "width": image_info.width,
                    "height": image_info.height,
                    "payload_original_bytes": len(payload),
                    "embedded_bits": int(comparison_bits.size),
                    "total_bpp": total_bpp,
                    "embedding_success": True,
                    "extraction_success": extraction_success,
                    "ber": ber,
                    "mse": metrics.mse,
                    "psnr": metrics.psnr,
                    "ssim": metrics.ssim,
                    "changed_pixels": metrics.modified_pixel_count,
                    "changed_pixel_ratio": (
                        metrics.modified_pixel_count / total_pixels * 100.0
                    ),
                    "maximum_change": metrics.maximum_absolute_change,
                    "embedding_seconds": embed_seconds,
                    "extraction_seconds": extract_seconds,
                    "failure_reason": "" if extraction_success else "BIT_ERRORS",
                }
            )

        except Exception as error:
            rows.append(
                {
                    "method": method_name,
                    "image_id": Path(arguments.cover).name,
                    "width": image_info.width,
                    "height": image_info.height,
                    "payload_original_bytes": len(payload),
                    "embedded_bits": int(comparison_bits.size),
                    "total_bpp": total_bpp,
                    "embedding_success": False,
                    "extraction_success": False,
                    "ber": "",
                    "mse": "",
                    "psnr": "",
                    "ssim": "",
                    "changed_pixels": "",
                    "changed_pixel_ratio": "",
                    "maximum_change": "",
                    "embedding_seconds": "",
                    "extraction_seconds": "",
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )

    run_baseline(
        "Sequential Replacement",
        lambda: embed_sequential_lsb(
            cover,
            comparison_bits,
            channel=arguments.channel,
        ),
        lambda stego: extract_sequential_lsb(
            stego,
            int(comparison_bits.size),
            channel=arguments.channel,
        ),
        "sequential_replacement",
    )

    run_baseline(
        "Sequential Matching",
        lambda: embed_matching_lsb(
            cover,
            comparison_bits,
            channel=arguments.channel,
        ),
        lambda stego: extract_matching_lsb(
            stego,
            int(comparison_bits.size),
            channel=arguments.channel,
        ),
        "sequential_matching",
    )

    run_baseline(
        "Keyed Random Matching",
        lambda: embed_keyed_random_matching(
            cover,
            comparison_bits,
            key=BASELINE_KEY,
            channel=arguments.channel,
        ),
        lambda stego: extract_keyed_random_matching(
            stego,
            int(comparison_bits.size),
            key=BASELINE_KEY,
            channel=arguments.channel,
        ),
        "keyed_random",
    )

    run_baseline(
        "Canny Synchronous",
        lambda: embed_canny_lsb(
            cover,
            comparison_bits,
            channel=arguments.channel,
        ),
        lambda stego: extract_canny_lsb(
            stego,
            int(comparison_bits.size),
            channel=arguments.channel,
        ),
        "canny_synchronous",
    )

    csv_path = output_directory / "baseline_comparison.csv"
    write_rows(rows, csv_path)
    create_metric_charts(rows, output_directory)

    summary = {
        "cover": Path(arguments.cover).name,
        "width": image_info.width,
        "height": image_info.height,
        "payload_original_bytes": len(payload),
        "equal_embedded_bits": int(comparison_bits.size),
        "equal_total_bpp": total_bpp,
        "methods": rows,
        "important_note": (
            "Canny Synchronous recalculates edges from the stego image and may "
            "fail due to position desynchronization. It must be reported separately."
        ),
    }

    (output_directory / "baseline_comparison.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("")
    print("Equal-bit baseline comparison completed.")
    print(f"Cover: {Path(arguments.cover).name}")
    print(f"Payload: {len(payload):,} original bytes")
    print(f"Equal embedded bits: {comparison_bits.size:,}")
    print(f"Equal total BPP: {total_bpp:.6f}")
    print("")

    for row in rows:
        print(
            f"{row['method']}: "
            f"success={row['extraction_success']}, "
            f"BER={row['ber']}, "
            f"PSNR={row['psnr']}, "
            f"SSIM={row['ssim']}"
        )

    print(f"Results: {output_directory}")


if __name__ == "__main__":
    main()
