from pathlib import Path
import os
import json
import mimetypes

from PIL import Image
import numpy as np

from backend.stego.embedder import embed_secret
from backend.stego.extractor import extract_secret
from backend.vision.config import load_vision_config
from backend.vision.preprocessing import load_and_validate_image
from backend.vision.service import GuardianPixelVisionService
from backend.metrics.image_quality import calculate_image_quality


COVERS_DIR = Path("data/div2k/validation")
OUTPUT_DIR = Path("data/div2k/steganalysis_stego")
PAYLOAD_PATH = Path("data/div2k/steganalysis_payload.bin")
CONFIG_PATH = Path("configs/vision.yaml")


def main():
    passphrase = os.environ.get("GUARDIANPIXEL_EXPERIMENT_PASSPHRASE")
    if not passphrase:
        raise SystemExit("GUARDIANPIXEL_EXPERIMENT_PASSPHRASE is not set.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = PAYLOAD_PATH.read_bytes()
    config = load_vision_config(CONFIG_PATH)
    vision_service = GuardianPixelVisionService(config=config)

    covers = sorted(COVERS_DIR.glob("*.png"))

    if len(covers) != 20:
        raise SystemExit(f"Expected 20 cover images, found {len(covers)}.")

    results = []

    for index, cover_path in enumerate(covers, start=1):
        print(f"[{index}/20] Processing {cover_path.name}...")

        cover_rgb, _ = load_and_validate_image(
            cover_path,
            max_pixels=config.image.max_pixels,
        )

        embedded = embed_secret(
            rgb=cover_rgb,
            payload=payload,
            payload_type="file",
            passphrase=passphrase,
            vision_service=vision_service,
            filename=PAYLOAD_PATH.name,
            mime_type="application/octet-stream",
        )

        output_path = OUTPUT_DIR / cover_path.name

        Image.fromarray(embedded.stego_image).save(
            output_path,
            format="PNG",
        )

        reloaded_stego = np.asarray(
            Image.open(output_path).convert("RGB"),
            dtype=np.uint8,
        ).copy()

        verification = extract_secret(
            reloaded_stego,
            passphrase,
        )

        round_trip_verified = (
            verification.parsed_envelope.payload == payload
        )

        metrics = calculate_image_quality(
            cover=cover_rgb,
            stego=reloaded_stego,
            payload_bits=len(embedded.encrypted_packet_bytes) * 8,
            total_embedded_bits=embedded.total_embedded_bits,
        )

        result = {
            "image_id": cover_path.name,
            "payload_bytes": len(payload),
            "round_trip_verified": round_trip_verified,
            "width": int(cover_rgb.shape[1]),
            "height": int(cover_rgb.shape[0]),
            "mse": metrics.mse,
            "psnr": metrics.psnr,
            "ssim": metrics.ssim,
            "payload_bpp": metrics.payload_bpp,
            "total_bpp": metrics.total_bpp,
            "modified_pixels": metrics.modified_pixel_count,
            "maximum_absolute_change": metrics.maximum_absolute_change,
        }

        results.append(result)

        print(
            f"    Verified: {round_trip_verified} | "
            f"PSNR: {metrics.psnr:.4f} dB | "
            f"SSIM: {metrics.ssim:.8f}"
        )

    results_path = OUTPUT_DIR / "embedding_results.json"
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("")
    print("20-image GuardianPixel embedding completed.")
    print(f"Stego images: {OUTPUT_DIR}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
