"""Generate a GuardianPixel HED edge map from a real image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.vision.config import load_vision_config
from backend.vision.hed.inference import HEDInference
from backend.vision.hed.model import load_hed_network
from backend.vision.preprocessing import (
    load_and_validate_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a pretrained HED edge map."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the cover image.",
    )

    parser.add_argument(
        "--output",
        default="sample_data/vision_outputs/hed_demo",
        help="Output directory.",
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
        help="Vision configuration file.",
    )

    arguments = parser.parse_args()

    config = load_vision_config(arguments.config)

    rgb, image_info = load_and_validate_image(
        arguments.image,
        max_pixels=config.image.max_pixels,
    )

    network = load_hed_network(
        prototxt_path=config.hed.prototxt_path,
        weights_path=config.hed.weights_path,
        device=config.hed.resolved_device,
    )

    inference = HEDInference(
        network=network,
        mean_bgr=config.hed.mean_bgr,
    )

    edge_map = inference.predict(rgb)

    output_directory = Path(arguments.output)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_uint8 = np.rint(
        edge_map * 255
    ).astype(np.uint8)

    Image.fromarray(edge_uint8).save(
        output_directory / "hed_map.png"
    )

    Image.fromarray(rgb).save(
        output_directory / "validated_cover.png"
    )

    analysis = {
        "width": image_info.width,
        "height": image_info.height,
        "original_format": image_info.original_format,
        "model": "HED",
        "framework": config.hed.framework,
        "device": config.hed.resolved_device,
        "inference_seconds": (
            inference.last_inference_seconds
        ),
        "edge_min": float(edge_map.min()),
        "edge_max": float(edge_map.max()),
        "edge_mean": float(edge_map.mean()),
    }

    analysis_path = (
        output_directory / "hed_analysis.json"
    )

    with analysis_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            analysis,
            file,
            indent=2,
        )

    print("HED inference completed.")
    print(
        f"Image size: "
        f"{image_info.width}x{image_info.height}"
    )
    print(
        f"Inference time: "
        f"{inference.last_inference_seconds:.4f} seconds"
    )
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()