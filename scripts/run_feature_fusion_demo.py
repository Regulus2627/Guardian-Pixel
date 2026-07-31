"""Generate HED, entropy, variance and fused maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from backend.vision.config import load_vision_config
from backend.vision.entropy_map import (
    calculate_local_entropy,
)
from backend.vision.fusion import fuse_feature_maps
from backend.vision.hed.inference import HEDInference
from backend.vision.hed.model import load_hed_network
from backend.vision.preprocessing import (
    load_and_validate_image,
    rgb_to_luminance,
)
from backend.vision.variance_map import (
    calculate_local_variance,
)
from backend.vision.visualization import (
    save_color_heatmap,
    save_grayscale_map,
)
from backend.vision.hed.tiling import (
    TiledHEDInference,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GuardianPixel feature maps "
            "and the fused suitability heatmap."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the cover image.",
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
        help="Path to vision configuration.",
    )

    parser.add_argument(
        "--output",
        default=(
            "sample_data/vision_outputs/"
            "feature_fusion_demo"
        ),
        help="Output directory.",
    )

    arguments = parser.parse_args()

    config = load_vision_config(
        arguments.config
    )

    output_directory = Path(
        arguments.output
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb, image_info = load_and_validate_image(
        arguments.image,
        max_pixels=config.image.max_pixels,
    )

    luminance = rgb_to_luminance(rgb)

    entropy_start = perf_counter()

    raw_entropy = calculate_local_entropy(
        luminance,
        window_size=(
            config.texture.entropy_window
        ),
        bins=config.texture.entropy_bins,
    )

    entropy_seconds = (
        perf_counter() - entropy_start
    )

    variance_start = perf_counter()

    raw_variance = calculate_local_variance(
        luminance,
        window_size=(
            config.texture.variance_window
        ),
    )

    variance_seconds = (
        perf_counter() - variance_start
    )

    network = load_hed_network(
        prototxt_path=(
            config.hed.prototxt_path
        ),
        weights_path=config.hed.weights_path,
        device=config.hed.resolved_device,
    )

    hed_inference = HEDInference(
        network=network,
        mean_bgr=config.hed.mean_bgr,
    )

    # raw_hed = hed_inference.predict(rgb)
    tiled_hed = TiledHEDInference(
        base_inference=hed_inference,
        tile_size=config.hed.tile_size,
        overlap=config.hed.overlap,
    )

    raw_hed = tiled_hed.predict(rgb)


    fusion_start = perf_counter()

    feature_maps = fuse_feature_maps(
        hed_map=raw_hed,
        entropy_map=raw_entropy,
        variance_map=raw_variance,
        hed_weight=(
            config.fusion.hed_weight
        ),
        entropy_weight=(
            config.fusion.entropy_weight
        ),
        variance_weight=(
            config.fusion.variance_weight
        ),
        median_kernel=(
            config.fusion.median_kernel
        ),
    )

    fusion_seconds = (
        perf_counter() - fusion_start
    )

    Image.fromarray(rgb).save(
        output_directory
        / "validated_cover.png"
    )

    Image.fromarray(luminance).save(
        output_directory
        / "luminance.png"
    )

    save_grayscale_map(
        feature_maps.hed_map,
        output_directory
        / "hed_map.png",
    )

    save_grayscale_map(
        feature_maps.entropy_map,
        output_directory
        / "entropy_map.png",
    )

    save_grayscale_map(
        feature_maps.variance_map,
        output_directory
        / "variance_map.png",
    )

    save_grayscale_map(
        feature_maps.fused_heatmap,
        output_directory
        / "fused_heatmap_grayscale.png",
    )

    save_color_heatmap(
        feature_maps.fused_heatmap,
        output_directory
        / "fused_heatmap_color.png",
    )

    np.save(
        output_directory / "hed_map.npy",
        feature_maps.hed_map,
    )

    np.save(
        output_directory / "entropy_map.npy",
        feature_maps.entropy_map,
    )

    np.save(
        output_directory / "variance_map.npy",
        feature_maps.variance_map,
    )

    np.save(
        output_directory / "fused_heatmap.npy",
        feature_maps.fused_heatmap,
    )

    analysis = {
        "width": image_info.width,
        "height": image_info.height,
        "original_format": (
            image_info.original_format
        ),
        "fusion_weights": {
            "hed": config.fusion.hed_weight,
            "entropy": (
                config.fusion.entropy_weight
            ),
            "variance": (
                config.fusion.variance_weight
            ),
        },
        "timings_seconds": {
            "hed": (
            tiled_hed.last_inference_seconds
            ),
            "entropy": entropy_seconds,
            "variance": variance_seconds,
            "fusion": fusion_seconds,
        },
        "map_means": {
            "hed": float(
                feature_maps.hed_map.mean()
            ),
            "entropy": float(
                feature_maps.entropy_map.mean()
            ),
            "variance": float(
                feature_maps.variance_map.mean()
            ),
            "fused": float(
                feature_maps
                .fused_heatmap.mean()
            ),
        },
            "hed_tile_count": (
            tiled_hed.last_tile_count
        ),
    }

    with (
        output_directory / "analysis.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            analysis,
            file,
            indent=2,
        )

    print("Feature fusion completed.")
    print(
        f"Image size: "
        f"{image_info.width}x"
        f"{image_info.height}"
    )
    print(
        f"HED time: "
        f"{tiled_hed.last_inference_seconds:.4f}s"
    )
    print(
        f"Entropy time: "
        f"{entropy_seconds:.4f}s"
    )
    print(
        f"Variance time: "
        f"{variance_seconds:.4f}s"
    )
    print(
        f"Fusion time: "
        f"{fusion_seconds:.4f}s"
    )
    print(
        f"Outputs: {output_directory}"
    )
    print(
    f"HED tile count: "
    f"{tiled_hed.last_tile_count}"
)


if __name__ == "__main__":
    main()