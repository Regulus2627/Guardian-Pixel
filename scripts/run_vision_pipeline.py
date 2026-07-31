"""Run the complete GuardianPixel Member 1 vision pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.vision.config import (
    load_vision_config,
)
from backend.vision.preprocessing import (
    load_and_validate_image,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)
from backend.vision.visualization import (
    save_binary_block_map,
    save_color_heatmap,
    save_grayscale_map,
    save_selected_blocks_overlay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete GuardianPixel "
            "sender-side vision pipeline."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the cover image.",
    )

    parser.add_argument(
        "--required-bits",
        type=int,
        required=True,
        help=(
            "Required encrypted-payload bits. "
            "Member 2 will eventually provide this value."
        ),
    )

    parser.add_argument(
        "--channels-per-pixel",
        type=int,
        default=1,
        help=(
            "Number of eligible colour channels "
            "per selected pixel."
        ),
    )

    parser.add_argument(
        "--reserved-positions",
        type=int,
        default=0,
        help=(
            "Positions reserved for bootstrap "
            "and metadata."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
        help="Path to the vision configuration.",
    )

    parser.add_argument(
        "--output",
        default=(
            "sample_data/vision_outputs/"
            "complete_vision_run"
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Save numerical maps as NPY files.",
    )

    arguments = parser.parse_args()

    if arguments.required_bits < 0:
        parser.error(
            "--required-bits cannot be negative."
        )

    if arguments.channels_per_pixel <= 0:
        parser.error(
            "--channels-per-pixel must be positive."
        )

    if arguments.reserved_positions < 0:
        parser.error(
            "--reserved-positions cannot be negative."
        )

    config = load_vision_config(
        arguments.config
    )

    service = GuardianPixelVisionService(
        config=config
    )

    result = service.analyze_cover(
        source=arguments.image,
        required_payload_bits=(
            arguments.required_bits
        ),
        channels_per_selected_pixel=(
            arguments.channels_per_pixel
        ),
        reserved_position_count=(
            arguments.reserved_positions
        ),
    )

    # Load the validated RGB cover again for visual output.
    rgb, _ = load_and_validate_image(
        arguments.image,
        max_pixels=config.image.max_pixels,
    )

    output_directory = Path(
        arguments.output
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(rgb).save(
        output_directory
        / "validated_cover.png"
    )

    save_grayscale_map(
        result.feature_maps.hed_map,
        output_directory
        / "hed_map.png",
    )

    save_grayscale_map(
        result.feature_maps.entropy_map,
        output_directory
        / "entropy_map.png",
    )

    save_grayscale_map(
        result.feature_maps.variance_map,
        output_directory
        / "variance_map.png",
    )

    save_grayscale_map(
        result.feature_maps.fused_heatmap,
        output_directory
        / "fused_heatmap_grayscale.png",
    )

    save_color_heatmap(
        result.feature_maps.fused_heatmap,
        output_directory
        / "fused_heatmap_color.png",
    )

    save_binary_block_map(
        result.block_map.selected_blocks,
        output_directory
        / "binary_block_map.png",
    )

    save_selected_blocks_overlay(
        rgb=rgb,
        selected_blocks=(
            result.block_map
            .selected_blocks
        ),
        block_size=(
            result.block_map.block_size
        ),
        output_path=(
            output_directory
            / "selected_blocks_overlay.png"
        ),
        alpha=(
            config.visualization
            .overlay_alpha
        ),
    )

    with (
        output_directory
        / "encoded_block_map.bin"
    ).open("wb") as file:
        file.write(
            result.map_encoding
            .encoded_data
        )

    if arguments.save_arrays:
        np.save(
            output_directory
            / "hed_map.npy",
            result.feature_maps.hed_map,
        )

        np.save(
            output_directory
            / "entropy_map.npy",
            result.feature_maps
            .entropy_map,
        )

        np.save(
            output_directory
            / "variance_map.npy",
            result.feature_maps
            .variance_map,
        )

        np.save(
            output_directory
            / "fused_heatmap.npy",
            result.feature_maps
            .fused_heatmap,
        )

        np.save(
            output_directory
            / "block_scores.npy",
            result.block_map.block_scores,
        )

    analysis = {
        "image": {
            "width": result.image_info.width,
            "height": result.image_info.height,
            "total_pixels": (
                result.image_info.total_pixels
            ),
            "channels": (
                result.image_info.channels
            ),
            "original_format": (
                result.image_info
                .original_format
            ),
            "original_mode": (
                result.image_info
                .original_mode
            ),
        },
        "payload": {
            "required_bits": (
                result.block_map
                .required_payload_bits
            ),
            "required_bytes_approximate": (
                result.block_map
                .required_payload_bits
                / 8.0
            ),
            "channels_per_selected_pixel": (
                arguments.channels_per_pixel
            ),
            "reserved_positions": (
                arguments.reserved_positions
            ),
            "safety_target_positions": (
                result.block_map
                .target_position_count
            ),
            "selected_positions": (
                result.block_map
                .selected_position_count
            ),
            "usable_positions": (
                result.block_map
                .usable_position_count
            ),
        },
        "block_map": {
            "block_size": (
                result.block_map.block_size
            ),
            "block_rows": (
                result.block_map.block_rows
            ),
            "block_columns": (
                result.block_map
                .block_columns
            ),
            "total_blocks": (
                result.block_map
                .total_block_count
            ),
            "selected_blocks": (
                result.block_map
                .selected_block_count
            ),
            "selected_percentage": (
                result.block_map
                .selected_percentage
            ),
        },
        "encoding": {
            "type": (
                result.map_encoding
                .encoding_type
            ),
            "original_bit_count": (
                result.map_encoding
                .original_bit_count
            ),
            "raw_bytes": (
                result.map_encoding
                .raw_byte_count
            ),
            "encoded_bytes": (
                result.map_encoding
                .encoded_byte_count
            ),
            "compression_ratio": (
                result.map_encoding
                .compression_ratio
            ),
        },
        "map_statistics": {
            "hed_mean": float(
                result.feature_maps
                .hed_map.mean()
            ),
            "entropy_mean": float(
                result.feature_maps
                .entropy_map.mean()
            ),
            "variance_mean": float(
                result.feature_maps
                .variance_map.mean()
            ),
            "fused_mean": float(
                result.feature_maps
                .fused_heatmap.mean()
            ),
            "fused_min": float(
                result.feature_maps
                .fused_heatmap.min()
            ),
            "fused_max": float(
                result.feature_maps
                .fused_heatmap.max()
            ),
        },
        "timings_seconds": (
            result.timings
        ),
        "configuration": (
            result.config_used
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

    print("")
    print("GuardianPixel vision analysis completed.")
    print(
        f"Image: "
        f"{result.image_info.width}x"
        f"{result.image_info.height}"
    )
    print(
        f"Required payload: "
        f"{result.block_map.required_payload_bits} bits"
    )
    print(
        f"Blocks selected: "
        f"{result.block_map.selected_block_count}/"
        f"{result.block_map.total_block_count}"
    )
    print(
        f"Selected percentage: "
        f"{result.block_map.selected_percentage:.2f}%"
    )
    print(
        f"Usable positions: "
        f"{result.block_map.usable_position_count}"
    )
    print(
        f"Map encoding: "
        f"{result.map_encoding.encoding_type}"
    )
    print(
        f"Map size: "
        f"{result.map_encoding.raw_byte_count} → "
        f"{result.map_encoding.encoded_byte_count} bytes"
    )
    print(
        f"HED tile count: "
        f"{result.config_used['hed_tile_count']}"
    )
    print(
        f"Total analysis time: "
        f"{result.timings['total']:.4f} seconds"
    )
    print(
        f"Outputs: {output_directory}"
    )


if __name__ == "__main__":
    main()