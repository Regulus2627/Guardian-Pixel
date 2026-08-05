"""Generate a capacity-aware block map from a fused heatmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.vision.block_map import (
    generate_capacity_aware_block_map,
)
from backend.vision.config import (
    load_vision_config,
)
from backend.vision.map_encoding import (
    encode_block_map,
)
from backend.vision.preprocessing import (
    load_and_validate_image,
)
from backend.vision.visualization import (
    save_binary_block_map,
    save_selected_blocks_overlay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GuardianPixel binary "
            "block-map outputs."
        )
    )

    parser.add_argument(
        "--cover",
        required=True,
        help="Path to the original cover image.",
    )

    parser.add_argument(
        "--heatmap",
        required=True,
        help="Path to fused_heatmap.npy.",
    )

    parser.add_argument(
        "--required-bits",
        type=int,
        required=True,
        help="Required payload bits for simulation.",
    )

    parser.add_argument(
        "--channels-per-pixel",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--reserved-positions",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
    )

    parser.add_argument(
        "--output",
        default=(
            "sample_data/vision_outputs/"
            "block_map_demo"
        ),
    )

    arguments = parser.parse_args()

    config = load_vision_config(
        arguments.config
    )

    rgb, image_info = (
        load_and_validate_image(
            arguments.cover,
            max_pixels=(
                config.image.max_pixels
            ),
        )
    )

    heatmap = np.load(
        arguments.heatmap
    ).astype(np.float32)

    if heatmap.shape != rgb.shape[:2]:
        raise ValueError(
            "Heatmap and cover dimensions do not match."
        )

    result = (
        generate_capacity_aware_block_map(
            heatmap=heatmap,
            required_payload_bits=(
                arguments.required_bits
            ),
            block_size=(
                config.block_map.block_size
            ),
            channels_per_selected_pixel=(
                arguments
                .channels_per_pixel
            ),
            reserved_position_count=(
                arguments
                .reserved_positions
            ),
            safety_margin=(
                config.block_map.safety_margin
            ),
            score_method=(
                config.block_map.score_method
            ),
        )
    )

    encoding = encode_block_map(
        result.selected_blocks
    )

    output_directory = Path(
        arguments.output
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_binary_block_map(
        result.selected_blocks,
        output_directory
        / "binary_block_map.png",
    )

    save_selected_blocks_overlay(
        rgb=rgb,
        selected_blocks=(
            result.selected_blocks
        ),
        block_size=result.block_size,
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
            encoding.encoded_data
        )

    analysis = {
        "image_width": image_info.width,
        "image_height": image_info.height,
        "block_size": result.block_size,
        "block_rows": result.block_rows,
        "block_columns": (
            result.block_columns
        ),
        "total_blocks": (
            result.total_block_count
        ),
        "selected_blocks": (
            result.selected_block_count
        ),
        "selected_percentage": (
            result.selected_percentage
        ),
        "required_payload_bits": (
            result.required_payload_bits
        ),
        "target_position_count": (
            result.target_position_count
        ),
        "selected_position_count": (
            result.selected_position_count
        ),
        "usable_position_count": (
            result.usable_position_count
        ),
        "encoding_type": (
            encoding.encoding_type
        ),
        "raw_map_bytes": (
            encoding.raw_byte_count
        ),
        "encoded_map_bytes": (
            encoding.encoded_byte_count
        ),
        "compression_ratio": (
            encoding.compression_ratio
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

    print("Block-map generation completed.")
    print(
        f"Blocks: "
        f"{result.block_rows}x"
        f"{result.block_columns}"
    )
    print(
        f"Selected: "
        f"{result.selected_block_count}/"
        f"{result.total_block_count}"
    )
    print(
        f"Usable positions: "
        f"{result.usable_position_count}"
    )
    print(
        f"Encoding: "
        f"{encoding.encoding_type}"
    )
    print(
        f"Map size: "
        f"{encoding.raw_byte_count} → "
        f"{encoding.encoded_byte_count} bytes"
    )


if __name__ == "__main__":
    main()