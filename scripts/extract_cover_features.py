"""Extract and categorize features from a folder of cover images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.compatibility.categories import (
    fit_category_thresholds,
    load_category_thresholds,
    save_category_thresholds,
    write_categorized_features_csv,
)
from backend.compatibility.features import (
    extract_cover_features,
)
from backend.vision.config import (
    load_vision_config,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract GuardianPixel cover features "
            "and calculate or apply texture categories."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing cover images.",
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "experiments/results/"
            "cover_features.csv"
        ),
    )

    parser.add_argument(
        "--thresholds-json",
        default=(
            "experiments/results/"
            "category_thresholds.json"
        ),
        help=(
            "Path used to save newly fitted "
            "training thresholds."
        ),
    )

    parser.add_argument(
        "--load-thresholds",
        default=None,
        help=(
            "Load fixed category thresholds "
            "instead of fitting new thresholds."
        ),
    )

    parser.add_argument(
        "--failures-json",
        default=(
            "experiments/results/"
            "cover_feature_failures.json"
        ),
    )

    parser.add_argument(
        "--probe-bits",
        type=int,
        default=0,
        help=(
            "Optional payload bits used only for "
            "block-selection statistics. Use 0 "
            "for payload-independent cover analysis."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/vision.yaml",
    )

    arguments = parser.parse_args()

    if arguments.probe_bits < 0:
        parser.error(
            "--probe-bits cannot be negative."
        )

    input_directory = Path(
        arguments.input_dir
    )

    if not input_directory.is_dir():
        raise SystemExit(
            f"Input directory was not found: "
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

    if not image_paths:
        raise SystemExit(
            "No supported images were found."
        )

    if (
        not arguments.load_thresholds
        and len(image_paths) < 4
    ):
        raise SystemExit(
            "At least four images are required "
            "when fitting new thresholds."
        )

    config = load_vision_config(
        arguments.config
    )

    service = GuardianPixelVisionService(
        config=config
    )

    features = []
    failures = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print(
            f"[{index}/{len(image_paths)}] "
            f"Analysing {image_path.name}"
        )

        try:
            result = service.analyze_cover(
                source=image_path,
                required_payload_bits=(
                    arguments.probe_bits
                ),
                channels_per_selected_pixel=1,
                reserved_position_count=0,
            )

            cover_features = (
                extract_cover_features(
                    image_id=image_path.name,
                    vision_result=result,
                )
            )

            features.append(
                cover_features
            )

        except Exception as error:
            failure = {
                "image": image_path.name,
                "error_type": (
                    type(error).__name__
                ),
                "error": str(error),
            }

            failures.append(failure)

            print(
                f"  Failed: "
                f"{failure['error_type']}: "
                f"{failure['error']}"
            )

    if not features:
        raise SystemExit(
            "No cover images were processed successfully."
        )

    if arguments.load_thresholds:
        thresholds = (
            load_category_thresholds(
                arguments.load_thresholds
            )
        )

        thresholds_path = Path(
            arguments.load_thresholds
        )

        threshold_mode = "loaded"

    else:
        if len(features) < 4:
            raise SystemExit(
                "Fewer than four covers were "
                "processed successfully."
            )

        thresholds = (
            fit_category_thresholds(
                features
            )
        )

        thresholds_path = (
            save_category_thresholds(
                thresholds,
                arguments.thresholds_json,
            )
        )

        threshold_mode = "fitted"

    csv_path = (
        write_categorized_features_csv(
            features=features,
            thresholds=thresholds,
            output_path=(
                arguments.output_csv
            ),
        )
    )

    failure_path = Path(
        arguments.failures_json
    )

    failure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_path.write_text(
        json.dumps(
            failures,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("Cover feature extraction completed.")
    print(
        f"Successful covers: {len(features)}"
    )
    print(
        f"Failed covers: {len(failures)}"
    )
    print(
        f"Threshold mode: {threshold_mode}"
    )
    print(
        f"Smooth upper score: "
        f"{thresholds.smooth_upper:.6f}"
    )
    print(
        f"Textured lower score: "
        f"{thresholds.textured_lower:.6f}"
    )
    print(f"Features CSV: {csv_path}")
    print(
        f"Thresholds JSON: "
        f"{thresholds_path}"
    )
    print(
        f"Failures JSON: {failure_path}"
    )


if __name__ == "__main__":
    main()