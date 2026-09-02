"""Serve real multi-cover baseline results and a single-cover case study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify

from backend.api.data_urls import encode_data_url

compare_blueprint = Blueprint("compare", __name__, url_prefix="/api/v1")

CHART_FILES = {
    "psnr": "multi_cover_psnr.png",
    "ssim": "multi_cover_ssim.png",
    "reliability": "multi_cover_reliability.png",
}

DIFFERENCE_FILES = {
    "GuardianPixel": "guardianpixel_difference_x255.png",
    "Sequential Replacement": "sequential_replacement_difference_x255.png",
    "Sequential Matching": "sequential_matching_difference_x255.png",
    "Keyed Random Matching": "keyed_random_difference_x255.png",
    "Canny Synchronous": "canny_synchronous_difference_x255.png",
}

INTEGER_FIELDS = {
    "total_runs",
    "embedding_success_runs",
    "reliable_ber_zero_runs",
    "reliable_runs",
    "width",
    "height",
    "payload_original_bytes",
    "embedded_bits",
    "changed_pixels",
    "requested_payload_bytes",
}

FLOAT_FIELDS = {
    "reliable_success_rate_percent",
    "mean_ber",
    "mean_psnr_all_embedded",
    "mean_ssim_all_embedded",
    "mean_psnr_reliable",
    "mean_ssim_reliable",
    "mean_changed_pixel_ratio",
    "mean_embedding_seconds",
    "mean_extraction_seconds",
    "texture_score",
    "total_bpp",
    "ber",
    "mse",
    "psnr",
    "ssim",
    "changed_pixel_ratio",
    "maximum_change",
    "embedding_seconds",
    "extraction_seconds",
}

BOOLEAN_FIELDS = {
    "embedding_success",
    "extraction_success",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    converted: list[dict] = []
    for row in rows:
        item: dict = {}
        for key, value in row.items():
            if key in INTEGER_FIELDS and value not in (None, ""):
                item[key] = int(float(value))
            elif key in FLOAT_FIELDS and value not in (None, ""):
                item[key] = float(value)
            elif key in BOOLEAN_FIELDS:
                item[key] = str(value).lower() == "true"
            else:
                item[key] = value
        converted.append(item)

    return converted


def _image_data_url(path: Path) -> str | None:
    if not path.is_file():
        return None
    return encode_data_url("image/png", path.read_bytes())


@compare_blueprint.get("/compare")
def get_compare_results():
    """Return real aggregate baseline results and the visual case study."""

    batch_directory = Path(current_app.config["BASELINE_BATCH_DIRECTORY"])
    method_path = batch_directory / "baseline_method_summary.csv"
    category_path = batch_directory / "baseline_category_summary.csv"
    runs_path = batch_directory / "baseline_runs.csv"

    try:
        method_summary = _read_csv(method_path)
        category_summary = _read_csv(category_path)
        runs = _read_csv(runs_path)
    except FileNotFoundError as error:
        return jsonify(
            {
                "code": "BASELINE_RESULT_NOT_FOUND",
                "message": f"Required baseline result is missing: {error}",
            }
        ), 404
    except (OSError, ValueError, csv.Error):
        current_app.logger.exception("Could not parse baseline CSV results.")
        return jsonify(
            {
                "code": "INVALID_BASELINE_RESULT",
                "message": "The baseline result files are invalid.",
            }
        ), 500

    covers = sorted({row["image_id"] for row in runs})
    payload_levels = sorted(
        {int(row["requested_payload_bytes"]) for row in runs}
    )
    methods = sorted({row["method"] for row in runs})

    charts = {
        key: value
        for key, filename in CHART_FILES.items()
        if (value := _image_data_url(batch_directory / filename)) is not None
    }

    single_case_path = Path(current_app.config["BASELINE_RESULT_JSON"])
    single_case = None

    if single_case_path.is_file():
        try:
            single_data = json.loads(single_case_path.read_text(encoding="utf-8"))
            difference_images = {
                method: value
                for method, filename in DIFFERENCE_FILES.items()
                if (
                    value := _image_data_url(single_case_path.parent / filename)
                )
                is not None
            }

            single_case = {
                "cover": single_data.get("cover"),
                "width": single_data.get("width"),
                "height": single_data.get("height"),
                "payloadOriginalBytes": single_data.get("payload_original_bytes"),
                "equalEmbeddedBits": single_data.get("equal_embedded_bits"),
                "equalTotalBpp": single_data.get("equal_total_bpp"),
                "methods": single_data.get("methods", []),
                "differenceImages": difference_images,
                "importantNote": single_data.get("important_note", ""),
            }
        except (OSError, json.JSONDecodeError):
            current_app.logger.warning("Single-cover case study could not be loaded.")

    response = {
        "isSimulated": False,
        "preliminary": True,
        "aggregate": {
            "independentCovers": len(covers),
            "payloadLevels": payload_levels,
            "methodCount": len(methods),
            "methodResultRows": len(runs),
            "methods": methods,
            "methodSummary": method_summary,
            "categorySummary": category_summary,
            "charts": charts,
        },
        "singleCaseStudy": single_case,
        "conclusion": (
            "Across the tested multi-cover batch, GuardianPixel achieved "
            "BER-zero recovery and the highest reliable mean SSIM with a "
            "small visual-quality advantage, while requiring greater sender "
            "time for HED and texture analysis. Canny synchronous results are "
            "not included in reliable rankings when extraction fails."
        ),
        "limitations": (
            "This evaluation covers the selected validation images and payload "
            "levels. Statistical steganalysis and a held-out final test remain "
            "necessary before a general security claim."
        ),
    }

    return jsonify(response), 200
