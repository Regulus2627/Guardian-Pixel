"""Serve real held-out GuardianPixel research results."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from flask import Blueprint, current_app, jsonify


research_blueprint = Blueprint(
    "research",
    __name__,
    url_prefix="/api/v1",
)


INTEGER_FIELDS = {
    "requested_payload_bytes",
    "width",
    "height",
    "payload_original_bytes",
    "embedded_bits",
    "changed_pixels",
    "maximum_change",
    "total_runs",
    "embedding_success_runs",
    "reliable_ber_zero_runs",
    "reliable_runs",
}

BOOLEAN_FIELDS = {
    "embedding_success",
    "extraction_success",
}

FLOAT_FIELDS = {
    "texture_score",
    "total_bpp",
    "ber",
    "mse",
    "psnr",
    "ssim",
    "changed_pixel_ratio",
    "embedding_seconds",
    "extraction_seconds",
    "histogram_l1_mean",
    "histogram_chi_square_mean",
    "cover_lsb_chi_square_statistic",
    "stego_lsb_chi_square_statistic",
    "lsb_chi_square_statistic_change",
    "absolute_lsb_chi_square_statistic_change",
    "cover_rs_combined_imbalance",
    "stego_rs_combined_imbalance",
    "rs_combined_imbalance_change",
    "absolute_rs_combined_imbalance_change",
    "smooth_pixel_fraction",
    "smooth_region_change_ratio",
    "mean_modified_position_suitability",
    "mean_image_suitability",
    "suitability_gain",
    "reliable_success_rate_percent",
    "mean_ber",
    "mean_psnr_all_embedded",
    "mean_ssim_all_embedded",
    "mean_psnr_reliable",
    "mean_ssim_reliable",
    "mean_changed_pixel_ratio",
    "mean_embedding_seconds",
    "mean_extraction_seconds",
}

RELIABLE_METHODS = {
    "GuardianPixel",
    "Sequential Replacement",
    "Sequential Matching",
    "Keyed Random Matching",
}


class ResearchResultError(ValueError):
    """Raised when persisted research results are invalid."""


def _convert_value(field: str, value: str | None):
    if value in (None, ""):
        return None
    if field in BOOLEAN_FIELDS:
        return str(value).strip().lower() == "true"
    if field in INTEGER_FIELDS:
        return int(float(value))
    if field in FLOAT_FIELDS:
        return float(value)
    return value


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ResearchResultError(f"Research CSV is empty: {path}")

    converted: list[dict] = []
    for row in rows:
        converted.append(
            {
                field: _convert_value(field, value)
                for field, value in row.items()
            }
        )
    return converted


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _numeric_mean(rows: list[dict], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    return float(mean(values)) if values else None


def _is_reliable(row: dict) -> bool:
    return bool(
        row.get("embedding_success") is True
        and row.get("extraction_success") is True
        and row.get("ber") is not None
        and float(row["ber"]) == 0.0
    )


def _metric_summary(rows: list[dict]) -> dict:
    """Aggregate only reliable BER-zero rows."""

    return {
        "psnr": _numeric_mean(rows, "psnr"),
        "ssim": _numeric_mean(rows, "ssim"),
        "changedPixelRatioPercent": _numeric_mean(
            rows,
            "changed_pixel_ratio",
        ),
        "histogramL1": _numeric_mean(rows, "histogram_l1_mean"),
        "histogramChiSquare": _numeric_mean(
            rows,
            "histogram_chi_square_mean",
        ),
        "absoluteLsbChiSquareChange": _numeric_mean(
            rows,
            "absolute_lsb_chi_square_statistic_change",
        ),
        "absoluteRsImbalanceChange": _numeric_mean(
            rows,
            "absolute_rs_combined_imbalance_change",
        ),
        "smoothRegionChangeRatio": _numeric_mean(
            rows,
            "smooth_region_change_ratio",
        ),
        "modifiedPositionSuitability": _numeric_mean(
            rows,
            "mean_modified_position_suitability",
        ),
        "imageSuitability": _numeric_mean(
            rows,
            "mean_image_suitability",
        ),
        "suitabilityGain": _numeric_mean(rows, "suitability_gain"),
    }


def _build_method_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    summaries: list[dict] = []
    for method, method_rows in sorted(grouped.items()):
        embedded = [
            row
            for row in method_rows
            if row.get("embedding_success") is True
        ]
        reliable = [row for row in method_rows if _is_reliable(row)]

        summaries.append(
            {
                "method": method,
                "includedInReliableRanking": method in RELIABLE_METHODS,
                "totalRuns": len(method_rows),
                "embeddingSuccessRuns": len(embedded),
                "reliableRuns": len(reliable),
                "reliableSuccessRatePercent": (
                    100.0 * len(reliable) / len(method_rows)
                ),
                "meanBerAllRuns": _numeric_mean(method_rows, "ber"),
                "reliableMetrics": _metric_summary(reliable),
                "timing": {
                    "embeddingSeconds": _numeric_mean(
                        embedded,
                        "embedding_seconds",
                    ),
                    "extractionSeconds": _numeric_mean(
                        method_rows,
                        "extraction_seconds",
                    ),
                },
            }
        )
    return summaries


def _build_grouped_summary(
    rows: list[dict],
    grouping_field: str,
) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[grouping_field]), str(row["method"]))].append(row)

    result: list[dict] = []
    for (group_value, method), group_rows in sorted(grouped.items()):
        reliable = [row for row in group_rows if _is_reliable(row)]
        result.append(
            {
                grouping_field: (
                    int(group_value)
                    if grouping_field == "requested_payload_bytes"
                    else group_value
                ),
                "method": method,
                "includedInReliableRanking": method in RELIABLE_METHODS,
                "totalRuns": len(group_rows),
                "reliableRuns": len(reliable),
                "reliableSuccessRatePercent": (
                    100.0 * len(reliable) / len(group_rows)
                ),
                "reliableMetrics": _metric_summary(reliable),
            }
        )
    return result


def _build_canny_diagnostics(rows: list[dict]) -> dict:
    canny_rows = [
        row
        for row in rows
        if row.get("method") == "Canny Synchronous"
    ]
    reasons = Counter(
        str(row.get("failure_reason") or "NONE")
        for row in canny_rows
    )
    return {
        "totalRuns": len(canny_rows),
        "embeddingSuccessRuns": sum(
            row.get("embedding_success") is True
            for row in canny_rows
        ),
        "reliableRuns": sum(_is_reliable(row) for row in canny_rows),
        "meanBer": _numeric_mean(canny_rows, "ber"),
        "failureReasons": dict(sorted(reasons.items())),
        "rankingNote": (
            "Canny Synchronous is excluded from reliable quality and "
            "steganalysis rankings when extraction is not BER-zero."
        ),
    }


@research_blueprint.get("/research")
def get_research_results():
    """Return real held-out equal-BPP research results."""

    directory = Path(current_app.config["STEGANALYSIS_DIRECTORY"])
    paths = {
        "runs": directory / "steganalysis_runs.csv",
        "storedMethodSummary": directory / "steganalysis_method_summary.csv",
        "storedCategorySummary": directory / "steganalysis_category_summary.csv",
        "methodology": directory / "methodology.json",
        "batchErrors": directory / "batch_errors.json",
    }

    try:
        runs = _read_csv(paths["runs"])
        stored_method_summary = _read_csv(paths["storedMethodSummary"])
        stored_category_summary = _read_csv(paths["storedCategorySummary"])
        methodology = _read_json(paths["methodology"])
        batch_errors = _read_json(paths["batchErrors"])
    except FileNotFoundError as error:
        return jsonify(
            {
                "code": "RESEARCH_RESULT_NOT_FOUND",
                "message": f"Required research result is missing: {error}",
            }
        ), 404
    except (OSError, csv.Error, json.JSONDecodeError, ValueError):
        current_app.logger.exception("Could not parse research result files.")
        return jsonify(
            {
                "code": "INVALID_RESEARCH_RESULT",
                "message": "The persisted research result files are invalid.",
            }
        ), 500

    covers = sorted({str(row["image_id"]) for row in runs})
    payload_levels = sorted(
        {
            int(row["requested_payload_bytes"])
            for row in runs
            if row.get("requested_payload_bytes") is not None
        }
    )
    methods = sorted({str(row["method"]) for row in runs})
    categories = sorted({str(row["cover_category"]) for row in runs})

    limitations = [
        str(
            methodology.get(
                "limitation",
                "Classical steganalysis metrics are indicators only.",
            )
        ),
        (
            "Results apply to the selected 20-cover test set, three payload "
            "levels and unchanged lossless PNG outputs."
        ),
        (
            "Suitability gain measures agreement with GuardianPixel's own "
            "fusion map; it is not an independent security proof."
        ),
    ]

    return jsonify(
        {
            "isSimulated": False,
            "preliminary": False,
            "dataset": {
                "coverCount": len(covers),
                "covers": covers,
                "payloadLevelsBytes": payload_levels,
                "methodCount": len(methods),
                "methods": methods,
                "categories": categories,
                "resultRows": len(runs),
                "equalBitComparison": True,
                "equalBppWithinEachCondition": True,
            },
            "methodology": methodology,
            "methodSummary": _build_method_summary(runs),
            "payloadSummary": _build_grouped_summary(
                runs,
                "requested_payload_bytes",
            ),
            "categorySummary": _build_grouped_summary(
                runs,
                "cover_category",
            ),
            "cannyDiagnostics": _build_canny_diagnostics(runs),
            "runs": runs,
            "batchErrors": batch_errors,
            "storedSummaries": {
                "methods": stored_method_summary,
                "categories": stored_category_summary,
            },
            "limitations": limitations,
        }
    ), 200
