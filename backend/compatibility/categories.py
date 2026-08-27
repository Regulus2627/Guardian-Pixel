"""Cover categorization using texture-score percentiles."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from backend.compatibility.features import (
    CoverFeatures,
)


class CategoryError(ValueError):
    """Raised when cover categories cannot be calculated."""


@dataclass(frozen=True, slots=True)
class CategoryThresholds:
    """Texture-score category boundaries."""

    smooth_upper: float
    textured_lower: float

    def to_dict(self) -> dict[str, float]:
        return {
            "smooth_upper": self.smooth_upper,
            "textured_lower": self.textured_lower,
        }


def fit_category_thresholds(
    features: list[CoverFeatures],
) -> CategoryThresholds:
    """Calculate thresholds from training-cover percentiles."""

    if len(features) < 4:
        raise CategoryError(
            "At least four cover images are required."
        )

    scores = np.asarray(
        [
            feature.texture_score
            for feature in features
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(scores)):
        raise CategoryError(
            "Texture scores contain invalid values."
        )

    smooth_upper = float(
        np.percentile(scores, 25)
    )

    textured_lower = float(
        np.percentile(scores, 75)
    )

    if smooth_upper > textured_lower:
        raise CategoryError(
            "Calculated thresholds are invalid."
        )

    return CategoryThresholds(
        smooth_upper=smooth_upper,
        textured_lower=textured_lower,
    )


def categorize_texture_score(
    texture_score: float,
    thresholds: CategoryThresholds,
) -> str:
    """Classify one cover as Smooth, Mixed or Textured."""

    if not isinstance(
        thresholds,
        CategoryThresholds,
    ):
        raise CategoryError(
            "thresholds must be CategoryThresholds."
        )

    if not np.isfinite(texture_score):
        raise CategoryError(
            "Texture score must be finite."
        )

    if texture_score <= thresholds.smooth_upper:
        return "Smooth"

    if texture_score >= thresholds.textured_lower:
        return "Textured"

    return "Mixed"


def save_category_thresholds(
    thresholds: CategoryThresholds,
    output_path: str | Path,
) -> Path:
    """Save fitted thresholds as JSON."""

    if not isinstance(
        thresholds,
        CategoryThresholds,
    ):
        raise CategoryError(
            "thresholds must be CategoryThresholds."
        )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            thresholds.to_dict(),
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def load_category_thresholds(
    input_path: str | Path,
) -> CategoryThresholds:
    """Load fixed category thresholds from JSON."""

    path = Path(input_path)

    if not path.exists():
        raise CategoryError(
            f"Threshold file was not found: {path}"
        )

    if not path.is_file():
        raise CategoryError(
            f"Threshold path is not a file: {path}"
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        smooth_upper = float(
            data["smooth_upper"]
        )

        textured_lower = float(
            data["textured_lower"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise CategoryError(
            "Threshold file is invalid."
        ) from error

    if not np.isfinite(
        smooth_upper
    ) or not np.isfinite(
        textured_lower
    ):
        raise CategoryError(
            "Category thresholds must be finite."
        )

    if not (
        0
        <= smooth_upper
        <= textured_lower
        <= 1
    ):
        raise CategoryError(
            "Category thresholds must satisfy "
            "0 <= smooth <= textured <= 1."
        )

    return CategoryThresholds(
        smooth_upper=smooth_upper,
        textured_lower=textured_lower,
    )


def write_categorized_features_csv(
    features: list[CoverFeatures],
    thresholds: CategoryThresholds,
    output_path: str | Path,
) -> Path:
    """Write cover features and fixed categories to CSV."""

    if not features:
        raise CategoryError(
            "At least one cover feature row is required."
        )

    if not all(
        isinstance(item, CoverFeatures)
        for item in features
    ):
        raise CategoryError(
            "Every item must be CoverFeatures."
        )

    rows = []

    for feature in features:
        row = asdict(feature)

        row["cover_category"] = (
            categorize_texture_score(
                feature.texture_score,
                thresholds,
            )
        )

        rows.append(row)

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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

    return path