"""Cover categorization using dataset texture-score percentiles."""

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
    """Texture-score boundaries calculated from pilot covers."""

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
    """Use the 25th and 75th percentiles as boundaries."""

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

    return CategoryThresholds(
        smooth_upper=float(
            np.percentile(scores, 25)
        ),
        textured_lower=float(
            np.percentile(scores, 75)
        ),
    )


def categorize_texture_score(
    texture_score: float,
    thresholds: CategoryThresholds,
) -> str:
    """Classify one texture score."""

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
    """Save category thresholds as JSON."""

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


def write_categorized_features_csv(
    features: list[CoverFeatures],
    thresholds: CategoryThresholds,
    output_path: str | Path,
) -> Path:
    """Write cover features and calculated categories."""

    if not features:
        raise CategoryError(
            "At least one cover feature row is required."
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