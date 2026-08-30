from dataclasses import replace

import pytest

from backend.compatibility.categories import (
    CategoryError,
    categorize_texture_score,
    fit_category_thresholds,
)
from tests.compatibility.test_features import (
    create_vision_result,
)
from backend.compatibility.features import (
    extract_cover_features,
)


def create_features_with_score(
    image_id: str,
    score: float,
):
    features = extract_cover_features(
        image_id=image_id,
        vision_result=create_vision_result(),
    )

    return replace(
        features,
        texture_score=score,
    )


def test_fit_and_categorize():
    features = [
        create_features_with_score(
            "cover-1",
            0.10,
        ),
        create_features_with_score(
            "cover-2",
            0.20,
        ),
        create_features_with_score(
            "cover-3",
            0.50,
        ),
        create_features_with_score(
            "cover-4",
            0.90,
        ),
    ]

    thresholds = fit_category_thresholds(
        features
    )

    assert categorize_texture_score(
        0.05,
        thresholds,
    ) == "Smooth"

    assert categorize_texture_score(
        0.95,
        thresholds,
    ) == "Textured"

    assert categorize_texture_score(
        0.45,
        thresholds,
    ) == "Mixed"


def test_too_few_covers_are_rejected():
    features = [
        create_features_with_score(
            "cover-1",
            0.10,
        )
    ]

    with pytest.raises(
        CategoryError,
        match="At least four",
    ):
        fit_category_thresholds(features)