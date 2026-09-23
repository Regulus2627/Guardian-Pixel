"""Tests for classical steganalysis and adaptive-placement metrics."""

from __future__ import annotations

import numpy as np
import pytest

from backend.metrics.steganalysis import (
    SteganalysisError,
    compute_steganalysis_metrics,
    histogram_chi_square_distance,
    histogram_l1_distance,
    lsb_pair_chi_square,
    rs_analysis,
)


def _random_image(seed: int = 7, height: int = 64, width: int = 64) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_identical_images_have_zero_pair_distances() -> None:
    cover = _random_image()

    l1 = histogram_l1_distance(cover, cover.copy())
    histogram_chi = histogram_chi_square_distance(cover, cover.copy())
    metrics = compute_steganalysis_metrics(cover, cover.copy())

    assert l1.red == pytest.approx(0.0)
    assert l1.green == pytest.approx(0.0)
    assert l1.blue == pytest.approx(0.0)
    assert l1.mean == pytest.approx(0.0)
    assert histogram_chi.mean == pytest.approx(0.0)
    assert metrics.modified_pixel_count == 0
    assert metrics.changed_pixel_ratio == pytest.approx(0.0)
    assert metrics.smooth_region_change_ratio == pytest.approx(0.0)
    assert metrics.lsb_chi_square_statistic_change == pytest.approx(0.0)
    assert metrics.rs_combined_imbalance_change == pytest.approx(0.0)


def test_controlled_changes_and_suitability_are_measured() -> None:
    cover = _random_image(seed=12)
    stego = cover.copy()

    positions = [(2, 3), (10, 11), (20, 21), (30, 31)]
    for row, column in positions:
        value = int(stego[row, column, 1])
        stego[row, column, 1] = value + 1 if value < 255 else value - 1

    suitability = np.full(cover.shape[:2], 0.25, dtype=np.float64)
    for row, column in positions:
        suitability[row, column] = 1.0

    metrics = compute_steganalysis_metrics(
        cover,
        stego,
        suitability_map=suitability,
    )

    assert metrics.modified_pixel_count == 4
    assert metrics.changed_pixel_ratio == pytest.approx(4 / (64 * 64))
    assert metrics.mean_modified_position_suitability == pytest.approx(1.0)
    assert metrics.mean_image_suitability is not None
    assert metrics.suitability_gain is not None
    assert metrics.suitability_gain > 1.0
    assert metrics.histogram_l1.green > 0.0


def test_metric_results_are_deterministic() -> None:
    cover = _random_image(seed=20)
    stego = cover.copy()
    stego[::8, ::8, 2] ^= np.uint8(1)

    first = compute_steganalysis_metrics(cover, stego).to_dict()
    second = compute_steganalysis_metrics(cover, stego).to_dict()

    assert first == second


def test_lsb_chi_square_returns_finite_values() -> None:
    image = _random_image(seed=33)
    result = lsb_pair_chi_square(image)

    assert np.isfinite(result.statistic)
    assert result.degrees_of_freedom > 0
    assert 0.0 <= result.p_value <= 1.0


def test_rs_group_accounting_is_exact() -> None:
    image = _random_image(seed=44)
    result = rs_analysis(image, channel=1, group_size=4)

    assert (
        result.regular_positive
        + result.singular_positive
        + result.unusable_positive
        == result.group_count
    )
    assert (
        result.regular_negative
        + result.singular_negative
        + result.unusable_negative
        == result.group_count
    )
    assert 0.0 <= result.combined_imbalance <= 1.0


def test_boundary_values_do_not_overflow() -> None:
    cover = np.zeros((16, 16, 3), dtype=np.uint8)
    cover[::2, :, :] = 255
    stego = cover.copy()
    stego[0, 0, :] = 254
    stego[1, 0, :] = 1

    metrics = compute_steganalysis_metrics(cover, stego)

    assert metrics.modified_pixel_count == 2
    assert np.isfinite(metrics.histogram_l1.mean)
    assert np.isfinite(metrics.histogram_chi_square.mean)


def test_invalid_shape_is_rejected() -> None:
    gray = np.zeros((32, 32), dtype=np.uint8)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(SteganalysisError, match="shape"):
        compute_steganalysis_metrics(gray, rgb)


def test_invalid_suitability_shape_is_rejected() -> None:
    cover = _random_image()

    with pytest.raises(SteganalysisError, match="H×W"):
        compute_steganalysis_metrics(
            cover,
            cover.copy(),
            suitability_map=np.zeros((10, 10), dtype=np.float64),
        )
