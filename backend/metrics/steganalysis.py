"""Classical steganalysis and adaptive-placement metrics for GuardianPixel.

The functions in this module compare an unchanged RGB cover image with its
lossless RGB stego image.  They provide research measurements; no individual
metric proves that an image is undetectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import chi2


class SteganalysisError(ValueError):
    """Raised when steganalysis inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ChannelMetric:
    """One value for each RGB channel plus their arithmetic mean."""

    red: float
    green: float
    blue: float
    mean: float


@dataclass(frozen=True, slots=True)
class LsbChiSquareResult:
    """Classic even/odd LSB-pair chi-square result."""

    statistic: float
    degrees_of_freedom: int
    p_value: float


@dataclass(frozen=True, slots=True)
class RsResult:
    """Regular/singular group counts for positive and negative masks."""

    group_count: int
    regular_positive: int
    singular_positive: int
    unusable_positive: int
    regular_negative: int
    singular_negative: int
    unusable_negative: int
    positive_imbalance: float
    negative_imbalance: float
    combined_imbalance: float


@dataclass(frozen=True, slots=True)
class SteganalysisMetrics:
    """Complete cover-to-stego research measurement."""

    histogram_l1: ChannelMetric
    histogram_chi_square: ChannelMetric
    cover_lsb_chi_square: LsbChiSquareResult
    stego_lsb_chi_square: LsbChiSquareResult
    lsb_chi_square_statistic_change: float
    lsb_chi_square_p_value_change: float
    cover_rs: RsResult
    stego_rs: RsResult
    rs_combined_imbalance_change: float
    modified_pixel_count: int
    changed_pixel_ratio: float
    smooth_pixel_fraction: float
    smooth_region_change_ratio: float
    mean_modified_position_suitability: float | None
    mean_image_suitability: float | None
    suitability_gain: float | None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def _validate_rgb_pair(
    cover: np.ndarray,
    stego: np.ndarray,
) -> None:
    if not isinstance(cover, np.ndarray) or not isinstance(stego, np.ndarray):
        raise SteganalysisError("Cover and stego images must be NumPy arrays.")
    if cover.ndim != 3 or cover.shape[2] != 3:
        raise SteganalysisError("Cover must have shape H×W×3.")
    if stego.ndim != 3 or stego.shape[2] != 3:
        raise SteganalysisError("Stego must have shape H×W×3.")
    if cover.shape != stego.shape:
        raise SteganalysisError("Cover and stego dimensions must match.")
    if cover.dtype != np.uint8 or stego.dtype != np.uint8:
        raise SteganalysisError("Cover and stego images must use uint8 values.")
    if cover.shape[0] == 0 or cover.shape[1] == 0:
        raise SteganalysisError("Images cannot be empty.")


def _channel_metric(values: list[float]) -> ChannelMetric:
    return ChannelMetric(
        red=float(values[0]),
        green=float(values[1]),
        blue=float(values[2]),
        mean=float(np.mean(values)),
    )


def _histogram(channel: np.ndarray) -> np.ndarray:
    counts = np.bincount(channel.reshape(-1), minlength=256).astype(np.float64)
    return counts / float(channel.size)


def histogram_l1_distance(
    cover: np.ndarray,
    stego: np.ndarray,
) -> ChannelMetric:
    """Calculate normalized RGB histogram L1 distances."""

    _validate_rgb_pair(cover, stego)
    values = []
    for channel in range(3):
        cover_hist = _histogram(cover[:, :, channel])
        stego_hist = _histogram(stego[:, :, channel])
        values.append(float(np.sum(np.abs(cover_hist - stego_hist))))
    return _channel_metric(values)


def histogram_chi_square_distance(
    cover: np.ndarray,
    stego: np.ndarray,
    epsilon: float = 1e-12,
) -> ChannelMetric:
    """Calculate symmetric normalized histogram chi-square distances."""

    _validate_rgb_pair(cover, stego)
    if epsilon <= 0:
        raise SteganalysisError("epsilon must be positive.")

    values = []
    for channel in range(3):
        cover_hist = _histogram(cover[:, :, channel])
        stego_hist = _histogram(stego[:, :, channel])
        numerator = np.square(cover_hist - stego_hist)
        denominator = cover_hist + stego_hist + epsilon
        values.append(float(0.5 * np.sum(numerator / denominator)))
    return _channel_metric(values)


def lsb_pair_chi_square(image: np.ndarray) -> LsbChiSquareResult:
    """Test whether even/odd histogram pairs are unusually equalized.

    All three RGB channels are pooled.  The p-value is reported together with
    the statistic because it must be interpreted using this exact null model
    and calibrated cover/stego experiments.
    """

    if not isinstance(image, np.ndarray):
        raise SteganalysisError("Image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise SteganalysisError("Image must be an RGB uint8 H×W×3 array.")

    counts = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    even = counts[0::2]
    odd = counts[1::2]
    pair_totals = even + odd
    valid = pair_totals > 0

    if not np.any(valid):
        return LsbChiSquareResult(0.0, 0, 1.0)

    expected = pair_totals[valid] / 2.0
    statistic = float(
        np.sum(np.square(even[valid] - expected) / expected)
        + np.sum(np.square(odd[valid] - expected) / expected)
    )
    degrees_of_freedom = int(np.count_nonzero(valid))
    p_value = float(chi2.sf(statistic, degrees_of_freedom))

    return LsbChiSquareResult(
        statistic=statistic,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
    )


def _discrimination(groups: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(groups.astype(np.int16), axis=1)), axis=1)


def _flip(values: np.ndarray, mask: np.ndarray, negative: bool) -> np.ndarray:
    result = values.astype(np.int16).copy()
    selected = mask != 0
    selected_values = result[:, selected]

    if negative:
        delta = np.where((selected_values & 1) == 0, -1, 1)
    else:
        delta = np.where((selected_values & 1) == 0, 1, -1)

    result[:, selected] = np.clip(selected_values + delta, 0, 255)
    return result


def rs_analysis(
    image: np.ndarray,
    channel: int = 1,
    group_size: int = 4,
) -> RsResult:
    """Calculate regular/singular group statistics on one image channel.

    This implementation reports an RS imbalance signal rather than claiming an
    exact payload-rate estimate.  A lower or higher value is not labelled safe
    without cover/stego calibration.
    """

    if not isinstance(image, np.ndarray):
        raise SteganalysisError("Image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise SteganalysisError("Image must be an RGB uint8 H×W×3 array.")
    if channel not in (0, 1, 2):
        raise SteganalysisError("channel must be 0, 1, or 2.")
    if group_size < 2:
        raise SteganalysisError("group_size must be at least 2.")

    flat = image[:, :, channel].reshape(-1)
    group_count = flat.size // group_size
    if group_count == 0:
        raise SteganalysisError("Image is too small for RS grouping.")

    groups = flat[: group_count * group_size].reshape(group_count, group_size)
    mask = np.zeros(group_size, dtype=np.int8)
    mask[1::2] = 1
    if not np.any(mask):
        mask[0] = 1

    original = _discrimination(groups)
    positive = _discrimination(_flip(groups, mask, negative=False))
    negative = _discrimination(_flip(groups, mask, negative=True))

    regular_positive = int(np.count_nonzero(positive > original))
    singular_positive = int(np.count_nonzero(positive < original))
    unusable_positive = group_count - regular_positive - singular_positive

    regular_negative = int(np.count_nonzero(negative > original))
    singular_negative = int(np.count_nonzero(negative < original))
    unusable_negative = group_count - regular_negative - singular_negative

    positive_imbalance = abs(regular_positive - singular_positive) / group_count
    negative_imbalance = abs(regular_negative - singular_negative) / group_count

    return RsResult(
        group_count=group_count,
        regular_positive=regular_positive,
        singular_positive=singular_positive,
        unusable_positive=unusable_positive,
        regular_negative=regular_negative,
        singular_negative=singular_negative,
        unusable_negative=unusable_negative,
        positive_imbalance=float(positive_imbalance),
        negative_imbalance=float(negative_imbalance),
        combined_imbalance=float((positive_imbalance + negative_imbalance) / 2.0),
    )


def _luminance(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float64)
    return (
        0.2126 * image_float[:, :, 0]
        + 0.7152 * image_float[:, :, 1]
        + 0.0722 * image_float[:, :, 2]
    )


def local_variance_map(image: np.ndarray, window_size: int = 11) -> np.ndarray:
    """Calculate a luminance local-variance map."""

    if window_size < 3 or window_size % 2 == 0:
        raise SteganalysisError("window_size must be an odd integer of at least 3.")
    luminance = _luminance(image)
    local_mean = uniform_filter(luminance, size=window_size, mode="reflect")
    local_square_mean = uniform_filter(
        np.square(luminance), size=window_size, mode="reflect"
    )
    return np.maximum(local_square_mean - np.square(local_mean), 0.0)


def compute_steganalysis_metrics(
    cover: np.ndarray,
    stego: np.ndarray,
    suitability_map: np.ndarray | None = None,
    smooth_quantile: float = 0.25,
    variance_window: int = 11,
    rs_channel: int = 1,
) -> SteganalysisMetrics:
    """Calculate the complete, deterministic cover-to-stego metric set.

    ``smooth_quantile=0.25`` defines smooth pixels as the least-variable 25%
    of the current cover.  Keep this rule fixed for every compared method.
    """

    _validate_rgb_pair(cover, stego)
    if not 0.0 < smooth_quantile < 1.0:
        raise SteganalysisError("smooth_quantile must be between 0 and 1.")

    modified_mask = np.any(cover != stego, axis=2)
    modified_pixel_count = int(np.count_nonzero(modified_mask))
    total_pixels = int(modified_mask.size)

    variance = local_variance_map(cover, window_size=variance_window)
    smooth_threshold = float(np.quantile(variance, smooth_quantile))
    smooth_mask = variance <= smooth_threshold
    smooth_pixel_fraction = float(np.count_nonzero(smooth_mask) / total_pixels)

    if modified_pixel_count == 0:
        smooth_region_change_ratio = 0.0
    else:
        smooth_region_change_ratio = float(
            np.count_nonzero(modified_mask & smooth_mask) / modified_pixel_count
        )

    modified_suitability: float | None = None
    image_suitability: float | None = None
    suitability_gain: float | None = None

    if suitability_map is not None:
        if not isinstance(suitability_map, np.ndarray):
            raise SteganalysisError("suitability_map must be a NumPy array.")
        if suitability_map.shape != cover.shape[:2]:
            raise SteganalysisError("suitability_map must have shape H×W.")
        if not np.all(np.isfinite(suitability_map)):
            raise SteganalysisError("suitability_map must contain finite values.")

        image_suitability = float(np.mean(suitability_map))
        if modified_pixel_count > 0:
            modified_suitability = float(np.mean(suitability_map[modified_mask]))
            if image_suitability > 0:
                suitability_gain = float(modified_suitability / image_suitability)

    cover_chi = lsb_pair_chi_square(cover)
    stego_chi = lsb_pair_chi_square(stego)
    cover_rs = rs_analysis(cover, channel=rs_channel)
    stego_rs = rs_analysis(stego, channel=rs_channel)

    return SteganalysisMetrics(
        histogram_l1=histogram_l1_distance(cover, stego),
        histogram_chi_square=histogram_chi_square_distance(cover, stego),
        cover_lsb_chi_square=cover_chi,
        stego_lsb_chi_square=stego_chi,
        lsb_chi_square_statistic_change=float(stego_chi.statistic - cover_chi.statistic),
        lsb_chi_square_p_value_change=float(stego_chi.p_value - cover_chi.p_value),
        cover_rs=cover_rs,
        stego_rs=stego_rs,
        rs_combined_imbalance_change=float(
            stego_rs.combined_imbalance - cover_rs.combined_imbalance
        ),
        modified_pixel_count=modified_pixel_count,
        changed_pixel_ratio=float(modified_pixel_count / total_pixels),
        smooth_pixel_fraction=smooth_pixel_fraction,
        smooth_region_change_ratio=smooth_region_change_ratio,
        mean_modified_position_suitability=modified_suitability,
        mean_image_suitability=image_suitability,
        suitability_gain=suitability_gain,
    )
