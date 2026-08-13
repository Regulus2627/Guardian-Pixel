import numpy as np
import pytest

from backend.compatibility.features import (
    CoverFeatureError,
    extract_cover_features,
    write_cover_features_csv,
)
from backend.vision.schemas import (
    BlockMapResult,
    FeatureMaps,
    ImageInfo,
    MapEncodingResult,
    VisionAnalysisResult,
)


def create_vision_result():
    height = 16
    width = 16

    hed = np.linspace(
        0,
        1,
        height * width,
        dtype=np.float32,
    ).reshape(height, width)

    entropy = np.full(
        (height, width),
        0.60,
        dtype=np.float32,
    )

    variance = np.full(
        (height, width),
        0.30,
        dtype=np.float32,
    )

    fused = (
        0.45 * hed
        + 0.35 * entropy
        + 0.20 * variance
    ).astype(np.float32)

    selected = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    scores = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )

    return VisionAnalysisResult(
        image_info=ImageInfo(
            width=width,
            height=height,
            channels=3,
            original_format="PNG",
            original_mode="RGB",
        ),
        feature_maps=FeatureMaps(
            hed_map=hed,
            entropy_map=entropy,
            variance_map=variance,
            fused_heatmap=fused,
        ),
        block_map=BlockMapResult(
            block_size=8,
            block_rows=2,
            block_columns=2,
            selected_blocks=selected,
            block_scores=scores,
            selected_block_count=2,
            total_block_count=4,
            selected_position_count=128,
            reserved_position_count=0,
            usable_position_count=128,
            required_payload_bits=100,
            target_position_count=110,
        ),
        map_encoding=MapEncodingResult(
            encoding_type="RAW",
            encoded_data=b"\x90",
            original_bit_count=4,
            raw_byte_count=1,
            encoded_byte_count=1,
        ),
        timings={"total": 1.5},
        config_used={},
    )


def test_extract_cover_features():
    result = create_vision_result()

    features = extract_cover_features(
        image_id="test-cover",
        vision_result=result,
    )

    assert features.image_id == "test-cover"
    assert features.width == 16
    assert features.height == 16
    assert features.total_pixels == 256
    assert features.total_blocks == 4
    assert features.selected_blocks == 2
    assert features.selected_percentage == 50.0

    assert 0 <= features.edge_density <= 1
    assert (
        0
        <= features.smooth_pixel_percentage
        <= 100
    )
    assert (
        0
        <= features.high_score_pixel_percentage
        <= 100
    )


def test_empty_image_id_is_rejected():
    with pytest.raises(
        CoverFeatureError,
        match="cannot be empty",
    ):
        extract_cover_features(
            image_id="",
            vision_result=create_vision_result(),
        )


def test_invalid_threshold_is_rejected():
    with pytest.raises(
        CoverFeatureError,
        match="between 0 and 1",
    ):
        extract_cover_features(
            image_id="cover",
            vision_result=create_vision_result(),
            edge_threshold=1.5,
        )


def test_csv_is_written(tmp_path):
    features = extract_cover_features(
        image_id="cover",
        vision_result=create_vision_result(),
    )

    output_path = (
        tmp_path / "cover_features.csv"
    )

    result_path = (
        write_cover_features_csv(
            [features],
            output_path,
        )
    )

    assert result_path.exists()

    content = result_path.read_text(
        encoding="utf-8"
    )

    assert "image_id" in content
    assert "cover" in content
    assert "hed_mean" in content