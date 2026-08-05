from pathlib import Path

import numpy as np
import pytest

from backend.vision.config import load_vision_config
from backend.vision.hed.inference import (
    HEDInference,
    HEDInferenceError,
)
from backend.vision.hed.model import (
    HEDModelError,
    load_hed_network,
)


class FakeHEDNetwork:
    """Small fake network for testing inference validation."""

    def __init__(self):
        self.input_blob = None

    def setInput(self, blob):
        self.input_blob = blob

    def forward(self):
        height = self.input_blob.shape[2]
        width = self.input_blob.shape[3]

        output = np.zeros(
            (1, 1, height, width),
            dtype=np.float32,
        )

        output[
            0,
            0,
            height // 4: 3 * height // 4,
            width // 4: 3 * width // 4,
        ] = 0.8

        return output


def test_fake_hed_inference_output():
    network = FakeHEDNetwork()

    inference = HEDInference(
        network=network,
        mean_bgr=[
            104.00698793,
            116.66876762,
            122.67891434,
        ],
    )

    rgb = np.zeros(
        (32, 48, 3),
        dtype=np.uint8,
    )

    edge_map = inference.predict(rgb)

    assert edge_map.shape == (32, 48)
    assert edge_map.dtype == np.float32
    assert edge_map.min() >= 0
    assert edge_map.max() <= 1
    assert np.all(np.isfinite(edge_map))


def test_rgb_shape_is_required():
    inference = HEDInference(
        network=FakeHEDNetwork(),
        mean_bgr=[104, 116, 122],
    )

    grayscale = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(
        HEDInferenceError,
        match="H×W×3",
    ):
        inference.predict(grayscale)


def test_uint8_input_is_required():
    inference = HEDInference(
        network=FakeHEDNetwork(),
        mean_bgr=[104, 116, 122],
    )

    rgb = np.zeros(
        (32, 32, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        HEDInferenceError,
        match="uint8",
    ):
        inference.predict(rgb)


def test_missing_model_files_are_rejected(tmp_path):
    with pytest.raises(
        HEDModelError,
        match="prototxt file was not found",
    ):
        load_hed_network(
            tmp_path / "missing.prototxt",
            tmp_path / "missing.caffemodel",
        )


def test_real_hed_model_can_be_loaded_and_run():
    config = load_vision_config()

    prototxt = Path(config.hed.prototxt_path)
    weights = Path(config.hed.weights_path)

    if not prototxt.exists() or not weights.exists():
        pytest.skip("Local HED model files are unavailable.")

    network = load_hed_network(
        prototxt_path=prototxt,
        weights_path=weights,
        device=config.hed.resolved_device,
    )

    inference = HEDInference(
        network=network,
        mean_bgr=config.hed.mean_bgr,
    )

    # A simple image with a strong square boundary.
    rgb = np.zeros(
        (128, 128, 3),
        dtype=np.uint8,
    )

    rgb[32:96, 32:96] = 255

    edge_map = inference.predict(rgb)

    assert edge_map.shape == (128, 128)
    assert edge_map.dtype == np.float32
    assert np.all(np.isfinite(edge_map))
    assert edge_map.min() >= 0
    assert edge_map.max() <= 1