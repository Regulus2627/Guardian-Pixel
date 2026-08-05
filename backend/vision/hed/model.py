"""OpenCV DNN model loading for pretrained HED."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class HEDModelError(RuntimeError):
    """Raised when the pretrained HED model cannot be loaded."""


class CropLayer:
    """
    OpenCV implementation of the Caffe Crop layer used by HED.

    HED upsamples its side outputs and crops them so they have the
    same spatial dimensions as the original network input.
    """

    def __init__(self, params, blobs):
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0

    def getMemoryShapes(self, inputs):
        input_shape = inputs[0]
        target_shape = inputs[1]

        batch_size = input_shape[0]
        channels = input_shape[1]

        height = target_shape[2]
        width = target_shape[3]

        self.y_start = (
            input_shape[2] - target_shape[2]
        ) // 2

        self.x_start = (
            input_shape[3] - target_shape[3]
        ) // 2

        self.y_end = self.y_start + height
        self.x_end = self.x_start + width

        return [
            [
                batch_size,
                channels,
                height,
                width,
            ]
        ]

    def forward(self, inputs):
        cropped = inputs[0][
            :,
            :,
            self.y_start:self.y_end,
            self.x_start:self.x_end,
        ]

        return [cropped]


_crop_layer_registered = False


def register_crop_layer() -> None:
    """Register HED's custom Crop layer with OpenCV."""

    global _crop_layer_registered

    if _crop_layer_registered:
        return

    try:
        cv2.dnn_registerLayer(
            "Crop",
            CropLayer,
        )
    except cv2.error as error:
        # Some OpenCV sessions may already have the layer registered.
        if "already" not in str(error).lower():
            raise HEDModelError(
                f"Could not register HED Crop layer: {error}"
            ) from error

    _crop_layer_registered = True


def load_hed_network(
    prototxt_path: str | Path,
    weights_path: str | Path,
    device: str = "cpu",
):
    """Load the pretrained Caffe HED network."""

    prototxt = Path(prototxt_path)
    weights = Path(weights_path)

    if not prototxt.exists():
        raise HEDModelError(
            f"HED prototxt file was not found: {prototxt}"
        )

    if not prototxt.is_file():
        raise HEDModelError(
            f"HED prototxt path is not a file: {prototxt}"
        )

    if not weights.exists():
        raise HEDModelError(
            f"HED weights file was not found: {weights}"
        )

    if not weights.is_file():
        raise HEDModelError(
            f"HED weights path is not a file: {weights}"
        )

    if prototxt.stat().st_size == 0:
        raise HEDModelError(
            "HED prototxt file is empty."
        )

    if weights.stat().st_size == 0:
        raise HEDModelError(
            "HED weights file is empty."
        )

    if device not in {"auto", "cpu"}:
        raise HEDModelError(
            "OpenCV HED currently supports auto or cpu."
        )

    register_crop_layer()

    # try:
    #     network = cv2.dnn.readNetFromCaffe(
    #         str(prototxt),
    #         str(weights),
    #     )
    # except cv2.error as error:
    #     raise HEDModelError(
    #         f"OpenCV could not load the HED model: {error}"
    #     ) from error
    

    try:
        read_from_caffe = getattr(
            cv2.dnn,
            "readNetFromCaffe",
            None,
        )

        if callable(read_from_caffe):
            network = read_from_caffe(
                str(prototxt),
                str(weights),
            )

        elif hasattr(cv2, "dnn_readNetFromCaffe"):
            network = cv2.dnn_readNetFromCaffe(
                str(prototxt),
                str(weights),
            )

        else:
            # Generic OpenCV loader.
            # readNet expects model first and configuration second.
            network = cv2.dnn.readNet(
                str(weights),
                str(prototxt),
                "Caffe",
            )

    except (cv2.error, AttributeError) as error:
        raise HEDModelError(
            f"OpenCV could not load the HED model: {error}"
        ) from error
    
    if network.empty():
        raise HEDModelError(
            "OpenCV loaded an empty HED network."
        )

    network.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    network.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    return network