"""Pretrained HED inference for GuardianPixel."""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np


class HEDInferenceError(RuntimeError):
    """Raised when HED inference cannot be completed."""


class HEDInference:
    """Run pretrained HED inference on a validated RGB image."""

    def __init__(
        self,
        network,
        mean_bgr: tuple[float, float, float]
        | list[float],
    ):
        if network is None:
            raise HEDInferenceError(
                "HED network cannot be None."
            )

        if len(mean_bgr) != 3:
            raise HEDInferenceError(
                "mean_bgr must contain three values."
            )

        self.network = network
        self.mean_bgr = tuple(
            float(value) for value in mean_bgr
        )

        self.last_inference_seconds = 0.0

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """
        Generate an H×W float32 edge-probability map.

        Input:
            RGB uint8 image with shape H×W×3.

        Output:
            Float32 H×W map clipped to [0, 1].
        """

        if not isinstance(rgb, np.ndarray):
            raise HEDInferenceError(
                "HED input must be a NumPy array."
            )

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise HEDInferenceError(
                "HED input must have shape H×W×3."
            )

        if rgb.size == 0:
            raise HEDInferenceError(
                "HED input cannot be empty."
            )

        if rgb.dtype != np.uint8:
            raise HEDInferenceError(
                "HED input must use uint8 values."
            )

        height, width = rgb.shape[:2]

        # The pretrained Caffe model expects BGR channel order.
        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        blob = cv2.dnn.blobFromImage(
            image=bgr,
            scalefactor=1.0,
            size=(width, height),
            mean=self.mean_bgr,
            swapRB=False,
            crop=False,
        )

        try:
            start_time = perf_counter()

            self.network.setInput(blob)
            output = self.network.forward()

            self.last_inference_seconds = (
                perf_counter() - start_time
            )

        except cv2.error as error:
            raise HEDInferenceError(
                f"HED forward inference failed: {error}"
            ) from error

        output_array = np.asarray(output)

        if output_array.ndim != 4:
            raise HEDInferenceError(
                f"Unexpected HED output shape: "
                f"{output_array.shape}"
            )

        edge_map = output_array[0, 0]

        if edge_map.shape != (height, width):
            edge_map = cv2.resize(
                edge_map,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        edge_map = np.asarray(
            edge_map,
            dtype=np.float32,
        )

        edge_map = np.clip(
            edge_map,
            0.0,
            1.0,
        )

        if edge_map.shape != (height, width):
            raise HEDInferenceError(
                "HED edge map does not match input dimensions."
            )

        if not np.all(np.isfinite(edge_map)):
            raise HEDInferenceError(
                "HED edge map contains invalid values."
            )

        return edge_map