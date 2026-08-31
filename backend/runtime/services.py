"""Reusable GuardianPixel model and service instances."""

from __future__ import annotations

from threading import Lock

from backend.vision.config import (
    load_vision_config,
)
from backend.vision.service import (
    GuardianPixelVisionService,
)


_vision_service = None
_service_lock = Lock()


def get_vision_service() -> GuardianPixelVisionService:
    """Load HED once and reuse it for later API requests."""

    global _vision_service

    if _vision_service is None:
        with _service_lock:
            if _vision_service is None:
                config = load_vision_config()

                _vision_service = (
                    GuardianPixelVisionService(
                        config=config
                    )
                )

    return _vision_service


def reset_runtime_services() -> None:
    """Reset cached services during testing."""

    global _vision_service

    with _service_lock:
        _vision_service = None