"""Flask configuration for GuardianPixel."""

from __future__ import annotations


class WebConfig:
    """Default Flask application configuration."""

    MAX_CONTENT_LENGTH = 25 * 1024 * 1024

    JSON_SORT_KEYS = False

    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]