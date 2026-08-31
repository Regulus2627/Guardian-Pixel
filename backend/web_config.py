class WebConfig:
    """Default Flask application configuration."""

    MAX_CONTENT_LENGTH = 35 * 1024 * 1024
    MAX_DECODED_FILE_BYTES = 20 * 1024 * 1024

    CATEGORY_THRESHOLDS_PATH = (
        "experiments/results/"
        "category_thresholds.json"
    )

    JSON_SORT_KEYS = False

    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]