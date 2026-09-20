"""Measure one real cover/stego image pair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add the Guardian-Pixel repository root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import numpy as np
from PIL import Image

from backend.metrics.steganalysis import (
    compute_steganalysis_metrics,
)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 NumPy array."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Image file was not found: {path}"
        )

    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


def main() -> None:
    """Load one cover/stego pair and print steganalysis metrics."""

    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage:\n"
            "python scripts/check_steganalysis_pair.py "
            "<cover-image> <stego-png>"
        )

    cover_path = Path(sys.argv[1])
    stego_path = Path(sys.argv[2])

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Cover: {cover_path}")
    print(f"Stego: {stego_path}")

    cover = load_rgb(cover_path)
    stego = load_rgb(stego_path)

    if cover.shape != stego.shape:
        raise ValueError(
            "Cover and stego dimensions do not match. "
            f"Cover shape: {cover.shape}; "
            f"stego shape: {stego.shape}."
        )

    metrics = compute_steganalysis_metrics(
        cover=cover,
        stego=stego,
        smooth_quantile=0.25,
        variance_window=11,
        rs_channel=1,
    )

    print()
    print("Steganalysis completed successfully.")
    print()

    print(
        json.dumps(
            metrics.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()