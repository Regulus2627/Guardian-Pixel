from pathlib import Path
import json
import csv
import cv2

from backend.metrics.steganalysis import compute_steganalysis_metrics


COVERS_DIR = Path("data/div2k/validation")
STEGO_DIR = Path("data/div2k/steganalysis_stego")
OUTPUT_JSON = STEGO_DIR / "steganalysis_results.json"
OUTPUT_CSV = STEGO_DIR / "steganalysis_results.csv"


def load_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    covers = sorted(COVERS_DIR.glob("*.png"))

    if len(covers) != 20:
        raise RuntimeError(
            f"Expected exactly 20 validation covers, found {len(covers)}"
        )

    results = []

    for index, cover_path in enumerate(covers, start=1):
        stego_path = STEGO_DIR / cover_path.name

        if not stego_path.exists():
            raise RuntimeError(f"Missing stego image: {stego_path}")

        print(f"[{index}/20] Analysing {cover_path.name}...")

        cover = load_rgb(cover_path)
        stego = load_rgb(stego_path)

        metrics = compute_steganalysis_metrics(
            cover=cover,
            stego=stego,
            smooth_quantile=0.25,
            variance_window=11,
            rs_channel=1,
        )

        row = metrics.to_dict()
        row["image_id"] = cover_path.name

        results.append(row)

        print(
            f"    Modified pixels: {row.get('modified_pixel_count')}"
            f" | Changed ratio: {row.get('changed_pixel_ratio')}"
            f" | Suitability gain: {row.get('suitability_gain')}"
        )

    OUTPUT_JSON.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    fieldnames = sorted({key for row in results for key in row.keys()})

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("20-image steganalysis completed.")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
