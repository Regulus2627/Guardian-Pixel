from pathlib import Path
import json
import csv
from statistics import mean, median

INPUT = Path("data/div2k/steganalysis_stego/steganalysis_results.json")
OUTPUT_JSON = Path("data/div2k/steganalysis_stego/steganalysis_summary.json")
OUTPUT_CSV = Path("data/div2k/steganalysis_stego/steganalysis_summary.csv")


def stats(values):
    values = [float(v) for v in values]
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    results = json.loads(INPUT.read_text(encoding="utf-8"))

    if len(results) != 20:
        raise RuntimeError(f"Expected 20 results, found {len(results)}")

    scalar_metrics = [
        "lsb_chi_square_statistic_change",
        "lsb_chi_square_p_value_change",
        "rs_combined_imbalance_change",
        "modified_pixel_count",
        "changed_pixel_ratio",
        "smooth_pixel_fraction",
        "smooth_region_change_ratio",
    ]

    histogram_metrics = [
        "histogram_l1",
        "histogram_chi_square",
    ]

    summary = {
        "validation": {
            "dataset": "DIV2K validation",
            "image_count": len(results),
            "image_ids": [row["image_id"] for row in results],
            "payload_bytes": 5000,
            "embedding_output": "PNG",
            "analysis_type": "Classical steganalysis",
            "interpretation_note": (
                "These metrics are steganalysis indicators and do not "
                "by themselves prove undetectability or security."
            ),
        },
        "metrics": {},
        "per_image": results,
    }

    for metric in scalar_metrics:
        values = [row[metric] for row in results]
        summary["metrics"][metric] = stats(values)

    for metric in histogram_metrics:
        channel_stats = {}

        for channel in ["red", "green", "blue", "mean"]:
            values = [row[metric][channel] for row in results]
            channel_stats[channel] = stats(values)

        summary["metrics"][metric] = channel_stats

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    csv_rows = []

    for row in results:
        csv_rows.append({
            "image_id": row["image_id"],
            "histogram_l1_mean": row["histogram_l1"]["mean"],
            "histogram_chi_square_mean": row["histogram_chi_square"]["mean"],
            "lsb_chi_square_statistic_change": row["lsb_chi_square_statistic_change"],
            "lsb_chi_square_p_value_change": row["lsb_chi_square_p_value_change"],
            "rs_combined_imbalance_change": row["rs_combined_imbalance_change"],
            "modified_pixel_count": row["modified_pixel_count"],
            "changed_pixel_ratio": row["changed_pixel_ratio"],
            "smooth_pixel_fraction": row["smooth_pixel_fraction"],
            "smooth_region_change_ratio": row["smooth_region_change_ratio"],
        })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print("20-image steganalysis summary created.")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
