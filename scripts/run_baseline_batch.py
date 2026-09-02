"""Run equal-bit baseline comparisons across multiple covers and payload sizes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Payload sizes must be integers.") from error

    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Payload sizes must be positive.")

    return sizes


def load_category_lookup(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Cover-feature CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return {row["image_id"]: row for row in csv.DictReader(file)}


def numeric(values, field: str) -> list[float]:
    result: list[float] = []
    for row in values:
        value = row.get(field, "")
        if value not in (None, ""):
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                pass
    return result


def average_or_blank(values: list[float]):
    return mean(values) if values else ""


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No rows available for {path.name}.")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_method_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary: list[dict] = []
    for method, method_rows in sorted(grouped.items()):
        embedded = [row for row in method_rows if row.get("embedding_success") is True]
        reliable = [
            row
            for row in method_rows
            if row.get("extraction_success") is True
            and float(row.get("ber", 1.0)) == 0.0
        ]

        summary.append(
            {
                "method": method,
                "total_runs": len(method_rows),
                "embedding_success_runs": len(embedded),
                "reliable_ber_zero_runs": len(reliable),
                "reliable_success_rate_percent": 100.0 * len(reliable) / len(method_rows),
                "mean_ber": average_or_blank(numeric(method_rows, "ber")),
                "mean_psnr_all_embedded": average_or_blank(numeric(embedded, "psnr")),
                "mean_ssim_all_embedded": average_or_blank(numeric(embedded, "ssim")),
                "mean_psnr_reliable": average_or_blank(numeric(reliable, "psnr")),
                "mean_ssim_reliable": average_or_blank(numeric(reliable, "ssim")),
                "mean_changed_pixel_ratio": average_or_blank(
                    numeric(embedded, "changed_pixel_ratio")
                ),
                "mean_embedding_seconds": average_or_blank(
                    numeric(embedded, "embedding_seconds")
                ),
                "mean_extraction_seconds": average_or_blank(
                    numeric(method_rows, "extraction_seconds")
                ),
            }
        )

    return summary


def build_category_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["cover_category"], row["method"])].append(row)

    summary: list[dict] = []
    for (category, method), group in sorted(grouped.items()):
        reliable = [
            row
            for row in group
            if row.get("extraction_success") is True
            and float(row.get("ber", 1.0)) == 0.0
        ]
        summary.append(
            {
                "cover_category": category,
                "method": method,
                "total_runs": len(group),
                "reliable_runs": len(reliable),
                "reliable_success_rate_percent": 100.0 * len(reliable) / len(group),
                "mean_psnr_reliable": average_or_blank(numeric(reliable, "psnr")),
                "mean_ssim_reliable": average_or_blank(numeric(reliable, "ssim")),
                "mean_changed_pixel_ratio": average_or_blank(
                    numeric(reliable, "changed_pixel_ratio")
                ),
            }
        )
    return summary


def create_charts(rows: list[dict], output_dir: Path) -> None:
    reliable = [
        row
        for row in rows
        if row.get("extraction_success") is True
        and float(row.get("ber", 1.0)) == 0.0
    ]

    by_method_size: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in reliable:
        by_method_size[(row["method"], int(row["requested_payload_bytes"]))].append(row)

    methods = sorted({row["method"] for row in rows})
    sizes = sorted({int(row["requested_payload_bytes"]) for row in rows})

    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    for method in methods:
        x_values, y_values = [], []
        for size in sizes:
            group = by_method_size.get((method, size), [])
            values = numeric(group, "psnr")
            if values:
                x_values.append(size)
                y_values.append(mean(values))
        if x_values:
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=method)
    axis.set_xscale("log")
    axis.set_xlabel("Original payload bytes")
    axis.set_ylabel("Mean reliable PSNR (dB)")
    axis.set_title("Multi-Cover PSNR by Payload Size")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "multi_cover_psnr.png", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    for method in methods:
        x_values, y_values = [], []
        for size in sizes:
            group = by_method_size.get((method, size), [])
            values = numeric(group, "ssim")
            if values:
                x_values.append(size)
                y_values.append(mean(values))
        if x_values:
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=method)
    axis.set_xscale("log")
    axis.set_xlabel("Original payload bytes")
    axis.set_ylabel("Mean reliable SSIM")
    axis.set_title("Multi-Cover SSIM by Payload Size")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "multi_cover_ssim.png", bbox_inches="tight")
    plt.close(fig)

    method_summary = build_method_summary(rows)
    names = [row["method"] for row in method_summary]
    rates = [float(row["reliable_success_rate_percent"]) for row in method_summary]

    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    bars = axis.bar(names, rates, color="#1aa6a6")
    axis.set_ylim(0, 105)
    axis.set_ylabel("BER-zero extraction success (%)")
    axis.set_title("Multi-Cover Reliability by Method")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    for bar, rate in zip(bars, rates, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1,
            f"{rate:.1f}%",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "multi_cover_reliability.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GuardianPixel baseline comparisons over multiple covers."
    )
    parser.add_argument("--covers-dir", required=True)
    parser.add_argument("--cover-features-csv", required=True)
    parser.add_argument(
        "--payload-sizes",
        type=parse_sizes,
        default=parse_sizes("5000,20000,50000"),
    )
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument(
        "--output",
        default="experiments/results/baseline_batch",
    )
    args = parser.parse_args()

    if not os.environ.get("GUARDIANPIXEL_EXPERIMENT_PASSPHRASE"):
        raise SystemExit(
            "Set GUARDIANPIXEL_EXPERIMENT_PASSPHRASE before running."
        )

    covers_dir = Path(args.covers_dir)
    if not covers_dir.is_dir():
        raise SystemExit(f"Cover directory not found: {covers_dir}")

    category_lookup = load_category_lookup(Path(args.cover_features_csv))
    covers = sorted(
        path
        for path in covers_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not covers:
        raise SystemExit("No supported cover images found.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    errors: list[dict] = []
    total_jobs = len(covers) * len(args.payload_sizes)
    job_number = 0

    for cover in covers:
        feature_row = category_lookup.get(cover.name)
        if feature_row is None:
            print(f"Skipping {cover.name}: not found in cover-feature CSV.")
            continue

        category = feature_row["cover_category"]
        texture_score = float(feature_row["texture_score"])

        for payload_size in args.payload_sizes:
            job_number += 1
            run_name = f"{cover.stem}_{payload_size}B"
            run_dir = output_dir / "runs" / run_name

            print(
                f"[{job_number}/{total_jobs}] {cover.name} | "
                f"{category} | {payload_size:,} bytes"
            )

            command = [
                sys.executable,
                "-m",
                "scripts.run_baseline_comparison",
                "--cover",
                str(cover),
                "--payload-bytes",
                str(payload_size),
                "--channel",
                str(args.channel),
                "--output",
                str(run_dir),
            ]

            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                env=os.environ.copy(),
            )

            if completed.returncode != 0:
                error = {
                    "cover": cover.name,
                    "category": category,
                    "payload_bytes": payload_size,
                    "return_code": completed.returncode,
                    "stderr": completed.stderr[-3000:],
                }
                errors.append(error)
                print("  Failed:", completed.stderr.strip().splitlines()[-1])
                continue

            json_path = run_dir / "baseline_comparison.json"
            if not json_path.is_file():
                errors.append(
                    {
                        "cover": cover.name,
                        "category": category,
                        "payload_bytes": payload_size,
                        "return_code": 0,
                        "stderr": "Result JSON was not generated.",
                    }
                )
                continue

            result = json.loads(json_path.read_text(encoding="utf-8"))
            for method_row in result["methods"]:
                all_rows.append(
                    {
                        "cover_category": category,
                        "texture_score": texture_score,
                        "requested_payload_bytes": payload_size,
                        **method_row,
                    }
                )

    if not all_rows:
        raise SystemExit("No baseline result rows were generated.")

    runs_path = output_dir / "baseline_runs.csv"
    method_path = output_dir / "baseline_method_summary.csv"
    category_path = output_dir / "baseline_category_summary.csv"

    method_summary = build_method_summary(all_rows)
    category_summary = build_category_summary(all_rows)

    write_csv(all_rows, runs_path)
    write_csv(method_summary, method_path)
    write_csv(category_summary, category_path)
    create_charts(all_rows, output_dir)

    (output_dir / "batch_errors.json").write_text(
        json.dumps(errors, indent=2),
        encoding="utf-8",
    )

    print("")
    print("Multi-cover baseline batch completed.")
    print(f"Covers: {len(covers)}")
    print(f"Payload levels: {len(args.payload_sizes)}")
    print(f"Method rows: {len(all_rows)}")
    print(f"Failed comparison jobs: {len(errors)}")
    print(f"Runs CSV: {runs_path}")
    print(f"Method summary: {method_path}")
    print(f"Category summary: {category_path}")
    print(f"Charts: {output_dir}")


if __name__ == "__main__":
    main()
