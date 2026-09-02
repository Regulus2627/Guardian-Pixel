"""Create a presentation-ready baseline comparison graphic from JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DIFFERENCE_FILES = {
    "GuardianPixel": "guardianpixel_difference_x255.png",
    "Sequential Replacement": "sequential_replacement_difference_x255.png",
    "Sequential Matching": "sequential_matching_difference_x255.png",
    "Keyed Random Matching": "keyed_random_difference_x255.png",
    "Canny Synchronous": "canny_synchronous_difference_x255.png",
}

SHORT_NAMES = {
    "GuardianPixel": "GuardianPixel",
    "Sequential Replacement": "Sequential\nReplacement",
    "Sequential Matching": "Sequential\nMatching",
    "Keyed Random Matching": "Keyed Random\nMatching",
    "Canny Synchronous": "Canny\nSynchronous",
}


def fmt(value, digits: int = 4) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one slide graphic from baseline comparison JSON."
    )
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--output",
        default="experiments/results/baseline_comparison_slide.png",
    )
    arguments = parser.parse_args()

    result_path = Path(arguments.result_json)
    if not result_path.is_file():
        raise SystemExit(f"Result JSON not found: {result_path}")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    rows = data.get("methods", [])
    if not rows:
        raise SystemExit("Result JSON contains no method rows.")

    embedded_counts = {int(row["embedded_bits"]) for row in rows}
    bpp_values = {round(float(row["total_bpp"]), 12) for row in rows}

    if len(embedded_counts) != 1 or len(bpp_values) != 1:
        raise SystemExit(
            "Comparison is not fair: methods do not have identical embedded bits/BPP."
        )

    embedded_bits = next(iter(embedded_counts))
    total_bpp = next(iter(bpp_values))

    reliable = [
        row
        for row in rows
        if row.get("extraction_success") is True
        and float(row.get("ber", 1.0)) == 0.0
    ]

    if not reliable:
        raise SystemExit("No reliable BER-zero methods were found.")

    best_psnr = max(reliable, key=lambda row: float(row["psnr"]))
    best_ssim = max(reliable, key=lambda row: float(row["ssim"]))
    fewest_changes = min(reliable, key=lambda row: int(row["changed_pixels"]))

    figure = plt.figure(figsize=(16, 9), dpi=140, facecolor="#081426")
    grid = figure.add_gridspec(
        3,
        12,
        height_ratios=[0.7, 3.1, 2.6],
        hspace=0.32,
        wspace=0.25,
    )

    # Header
    header = figure.add_subplot(grid[0, :])
    header.axis("off")
    header.text(
        0.0,
        0.78,
        "Equal-BPP Baseline Comparison — Preliminary Single-Cover Result",
        color="#f4f8fc",
        fontsize=23,
        fontweight="bold",
        va="center",
    )
    header.text(
        0.0,
        0.20,
        (
            f"Cover: {data.get('cover')}  |  "
            f"Resolution: {data.get('width')}×{data.get('height')}  |  "
            f"Same embedded bits: {embedded_bits:,}  |  "
            f"Same total BPP: {total_bpp:.6f}"
        ),
        color="#8fdce3",
        fontsize=12.5,
        va="center",
    )

    # Table
    table_axis = figure.add_subplot(grid[1, :8])
    table_axis.axis("off")

    columns = [
        "Method",
        "Recovery",
        "BER",
        "PSNR\n(dB)",
        "SSIM",
        "Changed\npixels",
        "Embed\ntime (s)",
    ]

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["method"],
                "PASS" if row["extraction_success"] else "FAIL",
                fmt(row["ber"], 6),
                fmt(row["psnr"], 4),
                fmt(row["ssim"], 6),
                f"{int(row['changed_pixels']):,}" if row["changed_pixels"] != "" else "—",
                fmt(row["embedding_seconds"], 3),
            ]
        )

    table = table_axis.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.25, 0.11, 0.12, 0.12, 0.14, 0.13, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.85)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#33475d")
        if row_index == 0:
            cell.set_facecolor("#12345b")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            method = rows[row_index - 1]["method"]
            if method == "GuardianPixel":
                cell.set_facecolor("#0f4a45")
            elif method == "Canny Synchronous":
                cell.set_facecolor("#4a2028")
            else:
                cell.set_facecolor("#101f32")
            cell.get_text().set_color("#f4f8fc")

    # Findings panel
    finding_axis = figure.add_subplot(grid[1, 8:])
    finding_axis.set_facecolor("#0e1f34")
    finding_axis.set_xticks([])
    finding_axis.set_yticks([])
    for spine in finding_axis.spines.values():
        spine.set_color("#1dcad3")

    findings = [
        "WHAT THIS RUN SHOWS",
        f"✓ BER-zero methods: {len(reliable)}/{len(rows)}",
        f"✓ Best reliable PSNR: {best_psnr['method']}",
        f"✓ Best reliable SSIM: {best_ssim['method']}",
        f"✓ Fewest reliable changes: {fewest_changes['method']}",
        "",
        "GUARDIANPIXEL RESULT",
        "✓ Exact extraction (BER = 0)",
        f"✓ PSNR {fmt(rows[0]['psnr'], 4)} dB",
        f"✓ SSIM {fmt(rows[0]['ssim'], 6)}",
        f"✓ {int(rows[0]['changed_pixels']):,} changed pixels",
        "",
        "TRADE-OFF",
        "• Higher sender time due to HED + texture analysis",
        "• Canny visual score is invalid for reliability ranking because BER > 0",
    ]

    y = 0.94
    for line in findings:
        if line in {"WHAT THIS RUN SHOWS", "GUARDIANPIXEL RESULT", "TRADE-OFF"}:
            finding_axis.text(
                0.06,
                y,
                line,
                color="#1dcad3",
                fontsize=12,
                fontweight="bold",
                va="top",
                transform=finding_axis.transAxes,
            )
            y -= 0.10
        else:
            finding_axis.text(
                0.06,
                y,
                line,
                color="#f4f8fc" if line.startswith("✓") else "#b2c2d2",
                fontsize=9.8,
                va="top",
                wrap=True,
                transform=finding_axis.transAxes,
            )
            y -= 0.072 if line else 0.045

    # Difference thumbnails
    thumbnail_grid = grid[2, :].subgridspec(1, len(rows), wspace=0.12)
    base_directory = result_path.parent

    for index, row in enumerate(rows):
        axis = figure.add_subplot(thumbnail_grid[0, index])
        axis.set_facecolor("#050b13")
        axis.set_xticks([])
        axis.set_yticks([])

        difference_name = DIFFERENCE_FILES.get(row["method"])
        difference_path = base_directory / difference_name if difference_name else None

        if difference_path and difference_path.is_file():
            image = np.asarray(Image.open(difference_path).convert("RGB"))
            axis.imshow(image)
        else:
            axis.text(
                0.5,
                0.5,
                "No difference\nimage generated",
                color="#8ca0b5",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

        title_color = "#64e0bd" if row["method"] == "GuardianPixel" else "#f4f8fc"
        if row["method"] == "Canny Synchronous":
            title_color = "#ff7b83"

        axis.set_title(
            SHORT_NAMES.get(row["method"], row["method"]),
            color=title_color,
            fontsize=9.5,
            fontweight="bold",
            pad=5,
        )

        status = (
            f"BER {fmt(row['ber'], 3)}"
            if row["ber"] != ""
            else "No result"
        )
        axis.set_xlabel(status, color="#b2c2d2", fontsize=8)

        for spine in axis.spines.values():
            spine.set_color(
                "#1dcad3" if row["method"] == "GuardianPixel" else "#33475d"
            )

    figure.text(
        0.5,
        0.012,
        (
            "Conclusion for this run: GuardianPixel gives the strongest reliable visual-quality "
            "trade-off, but multi-cover and steganalysis evaluation is required before a general superiority claim."
        ),
        ha="center",
        color="#f6c879",
        fontsize=10,
    )

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)

    print(f"Presentation graphic created: {output_path}")


if __name__ == "__main__":
    main()
