"""Create a clean light-mode baseline comparison slide from GuardianPixel JSON."""

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


def fmt(value, digits: int) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--output",
        default="experiments/results/baseline_slide_light.png",
    )
    args = parser.parse_args()

    result_path = Path(args.result_json)
    if not result_path.is_file():
        raise SystemExit(f"Result JSON not found: {result_path}")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    rows = data.get("methods", [])
    if not rows:
        raise SystemExit("No method rows found in JSON.")

    embedded_bits = {int(row["embedded_bits"]) for row in rows}
    total_bpp = {round(float(row["total_bpp"]), 12) for row in rows}
    if len(embedded_bits) != 1 or len(total_bpp) != 1:
        raise SystemExit("Methods do not use equal embedded bits and BPP.")

    equal_bits = next(iter(embedded_bits))
    equal_bpp = next(iter(total_bpp))

    reliable = [
        row
        for row in rows
        if row.get("extraction_success") is True
        and float(row.get("ber", 1.0)) == 0.0
    ]
    if not reliable:
        raise SystemExit("No reliable BER-zero methods found.")

    best_psnr = max(reliable, key=lambda row: float(row["psnr"]))
    best_ssim = max(reliable, key=lambda row: float(row["ssim"]))
    fewest = min(reliable, key=lambda row: int(row["changed_pixels"]))

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="white")
    gs = fig.add_gridspec(
        3,
        12,
        height_ratios=[0.72, 3.1, 2.55],
        hspace=0.34,
        wspace=0.30,
    )

    # Header
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(
        0.0,
        0.77,
        "Equal-BPP Baseline Comparison",
        fontsize=25,
        fontweight="bold",
        color="#12345b",
        va="center",
    )
    ax.text(
        0.0,
        0.18,
        (
            f"Preliminary single-cover result  |  {data.get('width')}×{data.get('height')}  |  "
            f"Same embedded bits: {equal_bits:,}  |  Same total BPP: {equal_bpp:.6f}"
        ),
        fontsize=12.5,
        color="#187b86",
        va="center",
    )

    # Main table
    table_ax = fig.add_subplot(gs[1, :8])
    table_ax.axis("off")
    columns = ["Method", "Recovery", "BER", "PSNR\n(dB)", "SSIM", "Changed\npixels", "Embed\ntime (s)"]
    values = []
    for row in rows:
        values.append(
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

    table = table_ax.table(
        cellText=values,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.26, 0.11, 0.12, 0.12, 0.14, 0.13, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.85)

    for (ri, ci), cell in table.get_celld().items():
        cell.set_edgecolor("#aebdca")
        if ri == 0:
            cell.set_facecolor("#12345b")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            method = rows[ri - 1]["method"]
            if method == "GuardianPixel":
                cell.set_facecolor("#d9f3ea")
                cell.get_text().set_fontweight("bold")
            elif method == "Canny Synchronous":
                cell.set_facecolor("#fde1e3")
            else:
                cell.set_facecolor("#f3f6f8")
            cell.get_text().set_color("#14283d")

    # Key conclusions panel
    result_ax = fig.add_subplot(gs[1, 8:])
    result_ax.set_facecolor("#f4f9fb")
    result_ax.set_xticks([])
    result_ax.set_yticks([])
    for spine in result_ax.spines.values():
        spine.set_color("#1aa6a6")
        spine.set_linewidth(1.4)

    lines = [
        ("KEY FINDINGS", "#0b7f87", 12.5, "bold"),
        (f"✓ Reliable BER-zero methods: {len(reliable)}/{len(rows)}", "#17324d", 10.5, "normal"),
        (f"✓ Best reliable PSNR: {best_psnr['method']}", "#17324d", 10.5, "normal"),
        (f"✓ Best reliable SSIM: {best_ssim['method']}", "#17324d", 10.5, "normal"),
        (f"✓ Fewest reliable changes: {fewest['method']}", "#17324d", 10.5, "normal"),
        ("", "#17324d", 7, "normal"),
        ("GUARDIANPIXEL", "#0b7f87", 12.5, "bold"),
        ("✓ Exact extraction: BER = 0", "#17324d", 10.5, "normal"),
        (f"✓ PSNR: {fmt(rows[0]['psnr'], 4)} dB", "#17324d", 10.5, "normal"),
        (f"✓ SSIM: {fmt(rows[0]['ssim'], 6)}", "#17324d", 10.5, "normal"),
        (f"✓ Changed pixels: {int(rows[0]['changed_pixels']):,}", "#17324d", 10.5, "normal"),
        ("", "#17324d", 7, "normal"),
        ("TRADE-OFF", "#b46715", 12.0, "bold"),
        # ("• Higher sender time due to HED + texture analysis", "#4e6173", 9.5, "normal"),
        ("• Canny is excluded from ranking because BER > 0", "#9b2d35", 9.5, "normal"),
    ]

    y = 0.94
    for text, color, size, weight in lines:
        result_ax.text(
            0.055,
            y,
            text,
            transform=result_ax.transAxes,
            color=color,
            fontsize=size,
            fontweight=weight,
            va="top",
            wrap=True,
        )
        y -= 0.071 if text else 0.035

    # Difference-map thumbnails
    thumbs = gs[2, :].subgridspec(1, len(rows), wspace=0.12)
    base_dir = result_path.parent

    for index, row in enumerate(rows):
        axis = fig.add_subplot(thumbs[0, index])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#f6f7f8")

        filename = DIFFERENCE_FILES.get(row["method"])
        path = base_dir / filename if filename else None

        if path and path.is_file():
            image = np.asarray(Image.open(path).convert("RGB"))
            axis.imshow(image)
        else:
            axis.text(
                0.5,
                0.5,
                "Difference image\nnot generated",
                ha="center",
                va="center",
                color="#697b8c",
                transform=axis.transAxes,
            )

        title_color = "#087f72" if row["method"] == "GuardianPixel" else "#17324d"
        if row["method"] == "Canny Synchronous":
            title_color = "#b4232c"

        axis.set_title(
            SHORT_NAMES.get(row["method"], row["method"]),
            fontsize=9.5,
            color=title_color,
            fontweight="bold",
            pad=5,
        )
        axis.set_xlabel(
            f"BER {fmt(row['ber'], 3)}",
            fontsize=8,
            color="#56697a",
        )

        for spine in axis.spines.values():
            spine.set_color("#1aa6a6" if row["method"] == "GuardianPixel" else "#aebdca")
            spine.set_linewidth(1.3 if row["method"] == "GuardianPixel" else 0.8)

    fig.text(
        0.5,
        0.012,
        (

        ),
        ha="center",
        fontsize=10.2,
        color="#8a570e",
        fontweight="bold",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Light-mode presentation graphic created: {output}")


if __name__ == "__main__":
    main()
