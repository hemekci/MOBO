"""Generate fig_path_c_v2_comparison.pdf:

Two-panel figure comparing Path-C v1 (5ZoneAirCooled IDF + south-only Radiance)
to Path-C v2 (DOE RefBldg Small Office + 4-orientation Radiance) on the
Houston Pareto subset.

Panel A: bar chart of Path-A vs Path-C Spearman rho per axis (thermal /
         daylight / structural) for v1 and v2.
Panel B: per-design sDA distribution showing the saturation effect that
         compresses the daylight rank correlation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "results"
FIG = REPO / "figures" / "fig_path_c_v2_comparison.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main() -> None:
    with open(RESULTS / "path_c_metrics.json") as f:
        v1m = json.load(f)
    with open(RESULTS / "path_c_v2_houston_metrics.json") as f:
        v2m = json.load(f)
    h_v1 = v1m["A_Houston"]
    rho_v1 = [
        h_v1["thermal_path_a_vs_c"]["spearman"],
        h_v1["daylight_path_a_vs_c"]["spearman"],
        h_v1["structural_path_a_vs_c"]["spearman"],
    ]
    rho_v2 = [
        v2m["thermal_path_a_vs_c_v2"]["spearman"],
        v2m["daylight_path_a_vs_c_v2"]["spearman"],
        v2m["structural_path_a_vs_c"]["spearman"],
    ]

    df_v1 = pd.read_csv(RESULTS / "path_c_subset.csv")
    df_v1 = df_v1[df_v1["case"] == "A_Houston"].copy()
    df_v2 = pd.read_csv(RESULTS / "path_c_v2_houston_subset.csv")

    sda_v1 = df_v1["sda_pct_path_c"].to_numpy(dtype=float)
    sda_v2 = df_v2["sda_pct_path_c_v2"].to_numpy(dtype=float)
    sat_v1 = (sda_v1 >= 99.5).mean() * 100.0
    sat_v2 = (sda_v2 >= 99.5).mean() * 100.0

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    # Panel A: bar chart of Spearman rho
    axes_labels = ["Thermal", "Daylight", "Structural"]
    x = np.arange(len(axes_labels))
    bw = 0.36
    cv1 = "#1f77b4"
    cv2 = "#d6622e"
    bA1 = axA.bar(x - bw / 2, rho_v1, width=bw, color=cv1, edgecolor="black",
                  linewidth=0.5, label="Path-C v1\n(5ZoneAirCooled, S-only)")
    bA2 = axA.bar(x + bw / 2, rho_v2, width=bw, color=cv2, edgecolor="black",
                  linewidth=0.5, label="Path-C v2\n(DOE RefBldg, 4-orient.)")
    for bars, vals in ((bA1, rho_v1), (bA2, rho_v2)):
        for b, v in zip(bars, vals):
            axA.text(b.get_x() + b.get_width() / 2, v + 0.035, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7.5)
    axA.set_xticks(x)
    axA.set_xticklabels(axes_labels)
    axA.set_ylabel(r"Path-A vs. Path-C Spearman $\rho$")
    axA.set_ylim(0, 1.55)
    axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.axhline(0.7, ls=":", color="gray", lw=0.7, alpha=0.7)
    # Bars occupy x <= 2.36; widening the axis reserves a clear column at the
    # right end of the reference line for its label, so neither the bars nor
    # the line itself are overprinted.
    axA.set_xlim(-0.45, 3.05)
    axA.text(3.0, 0.715, "0.70 floor", color="gray", fontsize=6.5,
             ha="right", va="bottom")
    axA.grid(axis="y", ls=":", alpha=0.4)
    axA.set_title("(a) Cross-tier rank correlation (Houston, $n = 20$)",
                  loc="left", pad=4, fontsize=9)
    # A two-column legend of two-line labels is wider than the axes box, so it
    # spilled past the left spine and the axis line printed through the text.
    # One column anchored inside the upper-left corner keeps it within the box;
    # the 1.55 y-limit above reserves the headroom it needs.
    axA.legend(loc="upper left", frameon=False, ncols=1, fontsize=7.5,
               handlelength=1.5, labelspacing=0.5, borderaxespad=0.4)

    # Panel B: sDA distribution (jitter + saturation rate)
    rng = np.random.default_rng(0)
    j_v1 = rng.uniform(-0.18, 0.18, size=len(sda_v1))
    j_v2 = rng.uniform(-0.18, 0.18, size=len(sda_v2))
    axB.scatter(np.zeros_like(sda_v1) + j_v1, sda_v1, s=22, color=cv1,
                edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3,
                label="Path-C v1")
    axB.scatter(np.ones_like(sda_v2) + j_v2, sda_v2, s=22, color=cv2,
                edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3,
                label="Path-C v2")
    axB.axhline(100, ls="--", color="black", lw=0.5, alpha=0.6)
    axB.text(-0.55, 103.0, "sDA = 100% saturation", fontsize=7,
             color="black", alpha=0.7, ha="left", va="center")
    axB.text(0, 50, f"{sat_v1:.0f}% at\nsDA = 100%",
             ha="center", va="center", fontsize=8, color=cv1,
             bbox=dict(facecolor="white", edgecolor=cv1, lw=0.6, pad=2))
    axB.text(1, 50, f"{sat_v2:.0f}% at\nsDA = 100%",
             ha="center", va="center", fontsize=8, color=cv2,
             bbox=dict(facecolor="white", edgecolor=cv2, lw=0.6, pad=2))
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["v1 (S-only)", "v2 (4-orient.)"])
    axB.set_xlim(-0.6, 1.6)
    axB.set_ylabel("Path-C sDA (%)")
    axB.set_ylim(20, 110)
    axB.set_yticks([25, 50, 75, 100])
    axB.grid(axis="y", ls=":", alpha=0.4)
    axB.set_title(r"(b) sDA saturation explains $\rho$ compression",
                  loc="left", pad=4, fontsize=9)

    # The two panel titles are left-aligned and long; without extra horizontal
    # separation the end of (a) runs into the start of (b) on the same line.
    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIG, bbox_inches="tight")
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
