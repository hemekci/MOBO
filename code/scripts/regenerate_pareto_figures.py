"""Regenerate the Pareto-front and variable-distribution figures from real CSVs.

Publication-quality matplotlib outputs sourced directly from
data/results/case_*_pareto.csv.

Outputs:
  - figures/fig_pareto_houston.pdf  (4-panel 2D projections of the Pareto front)
  - figures/fig_variable_distributions.pdf  (boxplot of design variables across cases)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CASES = {
    "A_Houston":     "case_a_houston_pareto.csv",
    "B_NYC":         "case_b_nyc_pareto.csv",
    "C_Minneapolis": "case_c_minneapolis_pareto.csv",
}

PALETTE = {"A_Houston": "#EF4444", "B_NYC": "#3B82F6", "C_Minneapolis": "#22C55E"}


def fig_pareto_houston() -> None:
    df = pd.read_csv(RESULTS_DIR / CASES["A_Houston"])
    # Headline B2 baseline for Houston (from baselines_b1_b2.json)
    bl = {"u": 0.596, "reserve": 0.080, "sda": 59.0, "cost": 720.0}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    # Panel (a): U_eff vs sDA
    ax = axes[0]
    sc = ax.scatter(df.u_eff_w_m2k, df.sda_pct, c=df.cost_usd_m2, cmap="viridis",
                    s=14, alpha=0.75, edgecolor="none")
    ax.scatter([bl["u"]], [bl["sda"]], marker="*", s=180, c="red",
               edgecolor="black", linewidth=0.8, zorder=5, label="B2 baseline")
    ax.set_xlabel(r"$U_\mathrm{eff}$ [W/(m$^2$K)]")
    ax.set_ylabel(r"sDA [%]")
    ax.set_title(r"(a) $U_\mathrm{eff}$ vs. sDA  (colour: cost)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(r"cost [USD/m$^2$]", fontsize=8)

    # Panel (b): structural reserve vs cost
    ax = axes[1]
    sc = ax.scatter(df.cost_usd_m2, df.structural_reserve, c=df.sda_pct, cmap="plasma",
                    s=14, alpha=0.75, edgecolor="none")
    ax.scatter([bl["cost"]], [bl["reserve"]], marker="*", s=180, c="red",
               edgecolor="black", linewidth=0.8, zorder=5, label="B2 baseline")
    ax.set_xlabel(r"cost [USD/m$^2$]")
    ax.set_ylabel(r"$\eta_S$ structural reserve")
    ax.set_title(r"(b) cost vs. $\eta_S$  (colour: sDA)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(r"sDA [%]", fontsize=8)

    fig.suptitle(
        f"Case A: Houston Pareto front  (n = {len(df)} non-dominated designs)",
        fontsize=10, weight="bold",
    )
    out = FIG_DIR / "fig_pareto_houston.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_variable_distributions() -> None:
    """Box plots of key design variables across the three Pareto fronts."""
    variables = [
        ("wwr_south",      r"WWR$_S$"),
        ("wwr_north",      r"WWR$_N$"),
        ("wwr_east",       r"WWR$_E$"),
        ("wwr_west",       r"WWR$_W$"),
        ("insulation_m",   r"insulation $t$ [m]"),
        ("member_depth_m", r"member depth [m]"),
        ("overhang_ratio", r"overhang ratio"),
    ]
    cases = list(CASES.keys())
    n_var = len(variables)

    fig, axes = plt.subplots(1, n_var, figsize=(11.0, 2.6), constrained_layout=True, sharey=False)
    if n_var == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, variables):
        data, colors = [], []
        for case in cases:
            df = pd.read_csv(RESULTS_DIR / CASES[case])
            data.append(df[col].values)
            colors.append(PALETTE[case])
        bp = ax.boxplot(
            data, patch_artist=True, widths=0.65,
            tick_labels=["A", "B", "C"],
            medianprops=dict(color="black", linewidth=1.4),
            flierprops=dict(marker=".", markersize=2, markerfacecolor="grey", markeredgecolor="grey"),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Design-variable distributions across Pareto-optimal designs (A: Houston, B: NYC, C: Minneapolis)",
        fontsize=10, weight="bold",
    )
    out = FIG_DIR / "fig_variable_distributions.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "savefig.dpi": 220,
    })
    fig_pareto_houston()
    fig_variable_distributions()


if __name__ == "__main__":
    main()
