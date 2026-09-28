"""Generate the full set of publication-quality figures from real CSVs/JSONs.

Outputs (in /figures/):
    fig_pareto_houston.pdf            (existing)
    fig_variable_distributions.pdf    (existing)
    fig_surrogate_accuracy.pdf        (NEW)
    fig_sobol_heatmap.pdf             (NEW)
    fig_path_c_verification.pdf       (NEW)
    fig_algo_baseline.pdf             (NEW)
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER
from mobo_envelope.objectives import evaluate_batch
from scipy.stats import qmc, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"A_Houston": "#EF4444", "B_NYC": "#3B82F6", "C_Minneapolis": "#22C55E"}
OBJ_NAMES = ["U_eff", r"$\log(1+u)$", "sDA", "cost"]
OBJ_UNITS = [r"$U_\mathrm{eff}$ [W/(m$^2$K)]", r"$\log(1+u)$ utilisation",
             r"sDA [%]", r"cost [USD/m$^2$]"]


def _build_kernel(n_features: int):
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
    )


def fig_surrogate_accuracy() -> None:
    """4-panel CV predicted-vs-true scatter, three cases overlaid."""
    sampler = qmc.LatinHypercube(d=N_VARS, seed=42)
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.7), constrained_layout=True)
    for case_key, case in CASES.items():
        u = sampler.random(n=200)
        X = qmc.scale(u, LOWER, UPPER)
        Y = evaluate_batch(X, case.city)
        Y[:, 1] = np.log1p(Y[:, 1])  # match training transform
        Y[:, 2] = -Y[:, 2]
        # Get CV predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        Y_pred = np.zeros_like(Y)
        for j in range(4):
            for tr, te in kf.split(X):
                gpr = GaussianProcessRegressor(
                    kernel=_build_kernel(N_VARS), normalize_y=True,
                    alpha=1e-8, n_restarts_optimizer=2,
                )
                gpr.fit(X[tr], Y[tr, j])
                Y_pred[te, j] = gpr.predict(X[te])
        case_label = case.name.split("_")[1] if "_" in case.name else case.name
        for j in range(4):
            ax = axes[j]
            ax.scatter(Y[:, j], Y_pred[:, j], s=10, alpha=0.55,
                       color=PALETTE[case.name], label=case_label,
                       edgecolor="none")
    for j, ax in enumerate(axes):
        # 1:1 line
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8, alpha=0.7)
        # OBJ_UNITS carries the typeset symbol and the unit; OBJ_NAMES is the
        # bare code-style name and left the axes unitless with a raw "U_eff".
        ax.set_xlabel(f"true {OBJ_UNITS[j]}", fontsize=8)
        ax.set_ylabel(f"predicted {OBJ_UNITS[j]}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Gaussian-process surrogate accuracy (5-fold CV) across four objectives and three cases",
                 fontsize=10, weight="bold")
    out = FIG_DIR / "fig_surrogate_accuracy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _fmt2(v: float) -> str:
    """Two-decimal string using round-half-up.

    Binary floating point puts values such as 0.865 and 0.245 marginally below
    the .xx5 boundary, so ``f"{v:.2f}"`` rounds them down while the manuscript
    tables and prose round them up. Formatting through ``Decimal`` on the
    repr keeps figure annotations and table entries identical.
    """
    return str(Decimal(repr(round(float(v), 6))).quantize(Decimal("0.01"),
                                                   rounding=ROUND_HALF_UP))


def _contrast_text_colour(cmap_name: str, value: float, vmin: float, vmax: float) -> str:
    """Pick black or white for maximum contrast against the cell's fill colour.

    Uses WCAG relative luminance rather than a fixed value threshold: a raw
    ``value > k`` test inverts on perceptually non-monotonic colormaps and put
    white text on bright viridis/plasma cells and black text on dark ones.
    """
    span = (vmax - vmin) or 1.0
    frac = min(max((value - vmin) / span, 0.0), 1.0)
    r, g, b = plt.get_cmap(cmap_name)(frac)[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.55 else "white"


def fig_sobol_heatmap() -> None:
    """Heatmap of Path-A and Path-B Sobol total-effect indices per case."""
    df_a = pd.read_csv(RESULTS_DIR / "sensitivity_analysis.csv")
    df_b = pd.read_csv(RESULTS_DIR / "sensitivity_pathb_thermal.csv")

    cases = ["A_Houston", "B_NYC", "C_Minneapolis"]
    case_lab = ["Houston", "NYC", "Mpls"]

    # Path-A: 4 objectives, all cases identical → just show one row per objective
    obj_lab = {"thermal": "Therm.", "structural": "Struct.", "daylight": "Daylight", "cost": "Cost"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Panel (a): Path-A Sobol S1 — 4 objectives × variables (Houston only since invariant)
    ax = axes[0]
    pivot_a = df_a[df_a.case == "A_Houston"].pivot(
        index="objective", columns="variable", values="sobol_s1"
    )
    pivot_a = pivot_a.reindex(index=["thermal", "structural", "daylight", "cost"])
    pivot_a.index = [obj_lab[i] for i in pivot_a.index]
    var_order = ["wwr_north", "wwr_south", "wwr_east", "wwr_west",
                 "insulation_m", "glazing_idx", "member_depth_m",
                 "framing_idx", "overhang_ratio", "panel_thickness_m"]
    var_lab = ["WWR$_N$", "WWR$_S$", "WWR$_E$", "WWR$_W$", "ins. $t$",
               "glaz. idx", "depth", "framing", "overhang", "panel $t$"]
    pivot_a = pivot_a.reindex(columns=var_order)
    im1 = ax.imshow(pivot_a.values, cmap="viridis", aspect="auto", vmin=0, vmax=0.9)
    ax.set_xticks(range(len(var_order)))
    ax.set_xticklabels(var_lab, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(pivot_a.index)))
    ax.set_yticklabels(pivot_a.index, fontsize=12)
    ax.set_title("(a) Path-A Sobol $S_1$ on analytical objectives\n(climate-invariant by construction)",
                 fontsize=11)
    for i in range(pivot_a.shape[0]):
        for j in range(pivot_a.shape[1]):
            v = pivot_a.values[i, j]
            ax.text(j, i, _fmt2(v), ha="center", va="center",
                    color=_contrast_text_colour("viridis", v, 0.0, 0.9),
                    fontsize=10, fontweight="bold")
    cb1 = plt.colorbar(im1, ax=ax, shrink=0.85, label=r"$S_1$")
    cb1.ax.tick_params(labelsize=10); cb1.set_label(r"$S_1$", fontsize=11)

    # Panel (b): Path-B Sobol S_T per case (climate-dependent)
    ax = axes[1]
    pivot_b = df_b.pivot(index="case", columns="variable", values="sobol_st")
    pivot_b = pivot_b.reindex(index=cases, columns=var_order)
    pivot_b.index = case_lab
    im2 = ax.imshow(pivot_b.values, cmap="plasma", aspect="auto", vmin=0, vmax=0.65)
    ax.set_xticks(range(len(var_order)))
    ax.set_xticklabels(var_lab, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(pivot_b.index)))
    ax.set_yticklabels(pivot_b.index, fontsize=12)
    ax.set_title("(b) Path-B Sobol $S_T$ on annual EUI (ISO 13790)\n(climate-dependent)",
                 fontsize=11)
    for i in range(pivot_b.shape[0]):
        for j in range(pivot_b.shape[1]):
            v = pivot_b.values[i, j]
            ax.text(j, i, _fmt2(v), ha="center", va="center",
                    color=_contrast_text_colour("plasma", v, 0.0, 0.65),
                    fontsize=10, fontweight="bold")
    cb2 = plt.colorbar(im2, ax=ax, shrink=0.85, label=r"$S_T$")
    cb2.ax.tick_params(labelsize=10); cb2.set_label(r"$S_T$", fontsize=11)

    fig.suptitle("Sobol global sensitivity across objectives and climates",
                 fontsize=12, weight="bold")
    out = FIG_DIR / "fig_sobol_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_path_c_verification() -> None:
    """Path-A vs Path-C agreement scatter: thermal, daylight, structural per case."""
    df = pd.read_csv(RESULTS_DIR / "path_c_subset.csv")
    cases = ["A_Houston", "B_NYC", "C_Minneapolis"]
    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.0), constrained_layout=True)

    for col_idx, case in enumerate(cases):
        sub = df[df["case"] == case].copy()
        sub = sub[sub["eplus_success"] & sub["radiance_success"]]
        # Thermal: U_eff (Path-A) vs total EUI (Path-C)
        ax = axes[0, col_idx]
        x = sub["u_eff_w_m2k_path_a"].to_numpy()
        y = sub["total_eui_kwh_m2_yr_path_c"].to_numpy()
        rho, _ = spearmanr(x, y)
        ax.scatter(x, y, s=20, color=PALETTE[case], alpha=0.75, edgecolor="none")
        ax.set_xlabel(r"$U_\mathrm{eff}$ Path-A [W/(m$^2$K)]", fontsize=8)
        ax.set_ylabel("EUI Path-C [kWh/m$^2$/yr]", fontsize=8)
        ax.set_title(f"{case.split('_')[1]} thermal  $\\rho$={rho:.2f}", fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)

        # Daylight
        ax = axes[1, col_idx]
        x = sub["sda_pct_path_a"].to_numpy()
        y = sub["sda_pct_path_c"].to_numpy()
        rho, _ = spearmanr(x, y)
        ax.scatter(x, y, s=20, color=PALETTE[case], alpha=0.75, edgecolor="none")
        ax.plot([0, 100], [0, 100], "--", color="grey", lw=0.7, alpha=0.6)
        ax.set_xlabel("sDA Path-A [%]", fontsize=8)
        ax.set_ylabel("sDA Path-C (Radiance) [%]", fontsize=8)
        ax.set_title(f"{case.split('_')[1]} daylight  $\\rho$={rho:.2f}", fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)

        # Structural
        ax = axes[2, col_idx]
        x = (1.0 - sub["structural_reserve_path_a"]).to_numpy()  # utilization
        y = sub["structural_utilization_path_c"].to_numpy()
        rho, _ = spearmanr(x, y)
        ax.scatter(x, y, s=20, color=PALETTE[case], alpha=0.75, edgecolor="none")
        m = max(x.max(), y.max()) * 1.05
        ax.plot([0, m], [0, m], "--", color="grey", lw=0.7, alpha=0.6)
        ax.set_xlabel("util Path-A", fontsize=8)
        ax.set_ylabel("util Path-C (FEA)", fontsize=8)
        ax.set_title(f"{case.split('_')[1]} structural  $\\rho$={rho:.2f}", fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Path-A analytical vs. Path-C high-fidelity-simulator agreement (n = 20 designs/case)",
                 fontsize=10, weight="bold")
    out = FIG_DIR / "fig_path_c_verification.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_algo_baseline() -> None:
    """HV trajectory: NSGA-II vs NSGA-III no-surrogate vs surrogate-assisted (~338)."""
    with open(RESULTS_DIR / "algorithmic_baseline.json") as f:
        d = json.load(f)
    cases_keys = list(d.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0),
                             constrained_layout=True, sharey=False)
    for ax, case_key in zip(axes, cases_keys):
        case_data = d[case_key]
        budgets = case_data["budgets"]
        nsga2 = [r["hv"] for r in case_data["nsga2"]]
        nsga3 = [r["hv"] for r in case_data["nsga3_no_surrogate"]]
        ax.plot(budgets, nsga2, "o-", color="#94A3B8", label="NSGA-II", lw=1.5, markersize=5)
        ax.plot(budgets, nsga3, "s-", color="#3B82F6", label="NSGA-III (no surrogate)", lw=1.5, markersize=5)
        # Surrogate HVs recomputed against the SAME fixed reference point as the
        # baseline curves (sensitivity_seed_robustness_consistent_ref.csv); the
        # earlier multiseed values used a run-adaptive reference point and are
        # not comparable to these axes.
        cons = pd.read_csv(RESULTS_DIR / "sensitivity_seed_robustness_consistent_ref.csv")
        long_key = {"A": "A_Houston", "B": "B_NYC", "C": "C_Minneapolis"}.get(case_key, case_key)
        g = cons[cons.case == long_key]
        sa = None
        if len(g):
            v = g.hv_surrogate.to_numpy()
            rng = np.random.default_rng(0)
            boots = rng.choice(v, size=(10_000, len(v)), replace=True).mean(axis=1)
            sa = {"n_full_evals_per_run": float(g.n_full_eval.mean()),
                  "hv_mean": float(v.mean()),
                  "hv_ci95_low": float(np.percentile(boots, 2.5)),
                  "hv_ci95_high": float(np.percentile(boots, 97.5))}
        if sa:
            ax.errorbar(
                [sa["n_full_evals_per_run"]], [sa["hv_mean"]],
                yerr=[[sa["hv_mean"] - sa["hv_ci95_low"]],
                      [sa["hv_ci95_high"] - sa["hv_mean"]]],
                fmt="*", color="#DC2626", markersize=14, capsize=4, lw=2.0,
                label="Surrogate-NSGA-III (5 seeds)",
            )
        ax.set_xscale("log")
        ax.margins(x=0.10, y=0.15)  # keep the surrogate star clear of the axes frame
        ax.set_xlabel("full-fidelity evaluations", fontsize=8)
        ax.set_ylabel("hypervolume (HV)", fontsize=8)
        case_lab = {"A": "A: Houston", "B": "B: NYC",
                    "C": "C: Minneapolis"}.get(case_key, case_key)
        ax.set_title(case_lab, fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Algorithmic-baseline ablation: HV vs. evaluation budget (single fixed reference point)",
                 fontsize=10, weight="bold")
    out = FIG_DIR / "fig_algo_baseline.pdf"
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
    fig_surrogate_accuracy()
    fig_sobol_heatmap()
    fig_path_c_verification()
    fig_algo_baseline()


if __name__ == "__main__":
    main()
