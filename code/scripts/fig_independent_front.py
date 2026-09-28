"""fig_independent_front.pdf — framework front vs an independently optimised
high-fidelity front, as a function of the high-fidelity evaluation budget.

The framework's front is compared against a
reference front the framework did not produce, obtained by running NSGA-III
directly on the Path-C objectives (EnergyPlus + Radiance + FEM in the loop).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pymoo.indicators.hv import HV

REPO = Path(__file__).resolve().parents[2]
R = REPO / "data" / "results"
FIG = REPO / "figures" / "fig_independent_front.pdf"
POP = 165


def _nd(X):
    n = len(X); m = np.ones(n, bool)
    for i in range(n):
        if not m[i]: continue
        d = X - X[i]
        dom = np.all(d <= 1e-12, axis=1) & np.any(d < -1e-12, axis=1); dom[i] = False
        if dom.any(): m[i] = False
    return m


def main() -> None:
    fw = pd.read_csv(R / "path_c_full_front_A_Houston.csv"); fw = fw[fw.success]
    F = np.column_stack([fw.total_eui_kwh_m2_yr_path_c_v2, fw.structural_utilization_path_c,
                         -fw.sda_pct_path_c_v2, fw.cost_usd_m2]).astype(float)
    F = F[F[:, 1] <= 1.0]; Ff = F[_nd(F)]

    d = pd.read_csv(R / "path_c_direct_opt_A_Houston_archive.csv")
    # Generation 0 is the initial population; it is part of the evaluation
    # budget and is included so the axis matches the 3,960 evaluations reported.
    d = d[d.success]
    D = np.column_stack([d.total_eui_kwh_m2_yr, d.utilization, -d.sda_pct, d.cost_usd_m2]).astype(float)
    Dall = D[D[:, 1] <= 1.0]; Df = Dall[_nd(Dall)]

    both = np.vstack([Ff, Df]); ref = both.max(axis=0) + 0.05 * np.abs(both.max(axis=0))
    ind = HV(ref_point=ref); hv_fw = float(ind(Ff))

    gens = sorted(d.generation.unique()); budgets, hvs = [], []
    for g in gens:
        X = np.column_stack([d[d.generation <= g].total_eui_kwh_m2_yr,
                             d[d.generation <= g].utilization,
                             -d[d.generation <= g].sda_pct,
                             d[d.generation <= g].cost_usd_m2]).astype(float)
        X = X[X[:, 1] <= 1.0]
        if len(X) < 2: continue
        budgets.append((int(g) + 1) * POP); hvs.append(float(ind(X[_nd(X)])))
    budgets, hvs = np.array(budgets), np.array(hvs)

    cross = None
    for i in range(1, len(hvs)):
        if hvs[i - 1] < hv_fw <= hvs[i]:
            t = (hv_fw - hvs[i - 1]) / (hvs[i] - hvs[i - 1])
            cross = budgets[i - 1] + t * (budgets[i] - budgets[i - 1]); break

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(budgets, hvs, "o-", color="#1f77b4", lw=1.6, ms=4,
            label="Direct high-fidelity NSGA-III")
    ax.axhline(hv_fw, ls="--", color="#d6622e", lw=1.4,
               label="MOBO-Envelope front (0 high-fidelity evaluations)")
    if cross:
        ax.axvline(cross, ls=":", color="grey", lw=1.0)
        ax.annotate(f"crossover\n$\\approx${cross:,.0f} evaluations",
                    xy=(cross, hv_fw), xytext=(cross * 1.15, hv_fw * 0.86),
                    fontsize=7.5, color="grey",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    ax.set_xscale("log")
    # A log axis over 165-3,960 draws exactly one labelled decade tick, which
    # leaves the reader unable to read off the budgets quoted in the text.
    # Label the budget-matched points reported in the manuscript instead.
    ticks = [165, 330, 825, 1204, 2000, 3960]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks], fontsize=7.5)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("high-fidelity (EnergyPlus + Radiance) evaluations")
    ax.set_ylabel("hypervolume (common reference point)")
    ax.set_title("Framework front vs. independently optimised high-fidelity front, Houston",
                 loc="left", pad=6)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG, bbox_inches="tight")
    print(f"Wrote {FIG}")
    print(f"crossover ~= {cross:,.0f} high-fidelity evaluations" if cross else "no crossover")
    json.dump({"hv_framework": hv_fw, "budgets": budgets.tolist(), "hv_direct": hvs.tolist(),
               "crossover_evals": float(cross) if cross else None},
              open(R / "independent_front_curve_A_Houston.json", "w"), indent=2)


if __name__ == "__main__":
    main()
