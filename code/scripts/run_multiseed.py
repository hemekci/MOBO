"""Multi-seed optimization runs to bootstrap 95% confidence intervals.

For each case, run NSGA-III + GP surrogate with 5 different random seeds.
Aggregate:
  - Hypervolume of each run's final Pareto front (using a fixed reference point)
  - Best-compromise metrics with bootstrap CIs
  - Surrogate CV R^2 across seeds
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from mobo_envelope.cases import CASES
from mobo_envelope.objectives import N_OBJ, evaluate_batch, report_metrics
from mobo_envelope.optimize import OptimizeConfig, run_optimization
from mobo_envelope.postprocess import (
    best_compromise_index,
    cost_constrained_compromise_index,
    pareto_to_dataframe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

SEEDS = [42, 7, 101, 2025, 314]


def _bootstrap_ci(values: list[float], conf: float = 0.95, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) by bootstrap percentile method."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1 - conf) / 2
    return float(arr.mean()), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def main() -> None:
    summary: dict[str, dict] = {}

    for case_key, case in CASES.items():
        print(f"\n=== Case {case_key}: {case.name} ===")

        per_seed: list[dict] = []
        all_hv: list[float] = []

        for seed in SEEDS:
            cfg = OptimizeConfig(seed=seed)
            t0 = time.time()
            res = run_optimization(case.city, cfg)
            elapsed = time.time() - t0

            # Hypervolume: reference point is the worst observed in each objective + 5%
            ref_pt = res.f.max(axis=0) * 1.05
            ref_pt[res.f.max(axis=0) < 0] = res.f.max(axis=0)[res.f.max(axis=0) < 0] * 0.95
            hv = HV(ref_point=ref_pt)
            hv_value = float(hv(res.pareto_f))
            all_hv.append(hv_value)

            bc_idx = best_compromise_index(res.pareto_f)
            bc_metrics = report_metrics(res.pareto_x[bc_idx], case.city)

            per_seed.append({
                "seed": seed,
                "n_pareto": int(res.pareto_x.shape[0]),
                "n_full_eval": res.n_full_eval,
                "wall_clock_s": elapsed,
                "hypervolume": hv_value,
                "best_compromise": bc_metrics,
                "surrogate_cv_r2": [float(v) for v in res.surrogate_metrics["r2"]],
            })

            print(f"  seed={seed}  HV={hv_value:.4g}  pareto={res.pareto_x.shape[0]}  best=U{bc_metrics['u_eff_w_m2k']:.2f}/cost${bc_metrics['cost_usd_m2']:.0f}  ({elapsed:.0f}s)")

        # Bootstrap CIs across seeds
        hv_mean, hv_lo, hv_hi = _bootstrap_ci([s["hypervolume"] for s in per_seed])
        np_mean, np_lo, np_hi = _bootstrap_ci([s["n_pareto"] for s in per_seed])
        bc_u_mean, bc_u_lo, bc_u_hi = _bootstrap_ci([s["best_compromise"]["u_eff_w_m2k"] for s in per_seed])
        bc_sda_mean, bc_sda_lo, bc_sda_hi = _bootstrap_ci([s["best_compromise"]["sda_pct"] for s in per_seed])
        bc_cost_mean, bc_cost_lo, bc_cost_hi = _bootstrap_ci([s["best_compromise"]["cost_usd_m2"] for s in per_seed])

        summary[case.name] = {
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "hypervolume":      {"mean": hv_mean, "ci95_low": hv_lo, "ci95_high": hv_hi},
            "n_pareto":         {"mean": np_mean, "ci95_low": np_lo, "ci95_high": np_hi},
            "best_compromise_u_eff": {"mean": bc_u_mean, "ci95_low": bc_u_lo, "ci95_high": bc_u_hi},
            "best_compromise_sda":   {"mean": bc_sda_mean, "ci95_low": bc_sda_lo, "ci95_high": bc_sda_hi},
            "best_compromise_cost":  {"mean": bc_cost_mean, "ci95_low": bc_cost_lo, "ci95_high": bc_cost_hi},
            "per_seed": per_seed,
        }
        print(f"  HV {hv_mean:.4g} [95% CI: {hv_lo:.4g}, {hv_hi:.4g}]")
        print(f"  n_pareto {np_mean:.0f} [95% CI: {np_lo:.0f}, {np_hi:.0f}]")

    with open(RESULTS_DIR / "multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote multiseed_summary.json")


if __name__ == "__main__":
    main()
