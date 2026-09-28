"""Run NSGA-III + GP surrogate on all three cases and write CSVs.

Reports three preference-articulated points per case:
  1. Best-compromise (minimum Euclidean distance to ideal on normalized front)
  2. Thermal-priority (minimum U_eff)
  3. Cost-constrained best-compromise (subject to cost <= baseline)

Surrogate accuracy is reported via 5-fold cross-validation (more rigorous than
a single holdout).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mobo_envelope.cases import CASES
from mobo_envelope.objectives import N_OBJ, evaluate_batch, report_metrics
from mobo_envelope.optimize import OptimizeConfig, run_optimization
from mobo_envelope.postprocess import (
    best_compromise_index,
    cost_constrained_compromise_index,
    pareto_to_dataframe,
    thermal_priority_index,
)
from mobo_envelope.surrogate import SurrogateBank


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _baseline_metrics(case_name: str) -> dict[str, float]:
    """Baseline = competent professional sequential design.

    These baselines reflect what licensed architects + structural engineers
    + energy consultants would typically deliver under sequential design,
    not the ASHRAE 90.1-2022 prescriptive minimum. Premium glazing, slim
    aluminum framing, code-plus insulation. Urban wind exposure means low
    structural reserve.

    Each tuple: (wwr, insul_m, glazing_idx, member_depth_m, framing_idx, overhang)
    """
    case = CASES[case_name]
    # Per-case baseline encoding (continuous design vector with categorical idx)
    BASELINES = {
        "A": {
            # Houston high-rise office: triple-LowE curtain wall, slim aluminum
            # mullion (heavily loaded under hurricane wind), WWR=0.55, no
            # overhang. Sequential design produced a structurally-borderline
            # mullion (reserve ~ 0.08) — typical of cost-driven curtain wall.
            "wwr_n": 0.55, "wwr_s": 0.55, "wwr_e": 0.55, "wwr_w": 0.55,
            "insulation_m": 0.10, "glazing_idx": 1.0,   # triple_lowe
            "member_depth_m": 0.067, "framing_idx": 1.0, # aluminum, thin
            "overhang": 0.0, "panel_t": 0.006,
        },
        "B": {
            # NYC mid-rise mixed-use: punched-window facade, triple-LowE,
            # code-plus insulation, conventional aluminum windows
            "wwr_n": 0.40, "wwr_s": 0.40, "wwr_e": 0.40, "wwr_w": 0.40,
            "insulation_m": 0.12, "glazing_idx": 1.0,
            "member_depth_m": 0.080, "framing_idx": 1.0,  # aluminum
            "overhang": 0.0, "panel_t": 0.006,
        },
        "C": {
            # Minneapolis residential: small windows for thermal, triple-LowE,
            # cold-climate insulation, dimensional-lumber stud wall (2x6)
            "wwr_n": 0.20, "wwr_s": 0.25, "wwr_e": 0.20, "wwr_w": 0.20,
            "insulation_m": 0.20, "glazing_idx": 1.0,
            "member_depth_m": 0.140, "framing_idx": 2.0,  # timber 2x6
            "overhang": 0.0, "panel_t": 0.006,
        },
    }
    bl = BASELINES[case_name]
    x = np.array([
        bl["wwr_n"], bl["wwr_s"], bl["wwr_e"], bl["wwr_w"],
        bl["insulation_m"], bl["glazing_idx"],
        bl["member_depth_m"], bl["framing_idx"],
        bl["overhang"], bl["panel_t"],
    ])
    return report_metrics(x, case.city)


def _delta(opt: dict[str, float], bl: dict[str, float]) -> dict[str, float]:
    return {
        "u_eff_pct":              (opt["u_eff_w_m2k"] / bl["u_eff_w_m2k"] - 1) * 100,
        "structural_reserve_pct": (opt["structural_reserve"] / bl["structural_reserve"] - 1) * 100 if bl["structural_reserve"] > 0 else float("nan"),
        "sda_pp":                  opt["sda_pct"] - bl["sda_pct"],
        "cost_pct":               (opt["cost_usd_m2"] / bl["cost_usd_m2"] - 1) * 100,
    }


def main() -> None:
    cfg = OptimizeConfig()
    summary: list[dict] = []

    for case_key, case in CASES.items():
        print(f"\n=== Case {case_key}: {case.name} ===")
        baseline = _baseline_metrics(case_key)
        print(f"  baseline: U={baseline['u_eff_w_m2k']:.3f}, reserve={baseline['structural_reserve']:.3f}, sda={baseline['sda_pct']:.1f}, cost=${baseline['cost_usd_m2']:.0f}")

        t0 = time.time()
        result = run_optimization(case.city, cfg)
        elapsed = time.time() - t0

        pareto_df = pareto_to_dataframe(result, case.city)
        pareto_path = RESULTS_DIR / f"case_{case_key.lower()}_{case.city.lower()}_pareto.csv"
        pareto_df.to_csv(pareto_path, index=False)

        # Three preference-articulated points
        bc_idx = best_compromise_index(result.pareto_f)
        tp_idx = thermal_priority_index(result.pareto_f)
        cc_idx = cost_constrained_compromise_index(result.pareto_f, baseline["cost_usd_m2"])

        bc_metrics = report_metrics(result.pareto_x[bc_idx], case.city)
        tp_metrics = report_metrics(result.pareto_x[tp_idx], case.city)
        cc_metrics = report_metrics(result.pareto_x[cc_idx], case.city) if cc_idx is not None else None

        # 5-fold CV surrogate accuracy on the analytical training set
        cv = SurrogateBank.cv_metrics(
            result.x, result.f, n_obj=N_OBJ, n_splits=5, seed=42
        )

        # All-four-dominator search
        dominators = pareto_df[
            (pareto_df.u_eff_w_m2k < baseline["u_eff_w_m2k"]) &
            (pareto_df.structural_reserve > baseline["structural_reserve"]) &
            (pareto_df.sda_pct > baseline["sda_pct"]) &
            (pareto_df.cost_usd_m2 < baseline["cost_usd_m2"])
        ]

        print(f"  pareto size: {len(pareto_df)}, dominators of baseline: {len(dominators)}")
        print(f"  wall-clock:  {elapsed:.1f} s")
        print(f"  best-compromise: U={bc_metrics['u_eff_w_m2k']:.3f} ({_delta(bc_metrics, baseline)['u_eff_pct']:+.1f}%), reserve={bc_metrics['structural_reserve']:.3f}, sda={bc_metrics['sda_pct']:.1f} ({_delta(bc_metrics, baseline)['sda_pp']:+.1f}pp), cost=${bc_metrics['cost_usd_m2']:.0f} ({_delta(bc_metrics, baseline)['cost_pct']:+.1f}%)")
        print(f"  thermal-priority: U={tp_metrics['u_eff_w_m2k']:.3f}, sda={tp_metrics['sda_pct']:.1f}, cost=${tp_metrics['cost_usd_m2']:.0f}")
        if cc_metrics is not None:
            print(f"  cost-constrained: U={cc_metrics['u_eff_w_m2k']:.3f}, sda={cc_metrics['sda_pct']:.1f}, cost=${cc_metrics['cost_usd_m2']:.0f} ({_delta(cc_metrics, baseline)['cost_pct']:+.1f}%)")
        print(f"  surrogate CV R^2: thermal={cv['r2'][0]:.3f}, struct={cv['r2'][1]:.3f}, daylight={cv['r2'][2]:.3f}, cost={cv['r2'][3]:.3f}")

        summary.append({
            "case": case.name,
            "climate_zone": case.climate_zone,
            "n_pareto": len(pareto_df),
            "n_dominators": len(dominators),
            "n_full_eval": result.n_full_eval,
            "n_full_pop_evals_equivalent": result.n_total_pop_evals,
            "wall_clock_s": elapsed,
            "baseline": baseline,
            "best_compromise": {"metrics": bc_metrics, "deltas": _delta(bc_metrics, baseline)},
            "thermal_priority": {"metrics": tp_metrics, "deltas": _delta(tp_metrics, baseline)},
            "cost_constrained": {"metrics": cc_metrics, "deltas": _delta(cc_metrics, baseline) if cc_metrics else None},
            "surrogate_cv": cv,
            "surrogate_holdout": result.surrogate_metrics,
        })

    # Surrogate accuracy CSV (5-fold CV per case x objective)
    sa_rows: list[dict] = []
    obj_names_pretty = ["thermal", "structural", "daylight", "cost"]
    for entry in summary:
        for j, on in enumerate(obj_names_pretty):
            sa_rows.append({
                "objective": on,
                "case": entry["case"],
                "rmse_cv5": round(entry["surrogate_cv"]["rmse"][j], 4),
                "r_squared_cv5": round(entry["surrogate_cv"]["r2"][j], 3),
                "mape_pct_cv5": round(entry["surrogate_cv"]["mape_pct"][j], 2),
            })
    pd.DataFrame(sa_rows).to_csv(RESULTS_DIR / "surrogate_accuracy.csv", index=False)

    # Computational cost CSV
    cc_rows: list[dict] = []
    for entry in summary:
        cc_rows.append({
            "case": entry["case"],
            "full_sim_calls": entry["n_full_pop_evals_equivalent"],
            "surrogate_sim_calls": entry["n_full_eval"],
            "full_time_hours": round(entry["wall_clock_s"] * (entry["n_full_pop_evals_equivalent"] / max(1, entry["n_full_eval"])) / 3600, 2),
            "surrogate_time_hours": round(entry["wall_clock_s"] / 3600, 4),
            "speedup": round(entry["n_full_pop_evals_equivalent"] / max(1, entry["n_full_eval"]), 1),
        })
    pd.DataFrame(cc_rows).to_csv(RESULTS_DIR / "computational_cost.csv", index=False)

    # Best-compromise summary CSV
    bc_rows: list[dict] = []
    for entry in summary:
        for selector in ("best_compromise", "thermal_priority", "cost_constrained"):
            sel = entry[selector]
            if sel["metrics"] is None:
                continue
            row = {"case": entry["case"], "selector": selector}
            row.update(sel["metrics"])
            row.update({f"delta_{k}": v for k, v in sel["deltas"].items()})
            bc_rows.append(row)
    pd.DataFrame(bc_rows).to_csv(RESULTS_DIR / "best_compromise.csv", index=False)

    with open(RESULTS_DIR / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
