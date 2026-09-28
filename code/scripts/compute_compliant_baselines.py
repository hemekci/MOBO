"""Compute the code-compliant baseline (B2) for each case.

For each case, take the as-practised baseline (B1) and increase member_depth
until utilization <= 1.0 (strength-limit-state per ASCE 7-22). Reports both
B1 (as-practised; may violate code) and B2 (code-compliant minimum) metrics
side by side, plus apples-to-apples deltas of the framework's best-compromise
design vs.\\ B2.

Output: data/results/baselines_b1_b2.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, UPPER
from mobo_envelope.objectives import report_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

# As-practised baselines (B1) — from run_optimization.py
B1_BASELINES: dict[str, dict[str, float]] = {
    "A_Houston": {
        "wwr_n": 0.55, "wwr_s": 0.55, "wwr_e": 0.55, "wwr_w": 0.55,
        "insulation_m": 0.10, "glazing_idx": 1.0,
        "member_depth_m": 0.067, "framing_idx": 1.0,
        "overhang": 0.0, "panel_t": 0.006,
    },
    "B_NYC": {
        "wwr_n": 0.40, "wwr_s": 0.40, "wwr_e": 0.40, "wwr_w": 0.40,
        "insulation_m": 0.12, "glazing_idx": 1.0,
        "member_depth_m": 0.080, "framing_idx": 1.0,
        "overhang": 0.0, "panel_t": 0.006,
    },
    "C_Minneapolis": {
        "wwr_n": 0.20, "wwr_s": 0.25, "wwr_e": 0.20, "wwr_w": 0.20,
        "insulation_m": 0.20, "glazing_idx": 1.0,
        "member_depth_m": 0.140, "framing_idx": 2.0,
        "overhang": 0.0, "panel_t": 0.006,
    },
}


def _to_x(bl: dict[str, float]) -> np.ndarray:
    return np.array([
        bl["wwr_n"], bl["wwr_s"], bl["wwr_e"], bl["wwr_w"],
        bl["insulation_m"], bl["glazing_idx"],
        bl["member_depth_m"], bl["framing_idx"],
        bl["overhang"], bl["panel_t"],
    ])


def _find_min_compliant_depth(
    bl: dict[str, float], city: str, target_util: float = 0.95
) -> tuple[float, dict]:
    """Increase member_depth until structural utilization <= target_util."""
    upper_depth = float(UPPER[6])  # member_depth bound
    depths = np.linspace(bl["member_depth_m"], upper_depth, 80)
    for d in depths:
        bl_test = {**bl, "member_depth_m": float(d)}
        m = report_metrics(_to_x(bl_test), city)
        if m["utilization"] <= target_util:
            return float(d), m
    # Fallback: max depth
    bl_test = {**bl, "member_depth_m": upper_depth}
    m = report_metrics(_to_x(bl_test), city)
    return upper_depth, m


def _delta(opt: dict, bl: dict) -> dict:
    return {
        "u_eff_pct":   (opt["u_eff_w_m2k"] / bl["u_eff_w_m2k"] - 1) * 100,
        "reserve_pp":  (opt["structural_reserve"] - bl["structural_reserve"]),
        "sda_pp":       opt["sda_pct"] - bl["sda_pct"],
        "cost_pct":    (opt["cost_usd_m2"] / bl["cost_usd_m2"] - 1) * 100,
    }


def main() -> None:
    # Read framework's best-compromise from existing summary
    opt_path = RESULTS_DIR / "optimization_summary.json"
    with open(opt_path) as f:
        opt_summary = json.load(f)
    bc_metrics: dict[str, dict] = {
        item["case"]: item["best_compromise"]["metrics"] for item in opt_summary
    }

    output: dict[str, dict] = {}
    print(f"{'Case':<18}{'B1 depth':>10}{'B1 util':>10}{'B2 depth':>10}{'B2 util':>10}{'Δ_BC vs B2':>40}")
    print("-" * 100)
    for case_key, bl in B1_BASELINES.items():
        case = CASES[case_key.split("_")[0]]
        b1 = report_metrics(_to_x(bl), case.city)
        d2, b2 = _find_min_compliant_depth(bl, case.city)
        bc = bc_metrics[case_key]
        d_bc_b2 = _delta(bc, b2)
        print(
            f"{case_key:<18}{bl['member_depth_m']:>10.3f}{b1['utilization']:>10.3f}"
            f"{d2:>10.3f}{b2['utilization']:>10.3f}"
            f"  ΔU={d_bc_b2['u_eff_pct']:+5.1f}%  Δres={d_bc_b2['reserve_pp']:+5.2f}"
            f"  ΔsDA={d_bc_b2['sda_pp']:+5.1f}pp  Δ$={d_bc_b2['cost_pct']:+5.1f}%"
        )
        output[case_key] = {
            "b1_as_practised": {"design": bl, "metrics": b1},
            "b2_code_compliant": {
                "design": {**bl, "member_depth_m": d2},
                "metrics": b2,
            },
            "best_compromise_metrics": bc,
            "delta_bc_vs_b1": _delta(bc, b1),
            "delta_bc_vs_b2": _delta(bc, b2),
        }

    with open(RESULTS_DIR / "baselines_b1_b2.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'baselines_b1_b2.json'}")


if __name__ == "__main__":
    main()
