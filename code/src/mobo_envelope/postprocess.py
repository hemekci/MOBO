"""Pareto post-processing: best-compromise selection and CSV export."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mobo_envelope.envelope import VARIABLE_NAMES, decode_design
from mobo_envelope.objectives import report_metrics
from mobo_envelope.optimize import OptimizeResult


def normalize(f: np.ndarray) -> np.ndarray:
    lo = f.min(axis=0)
    hi = f.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (f - lo) / span


def best_compromise_index(pareto_f: np.ndarray) -> int:
    """Minimum Euclidean distance to the ideal point on the normalized front."""
    fn = normalize(pareto_f)
    ideal = fn.min(axis=0)
    distances = np.linalg.norm(fn - ideal, axis=1)
    return int(np.argmin(distances))


def thermal_priority_index(pareto_f: np.ndarray) -> int:
    """Lowest U_eff (objective 0)."""
    return int(np.argmin(pareto_f[:, 0]))


def cost_constrained_compromise_index(
    pareto_f: np.ndarray, baseline_cost: float
) -> int | None:
    """Best-compromise on the sub-front with cost <= baseline_cost.
    Returns None if no Pareto point meets the cost constraint.
    """
    cost_mask = pareto_f[:, 3] <= baseline_cost
    if not cost_mask.any():
        return None
    sub = pareto_f[cost_mask]
    fn = normalize(sub)
    ideal = fn.min(axis=0)
    distances = np.linalg.norm(fn - ideal, axis=1)
    sub_idx = int(np.argmin(distances))
    # map back to original index
    original_indices = np.where(cost_mask)[0]
    return int(original_indices[sub_idx])


def pareto_to_dataframe(result: OptimizeResult, city: str) -> pd.DataFrame:
    """Return per-solution metrics + design vector in human-readable form."""
    rows: list[dict[str, float | str | int]] = []
    for i, x in enumerate(result.pareto_x):
        d = decode_design(x)
        m = report_metrics(x, city)
        row: dict[str, float | str | int] = {
            "solution_id": i + 1,
            "u_eff_w_m2k": round(m["u_eff_w_m2k"], 3),
            "structural_reserve": round(m["structural_reserve"], 3),
            "sda_pct": round(m["sda_pct"], 1),
            "cost_usd_m2": round(m["cost_usd_m2"], 0),
            "wwr_north": round(d.wwr_north, 3),
            "wwr_south": round(d.wwr_south, 3),
            "wwr_east": round(d.wwr_east, 3),
            "wwr_west": round(d.wwr_west, 3),
            "insulation_m": round(d.insulation_m, 3),
            "glazing_type": d.glazing_type,
            "member_depth_m": round(d.member_depth_m, 3),
            "framing_material": d.framing_material,
            "overhang_ratio": round(d.overhang_ratio, 3),
            "panel_thickness_m": round(d.panel_thickness_m, 4),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("u_eff_w_m2k").reset_index(drop=True)
