"""Unified analytical objective vector for the optimizer."""

from __future__ import annotations

import numpy as np

from mobo_envelope import cost as cost_mod
from mobo_envelope import daylight, structural, thermal
from mobo_envelope.envelope import decode_design

# Objective names in canonical order. All four are minimized internally:
# thermal: minimize U_eff
# structural: minimize utilization (== maximize reserve)
# daylight: minimize -sda  (== maximize sda)
# cost: minimize $/m^2
OBJECTIVE_NAMES: list[str] = ["U_eff", "utilization", "neg_sda", "cost"]
N_OBJ: int = 4


def evaluate(x: np.ndarray, city: str) -> np.ndarray:
    """Evaluate the four-objective vector for a single design."""
    d = decode_design(x)
    f1 = thermal.u_eff(d)
    f2 = structural.utilization(d, city)
    f3 = -daylight.sda(d, city)
    f4 = cost_mod.cost_per_envelope_m2(d)
    return np.array([f1, f2, f3, f4], dtype=float)


def evaluate_batch(xs: np.ndarray, city: str) -> np.ndarray:
    """Vectorized over rows of xs (n, n_vars) -> (n, 4)."""
    out = np.empty((xs.shape[0], N_OBJ), dtype=float)
    for i in range(xs.shape[0]):
        out[i] = evaluate(xs[i], city)
    return out


def report_metrics(x: np.ndarray, city: str) -> dict[str, float]:
    """Human-readable metrics for a single design (sda positive, reserve positive)."""
    d = decode_design(x)
    return {
        "u_eff_w_m2k": float(thermal.u_eff(d)),
        "structural_reserve": float(structural.reserve(d, city)),
        "utilization": float(structural.utilization(d, city)),
        "sda_pct": float(daylight.sda(d, city)),
        "cost_usd_m2": float(cost_mod.cost_per_envelope_m2(d)),
    }
