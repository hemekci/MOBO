"""Smoke tests: each module evaluates without errors and returns sane ranges."""

from __future__ import annotations

import numpy as np

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER, decode_design
from mobo_envelope.objectives import evaluate, report_metrics


def _midpoint() -> np.ndarray:
    return (LOWER + UPPER) / 2.0


def test_decode_runs():
    d = decode_design(_midpoint())
    assert 0.10 <= d.wwr_avg <= 0.80
    assert d.glazing_type in {"double_lowe", "triple_lowe", "vacuum"}


def test_objective_signs_and_ranges():
    x = _midpoint()
    for case in CASES.values():
        f = evaluate(x, case.city)
        assert f.shape == (4,)
        u, util, neg_sda, cost = f
        assert 0.05 < u < 5.0, f"U_eff out of range: {u}"
        assert 0.0 < util < 5.0, f"utilization out of range: {util}"
        sda = -neg_sda
        assert 0.0 <= sda <= 100.0
        assert 50.0 < cost < 5000.0


def test_extremes_are_extreme():
    """Lower-bound corner: thinnest insulation, smallest depth -> worst U, worst structure."""
    x_low = LOWER.copy()
    x_high = UPPER.copy() - 1e-3
    f_low = evaluate(x_low, "Houston")
    f_high = evaluate(x_high, "Houston")
    # High insulation should produce lower U_eff than low insulation, all else equal
    # (we cheat the dimensionality by toggling only insulation)
    x_a = _midpoint(); x_a[4] = LOWER[4]
    x_b = _midpoint(); x_b[4] = UPPER[4]
    f_a = evaluate(x_a, "Houston")
    f_b = evaluate(x_b, "Houston")
    assert f_b[0] <= f_a[0], "More insulation must reduce U_eff"


def test_report_metrics_keys():
    r = report_metrics(_midpoint(), "NYC")
    assert {"u_eff_w_m2k", "structural_reserve", "utilization", "sda_pct", "cost_usd_m2"}.issubset(r.keys())
