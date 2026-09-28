"""Algorithmic baselines.

Compares the surrogate-assisted NSGA-III pipeline (Path-A in run_optimization.py)
against vanilla NSGA-II without surrogates on the same analytical problem at
multiple evaluation budgets. Reports hypervolume (HV, against the same fixed
reference point used in tab:pareto_stats) versus number of full-fidelity
evaluations.

Output: data/results/algorithmic_baseline.json + console table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER
from mobo_envelope.objectives import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

# Reference point matched to tab:pareto_stats (worst-observed dominator)
HV_REF = np.array([1.0, 1.5, -10.0, 1500.0], dtype=float)
SEED = 42
BUDGETS = [500, 1000, 2500, 5000, 10000]


class AnalyticalProblem(Problem):
    def __init__(self, city: str):
        self.city = city
        super().__init__(
            n_var=N_VARS, n_obj=4, n_ieq_constr=1,
            xl=LOWER, xu=UPPER,
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        n = X.shape[0]
        F = np.empty((n, 4), dtype=float)
        G = np.empty((n, 1), dtype=float)
        for i in range(n):
            f = evaluate(X[i], self.city)
            F[i] = f
            G[i, 0] = f[1] - 1.0  # utilisation <= 1.0
        out["F"] = F
        out["G"] = G


def _run_nsga2(city: str, n_evals: int, seed: int) -> tuple[float, int, float]:
    """Run NSGA-II without surrogates; return (HV, n_full_evals, wall_clock)."""
    pop = 100
    n_gen = max(1, n_evals // pop)
    problem = AnalyticalProblem(city)
    algo = NSGA2(pop_size=pop)
    t0 = time.time()
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    wall = time.time() - t0
    F = res.F if res.F is not None else None
    if F is None or len(F) == 0:
        return float("nan"), pop * n_gen, wall
    hv = HV(ref_point=HV_REF)
    # filter feasible
    feas_mask = res.G[:, 0] <= 0 if res.G is not None else np.ones(len(F), dtype=bool)
    F_feas = F[feas_mask]
    if len(F_feas) == 0:
        return float("nan"), pop * n_gen, wall
    return float(hv(F_feas)), int(pop * n_gen), wall


def _run_nsga3_no_surrogate(city: str, n_evals: int, seed: int) -> tuple[float, int, float]:
    """Run NSGA-III on the analytical problem (no surrogate); return (HV, n_evals, wall_clock).

    This isolates the contribution of the surrogate from the algorithm choice.
    """
    pop = 100
    n_gen = max(1, n_evals // pop)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=8)
    problem = AnalyticalProblem(city)
    algo = NSGA3(pop_size=pop, ref_dirs=ref_dirs)
    t0 = time.time()
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    wall = time.time() - t0
    F = res.F if res.F is not None else None
    if F is None or len(F) == 0:
        return float("nan"), pop * n_gen, wall
    hv = HV(ref_point=HV_REF)
    feas_mask = res.G[:, 0] <= 0 if res.G is not None else np.ones(len(F), dtype=bool)
    F_feas = F[feas_mask]
    if len(F_feas) == 0:
        return float("nan"), pop * n_gen, wall
    return float(hv(F_feas)), int(pop * n_gen), wall


def main() -> None:
    summary: dict = {}
    print(f"\n{'='*72}\nAlgorithmic baseline: vanilla NSGA-II vs surrogate-assisted NSGA-III\n{'='*72}")
    print(f"HV reference point: {HV_REF}")
    print(f"Seed: {SEED}\n")

    for case_key, case in CASES.items():
        print(f"\n=== {case.name} ===")
        case_data = {"budgets": BUDGETS, "nsga2": [], "nsga3_no_surrogate": []}
        for b in BUDGETS:
            hv2, n2, w2 = _run_nsga2(case.city, b, SEED)
            hv3, n3, w3 = _run_nsga3_no_surrogate(case.city, b, SEED)
            case_data["nsga2"].append({"hv": hv2, "n_evals": n2, "wall_s": w2})
            case_data["nsga3_no_surrogate"].append({"hv": hv3, "n_evals": n3, "wall_s": w3})
            print(f"  budget={b:>5}  NSGA-II  HV={hv2:>10.0f}  ({w2:.1f}s)   "
                  f"NSGA-III(no-surr) HV={hv3:>10.0f}  ({w3:.1f}s)")
        summary[case_key] = case_data

    # Insert reference: surrogate-assisted NSGA-III HV from existing multiseed_summary.json
    multi = json.loads((RESULTS_DIR / "multiseed_summary.json").read_text())
    for case_key in summary:
        if case_key in multi:
            summary[case_key]["surrogate_assisted_nsga3_5seed"] = {
                "hv_mean":    float(multi[case_key]["hypervolume"]["mean"]),
                "hv_ci95_low": float(multi[case_key]["hypervolume"]["ci95_low"]),
                "hv_ci95_high": float(multi[case_key]["hypervolume"]["ci95_high"]),
                "n_full_evals_per_run": 338,  # mean from optimization_summary
            }

    out = RESULTS_DIR / "algorithmic_baseline.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")

    # Final printable summary table
    print(f"\n{'='*72}\nSummary: HV at terminal budget (10000 evals) vs surrogate-assisted (~338 evals)\n{'='*72}")
    for case_key, d in summary.items():
        if "surrogate_assisted_nsga3_5seed" in d:
            sa = d["surrogate_assisted_nsga3_5seed"]
            n2 = d["nsga2"][-1]
            n3 = d["nsga3_no_surrogate"][-1]
            print(f"\n  {case_key}:")
            print(f"    NSGA-II vanilla, 10000 evals:           HV = {n2['hv']:>10.0f}  ({n2['wall_s']:.1f}s)")
            print(f"    NSGA-III no surrogate, 10000 evals:     HV = {n3['hv']:>10.0f}  ({n3['wall_s']:.1f}s)")
            print(f"    Surrogate-assisted NSGA-III (~338 evals): HV = {sa['hv_mean']:>10.0f}  "
                  f"[{sa['hv_ci95_low']:.0f}, {sa['hv_ci95_high']:.0f}]")


if __name__ == "__main__":
    main()
