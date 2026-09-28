"""Direct NSGA-III optimisation on the Path-C high-fidelity objectives.

Verifying designs drawn from the Path-A front can only confirm that proposed
designs survive; it cannot reveal high-fidelity optima the framework never
proposed. This script therefore builds an independent high-fidelity reference. It runs NSGA-III directly
against the Path-C objective vector, with the same reference directions and
design bounds as the Path-A search, so the comparison isolates *fidelity* rather
than *algorithm*:

    f1 = total_eui       (EnergyPlus 26.1, kWh/m^2/yr)
    f2 = utilization     (openseespy FEM, dimensionless)
    f3 = -sda_pct        (Radiance, %)
    f4 = cost            (USD/m^2, analytical - identical in both paths)

Constraint: utilization - 1.0 <= 0 (ASCE 7-22 ULS), matching Path-A.

Evaluations are independent, so the population is evaluated across a process
pool. Both simulator runners allocate per-call temporary directories, which
makes them safe to run concurrently.

Outputs (per case):
    data/results/path_c_direct_opt_{case}_archive.csv   every evaluated design
    data/results/path_c_direct_opt_{case}_front.csv     non-dominated subset
    data/results/path_c_direct_opt_{case}_meta.json     budget, timing, failures

The archive is checkpointed after every generation, so an interrupted run keeps
all completed evaluations.

Usage:
    source scripts/sim_env.sh
    .venv/bin/python scripts/run_path_c_direct_opt.py --case A --pop 165 --gen 24 --workers 5
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from mobo_envelope import cost as cost_mod
from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER, decode_design
from mobo_envelope.path_b.structural_opensees import utilization_fem
from mobo_envelope.path_c.energyplus_runner_v2 import run_eplus_v2
from mobo_envelope.path_c.radiance_runner_v2 import run_radiance_v2

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

# Objective vector order used for all Path-C front comparisons
OBJ_NAMES = ["total_eui", "utilization", "neg_sda", "cost"]
N_OBJ = 4

# Assigned when a simulator fails, so NSGA-III drives away from the design.
# Failures are recorded and excluded from the reported front.
PENALTY = np.array([1.0e6, 1.0e6, 0.0, 1.0e6], dtype=float)


def evaluate_one(args: tuple[np.ndarray, str]) -> dict:
    """Evaluate a single design at Path-C fidelity. Runs in a worker process."""
    x, city = args
    d = decode_design(np.asarray(x, dtype=float))
    t0 = time.time()

    ep = run_eplus_v2(d, city)
    rad = run_radiance_v2(d, city, n_amb=1)
    try:
        util = float(utilization_fem(d, city))
        fem_ok = True
        fem_err = ""
    except Exception as exc:  # noqa: BLE001 - record and penalise, never abort the run
        util = float("nan")
        fem_ok = False
        fem_err = str(exc)[:200]

    cost = float(cost_mod.cost_per_envelope_m2(d))
    ok = bool(ep.success and rad.success and fem_ok)

    rec = {
        "total_eui_kwh_m2_yr": float(ep.total_eui_kwh_m2_yr) if ep.success else None,
        "heating_eui_kwh_m2_yr": float(ep.heating_eui_kwh_m2_yr) if ep.success else None,
        "cooling_eui_kwh_m2_yr": float(ep.cooling_eui_kwh_m2_yr) if ep.success else None,
        "sda_pct": float(rad.sda_pct) if rad.success else None,
        "utilization": util if fem_ok else None,
        "cost_usd_m2": cost,
        "eplus_success": bool(ep.success),
        "radiance_success": bool(rad.success),
        "fem_success": fem_ok,
        "success": ok,
        "eplus_error": ep.error[:200] if not ep.success else "",
        "radiance_error": rad.error[:200] if not rad.success else "",
        "fem_error": fem_err,
        "eval_time_s": time.time() - t0,
    }
    for i, name in enumerate(np.asarray(x, dtype=float)):
        rec[f"x{i}"] = float(name)
    return rec


def _objective_row(rec: dict) -> np.ndarray:
    if not rec["success"]:
        return PENALTY.copy()
    return np.array([
        rec["total_eui_kwh_m2_yr"],
        rec["utilization"],
        -rec["sda_pct"],
        rec["cost_usd_m2"],
    ], dtype=float)


class PathCProblem(Problem):
    """NSGA-III problem evaluating EnergyPlus + Radiance + FEM per design."""

    def __init__(self, city: str, workers: int, archive: list[dict], state: dict):
        super().__init__(n_var=N_VARS, n_obj=N_OBJ, n_ieq_constr=1, xl=LOWER, xu=UPPER)
        self.city = city
        self.workers = workers
        self.archive = archive
        self.state = state
        self.cache: dict[tuple, dict] = {}

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=float)
        keys = [tuple(np.round(row, 9)) for row in X]

        todo_idx = [i for i, k in enumerate(keys) if k not in self.cache]
        t0 = time.time()
        if todo_idx:
            payload = [(X[i], self.city) for i in todo_idx]
            if self.workers > 1:
                with ProcessPoolExecutor(max_workers=self.workers) as pool:
                    results = list(pool.map(evaluate_one, payload))
            else:
                results = [evaluate_one(p) for p in payload]
            for i, rec in zip(todo_idx, results):
                self.cache[keys[i]] = rec
                self.archive.append({**rec, "generation": self.state["gen"]})

        recs = [self.cache[k] for k in keys]
        F = np.vstack([_objective_row(r) for r in recs])
        # utilization <= 1.0 (feasible when G <= 0); penalised rows stay infeasible
        G = (F[:, 1] - 1.0).reshape(-1, 1)
        out["F"] = F
        out["G"] = G

        self.state["gen"] += 1
        n_new = len(todo_idx)
        n_fail = sum(1 for r in recs if not r["success"])
        elapsed = time.time() - t0
        rate = elapsed / n_new if n_new else 0.0
        print(
            f"  gen {self.state['gen']:>3}  new={n_new:>4}  cached={len(keys) - n_new:>4}  "
            f"fail={n_fail:>3}  {elapsed:6.1f}s  ({rate:5.2f}s/eval)  "
            f"total_evals={len(self.archive)}",
            flush=True,
        )
        _checkpoint(self.archive, self.state["case_key"])


def _checkpoint(archive: list[dict], case_key: str) -> None:
    if archive:
        pd.DataFrame(archive).to_csv(
            RESULTS_DIR / f"path_c_direct_opt_{case_key}_archive.csv", index=False
        )


def _non_dominated_mask(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        diff = F - F[i]
        worse_or_equal = np.all(diff <= 1e-12, axis=1)
        strictly_better = np.any(diff < -1e-12, axis=1)
        dominators = worse_or_equal & strictly_better
        dominators[i] = False
        if dominators.any():
            nd[i] = False
    return nd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=sorted(CASES.keys()))
    ap.add_argument("--pop", type=int, default=165)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-partitions", type=int, default=8)
    args = ap.parse_args()

    case = CASES[args.case]
    case_key = case.name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=args.n_partitions)
    print(f"=== Direct Path-C NSGA-III: {case_key} ({case.city}) ===")
    print(f"ref_dirs={len(ref_dirs)}  pop={args.pop}  gen={args.gen}  "
          f"workers={args.workers}  seed={args.seed}")
    print(f"budget <= {args.pop * args.gen} Path-C evaluations", flush=True)

    archive: list[dict] = []
    state = {"gen": 0, "case_key": case_key}
    problem = PathCProblem(case.city, args.workers, archive, state)
    algo = NSGA3(pop_size=args.pop, ref_dirs=ref_dirs)

    t_start = time.time()
    res = minimize(
        problem,
        algo,
        termination=("n_gen", args.gen),
        seed=args.seed,
        verbose=False,
    )
    wall = time.time() - t_start

    df = pd.DataFrame(archive)
    _checkpoint(archive, case_key)

    ok = df[df["success"]].reset_index(drop=True)
    F_ok = np.column_stack([
        ok["total_eui_kwh_m2_yr"].to_numpy(float),
        ok["utilization"].to_numpy(float),
        -ok["sda_pct"].to_numpy(float),
        ok["cost_usd_m2"].to_numpy(float),
    ]) if len(ok) else np.empty((0, N_OBJ))

    # Report the front over structurally feasible, successfully simulated designs
    feasible = ok["utilization"].to_numpy(float) <= 1.0 if len(ok) else np.zeros(0, bool)
    idx_feas = np.where(feasible)[0]
    if len(idx_feas):
        nd_local = _non_dominated_mask(F_ok[idx_feas])
        front_idx = idx_feas[nd_local]
    else:
        front_idx = np.array([], dtype=int)

    front = ok.iloc[front_idx].reset_index(drop=True)
    front.to_csv(RESULTS_DIR / f"path_c_direct_opt_{case_key}_front.csv", index=False)

    meta = {
        "case": case_key,
        "city": case.city,
        "algorithm": "NSGA-III (direct, Path-C objectives)",
        "ref_dirs": int(len(ref_dirs)),
        "n_partitions": args.n_partitions,
        "pop_size": args.pop,
        "n_generations": args.gen,
        "seed": args.seed,
        "workers": args.workers,
        "n_evaluations_total": int(len(df)),
        "n_evaluations_successful": int(len(ok)),
        "n_evaluations_failed": int((~df["success"]).sum()) if len(df) else 0,
        "n_eplus_fail": int((~df["eplus_success"]).sum()) if len(df) else 0,
        "n_radiance_fail": int((~df["radiance_success"]).sum()) if len(df) else 0,
        "n_feasible": int(len(idx_feas)),
        "n_front": int(len(front)),
        "wall_clock_s": wall,
        "mean_eval_s": float(df["eval_time_s"].mean()) if len(df) else None,
        "objective_order": OBJ_NAMES,
        "model_notes": {
            "energyplus_idf": "RefBldgSmallOfficeNew2004_Chicago.idf",
            "radiance_geometry": "8x8x3 m perimeter zone with 4-orientation glazing",
        },
    }
    with open(RESULTS_DIR / f"path_c_direct_opt_{case_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"evaluations:      {meta['n_evaluations_total']} "
          f"({meta['n_evaluations_failed']} failed)")
    print(f"feasible designs: {meta['n_feasible']}")
    print(f"front size:       {meta['n_front']}")
    print(f"wall clock:       {wall / 60:.1f} min")
    print(f"written: path_c_direct_opt_{case_key}_{{archive,front,meta}}")


if __name__ == "__main__":
    main()
