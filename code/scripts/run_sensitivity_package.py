"""Robustness / sensitivity package.

Tests the robustness of the reported results in three parts, each targeting a different way the
headline numbers could be an artefact rather than a result.

S1 - Seed robustness of the hypervolume gain.
    Both the surrogate-assisted pipeline and the baselines are run across the
    same seed set and the hypervolume gain is reported with a bootstrap 95%
    CI, so the comparison is like-for-like.

S2 - Hypervolume reference-point sensitivity.
    Hypervolume is notoriously dependent on the reference point, which is a
    free parameter. The gain is recomputed over a grid of reference points
    derived from the observed nadir, plus the fixed point used in the paper,
    to show the conclusion is not an artefact of that choice.

S3 - Total-order Sobol indices.
    S1 alone attributes nothing to
    interactions, so a variable can look inert while mattering through
    couplings. ST is computed here for all four objectives in all cases;
    ST - S1 is the interaction share.

Outputs:
    data/results/sensitivity_package.json
    data/results/sensitivity_seed_robustness.csv
    data/results/sensitivity_refpoint.csv
    data/results/sensitivity_sobol_st.csv

Usage:
    .venv/bin/python scripts/run_sensitivity_package.py [--seeds 42 7 101 2025 314]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample as sobol_sample

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER, VARIABLE_NAMES
from mobo_envelope.objectives import N_OBJ, evaluate_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

# Reference point used in the manuscript (scripts/algorithmic_baseline.py)
HV_REF_PAPER = np.array([1.0, 1.5, -10.0, 1500.0], dtype=float)
TERMINAL_BUDGET = 10_000
OBJ_LABELS = ["thermal", "structural", "daylight", "cost"]

CASE_FRONT_CSV = {
    "A": "case_a_houston_pareto.csv",
    "B": "case_b_nyc_pareto.csv",
    "C": "case_c_minneapolis_pareto.csv",
}


class _AnalyticalProblem(Problem):
    def __init__(self, city: str):
        super().__init__(n_var=N_VARS, n_obj=N_OBJ, n_ieq_constr=1, xl=LOWER, xu=UPPER)
        self.city = city

    def _evaluate(self, x, out, *args, **kwargs):
        f = evaluate_batch(np.asarray(x), self.city)
        out["F"] = f
        out["G"] = (f[:, 1] - 1.0).reshape(-1, 1)


def _non_dominated(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        diff = F - F[i]
        dom = np.all(diff <= 1e-12, axis=1) & np.any(diff < -1e-12, axis=1)
        dom[i] = False
        if dom.any():
            nd[i] = False
    return nd


def _run_baseline(kind: str, city: str, n_evals: int, seed: int) -> np.ndarray:
    """Return the non-dominated objective matrix for a baseline optimiser."""
    problem = _AnalyticalProblem(city)
    pop = 100
    n_gen = max(1, n_evals // pop)
    if kind == "nsga2":
        algo = NSGA2(pop_size=pop)
    else:
        ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=8)
        algo = NSGA3(pop_size=pop, ref_dirs=ref_dirs)
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    if res.F is None or len(res.F) == 0:
        return np.empty((0, N_OBJ))
    F = np.atleast_2d(res.F)
    # algorithmic_baseline.py scores the FEASIBLE subset of the final front;
    # reproduce that exactly so these numbers stay comparable to the published ones.
    feas = res.G[:, 0] <= 0 if res.G is not None else np.ones(len(F), dtype=bool)
    return F[feas]


def _surrogate_front(case_id: str) -> np.ndarray:
    df = pd.read_csv(RESULTS_DIR / CASE_FRONT_CSV[case_id])
    return np.column_stack([
        df["u_eff_w_m2k"].to_numpy(float),
        (1.0 - df["structural_reserve"].to_numpy(float)),   # utilisation
        -df["sda_pct"].to_numpy(float),
        df["cost_usd_m2"].to_numpy(float),
    ])


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def s1_seed_robustness(seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    """Baselines re-run across the same seeds as the surrogate."""
    multi = json.loads((RESULTS_DIR / "multiseed_summary.json").read_text())
    key = {"A": "A_Houston", "B": "B_NYC", "C": "C_Minneapolis"}
    hv = HV(ref_point=HV_REF_PAPER)

    rows = []
    for cid, case in CASES.items():
        sur = {d["seed"]: d["hypervolume"] for d in multi[key[cid]]["per_seed"]}
        for seed in seeds:
            for kind in ("nsga2", "nsga3_no_surrogate"):
                F = _run_baseline(kind, case.city, TERMINAL_BUDGET, seed)
                rows.append({
                    "case": key[cid], "seed": seed, "method": kind,
                    "hv": float(hv(F)) if len(F) else float("nan"),
                    "n_evals": TERMINAL_BUDGET,
                })
            rows.append({"case": key[cid], "seed": seed, "method": "surrogate_nsga3",
                         "hv": sur.get(seed, float("nan")),
                         "n_evals": multi[key[cid]]["per_seed"][0]["n_full_eval"]})
        print(f"  {key[cid]}: {len(seeds)} seeds x 2 baselines done", flush=True)

    df = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    for case_key, g in df.groupby("case"):
        piv = g.pivot_table(index="seed", columns="method", values="hv")
        piv = piv.dropna()
        entry: dict = {"n_seeds": int(len(piv))}
        for base in ("nsga2", "nsga3_no_surrogate"):
            gains = (piv["surrogate_nsga3"] / piv[base] - 1.0).to_numpy() * 100.0
            lo, hi = _bootstrap_ci(gains)
            entry[f"gain_vs_{base}_pct"] = {
                "mean": float(gains.mean()), "min": float(gains.min()),
                "max": float(gains.max()), "ci95_low": lo, "ci95_high": hi,
            }
        summary[case_key] = entry
    return df, summary


def s2_refpoint_sensitivity() -> tuple[pd.DataFrame, dict]:
    """Recompute the gain over a grid of reference points."""
    key = {"A": "A_Houston", "B": "B_NYC", "C": "C_Minneapolis"}
    rows = []
    for cid, case in CASES.items():
        F_sur = _surrogate_front(cid)
        fronts = {"surrogate_nsga3": F_sur}
        for kind in ("nsga2", "nsga3_no_surrogate"):
            fronts[kind] = _run_baseline(kind, case.city, TERMINAL_BUDGET, seed=42)

        stacked = np.vstack([f for f in fronts.values() if len(f)])
        nadir, ideal = stacked.max(axis=0), stacked.min(axis=0)
        span = np.where((nadir - ideal) == 0, 1.0, nadir - ideal)

        ref_points = {f"nadir+{int(k*100)}%": nadir + k * span
                      for k in (0.05, 0.10, 0.20, 0.30, 0.50)}
        ref_points["paper_fixed"] = HV_REF_PAPER

        for label, ref in ref_points.items():
            # Points outside the reference box contribute zero volume; that is
            # standard and must NOT exclude the reference point from the sweep.
            n_outside = int(np.any(stacked >= ref, axis=1).sum())
            ind = HV(ref_point=ref)
            hvs = {k: (float(ind(f)) if len(f) else float("nan")) for k, f in fronts.items()}
            rows.append({
                "case": key[cid], "ref_point": label,
                "hv_surrogate": hvs["surrogate_nsga3"],
                "hv_nsga2": hvs["nsga2"], "hv_nsga3": hvs["nsga3_no_surrogate"],
                "gain_vs_nsga2_pct": (hvs["surrogate_nsga3"] / hvs["nsga2"] - 1) * 100,
                "gain_vs_nsga3_pct": (hvs["surrogate_nsga3"] / hvs["nsga3_no_surrogate"] - 1) * 100,
                "n_points_outside_ref": n_outside,
                "note": "",
            })
        print(f"  {key[cid]}: {len(ref_points)} reference points done", flush=True)

    df = pd.DataFrame(rows)
    ok = df[df.note == ""]
    summary = {
        "gain_vs_nsga2_pct": {"min": float(ok.gain_vs_nsga2_pct.min()),
                              "max": float(ok.gain_vs_nsga2_pct.max())},
        "gain_vs_nsga3_pct": {"min": float(ok.gain_vs_nsga3_pct.min()),
                              "max": float(ok.gain_vs_nsga3_pct.max())},
        "n_reference_points": int(ok.ref_point.nunique()),
        "conclusion_sign_stable": bool((ok.gain_vs_nsga2_pct > 0).all()
                                       and (ok.gain_vs_nsga3_pct > 0).all()),
    }
    return df, summary


def s3_sobol_total_order(n_base: int = 512) -> tuple[pd.DataFrame, dict]:
    """First- and total-order Sobol indices for all four Path-A objectives."""
    problem = {"num_vars": N_VARS, "names": list(VARIABLE_NAMES),
               "bounds": [[lo, hi] for lo, hi in zip(LOWER, UPPER)]}
    X = sobol_sample(problem, N=n_base, seed=42)
    rows = []
    for cid, case in CASES.items():
        key = {"A": "A_Houston", "B": "B_NYC", "C": "C_Minneapolis"}[cid]
        Y = evaluate_batch(X, case.city)
        # sensitivity.py flips the daylight axis so a large index reads as
        # "matters to sDA"; variance-based indices are invariant to the sign.
        Y = Y.copy()
        Y[:, 2] = -Y[:, 2]
        for j, obj in enumerate(OBJ_LABELS):
            Si = analyze(problem, Y[:, j], print_to_console=False, seed=42)
            for name, s1, st in zip(VARIABLE_NAMES, Si["S1"], Si["ST"]):
                rows.append({
                    "objective": obj, "case": key, "variable": name,
                    "sobol_s1": round(float(s1), 4),
                    "sobol_st": round(float(st), 4),
                    "interaction_share": round(float(st - s1), 4),
                })
        print(f"  {key}: 4 objectives x {N_VARS} variables done", flush=True)

    df = pd.DataFrame(rows)
    top = (df.groupby(["objective", "variable"]).interaction_share.mean()
             .reset_index().sort_values("interaction_share", ascending=False).head(8))
    summary = {
        "n_base_samples": n_base,
        "n_model_evals_per_case": int(X.shape[0]),
        "max_interaction_share": float(df.interaction_share.max()),
        "largest_interactions": top.to_dict(orient="records"),
    }
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 101, 2025, 314])
    ap.add_argument("--sobol-n", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()
    print("S1: seed robustness (baselines re-run on the surrogate's seed set)")
    df_s1, sum_s1 = s1_seed_robustness(args.seeds)
    df_s1.to_csv(RESULTS_DIR / "sensitivity_seed_robustness.csv", index=False)

    print("\nS2: hypervolume reference-point sensitivity")
    df_s2, sum_s2 = s2_refpoint_sensitivity()
    df_s2.to_csv(RESULTS_DIR / "sensitivity_refpoint.csv", index=False)

    print("\nS3: total-order Sobol indices")
    df_s3, sum_s3 = s3_sobol_total_order(args.sobol_n)
    df_s3.to_csv(RESULTS_DIR / "sensitivity_sobol_st.csv", index=False)

    out = {"seeds": args.seeds, "wall_clock_s": time.time() - t0,
           "S1_seed_robustness": sum_s1, "S2_refpoint": sum_s2, "S3_sobol": sum_s3}
    with open(RESULTS_DIR / "sensitivity_package.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*70}\nS1 - hypervolume gain, matched seeds (mean [95% CI])\n{'='*70}")
    for case_key, s in sum_s1.items():
        a = s["gain_vs_nsga2_pct"]; b = s["gain_vs_nsga3_no_surrogate_pct"]
        print(f"{case_key:16s} vs NSGA-II  {a['mean']:6.1f}%  [{a['ci95_low']:5.1f}, {a['ci95_high']:5.1f}]")
        print(f"{'':16s} vs NSGA-III {b['mean']:6.1f}%  [{b['ci95_low']:5.1f}, {b['ci95_high']:5.1f}]")
    print(f"\nS2 - gain across {sum_s2['n_reference_points']} reference points: "
          f"vs NSGA-II {sum_s2['gain_vs_nsga2_pct']['min']:.1f}-{sum_s2['gain_vs_nsga2_pct']['max']:.1f}%, "
          f"vs NSGA-III {sum_s2['gain_vs_nsga3_pct']['min']:.1f}-{sum_s2['gain_vs_nsga3_pct']['max']:.1f}%")
    print(f"     sign stable across all reference points: {sum_s2['conclusion_sign_stable']}")
    print(f"\nS3 - max interaction share (ST - S1): {sum_s3['max_interaction_share']:.3f}")
    print(f"\nwall clock: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
