"""NSGA-III with surrogate-assisted infill on the analytical Path-A objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import qmc

from mobo_envelope.envelope import LOWER, N_VARS, UPPER
from mobo_envelope.objectives import N_OBJ, evaluate_batch
from mobo_envelope.surrogate import SurrogateBank


@dataclass
class OptimizeConfig:
    pop_size: int = 200
    n_generations: int = 300
    n_initial_lhs: int = 200
    surrogate_retrain_every: int = 5
    seed: int = 42
    n_holdout: int = 60


@dataclass
class OptimizeResult:
    x: np.ndarray              # (n, n_vars) all evaluated designs (analytical)
    f: np.ndarray              # (n, n_obj) analytical objective values
    pareto_x: np.ndarray       # (k, n_vars) non-dominated subset
    pareto_f: np.ndarray       # (k, n_obj)
    surrogate_metrics: dict[str, list[float]]
    n_full_eval: int
    n_total_pop_evals: int
    seed: int
    hv_history: list[float] | None = None  # hypervolume per generation (post-step 4)


class _AnalyticalProblem(Problem):
    """pymoo wrapper around the analytical objectives.

    Constraint: utilization (objective 1) <= 1.0 (no yielding, ASCE 7-22 ULS).
    """

    def __init__(self, city: str):
        super().__init__(n_var=N_VARS, n_obj=N_OBJ, n_ieq_constr=1, xl=LOWER, xu=UPPER)
        self.city = city

    def _evaluate(self, x, out, *args, **kwargs):
        f = evaluate_batch(np.asarray(x), self.city)
        out["F"] = f
        # G_i <= 0 means feasible. utilization - 1.0 <= 0 -> not yielded.
        out["G"] = (f[:, 1] - 1.0).reshape(-1, 1)


def _lhs(n: int, n_vars: int, lower: np.ndarray, upper: np.ndarray, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=n_vars, seed=seed)
    sample = sampler.random(n)
    return lower + sample * (upper - lower)


def _non_dominated(f: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-non-dominated rows. Naive O(n^2)."""
    n = f.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        # Dominated if any j has all components <= f[i] and at least one strict.
        diff = f - f[i]
        worse_or_equal = np.all(diff <= 1e-12, axis=1)
        strictly_better = np.any(diff < -1e-12, axis=1)
        dominators = worse_or_equal & strictly_better
        dominators[i] = False
        if dominators.any():
            nd[i] = False
    return nd


def run_optimization(city: str, cfg: OptimizeConfig) -> OptimizeResult:
    """Hybrid surrogate-assisted NSGA-III on the analytical objectives.

    Strategy:
      1) Latin-Hypercube sample n_initial_lhs analytical points (full eval).
      2) Fit one GP per objective on those points.
      3) Run NSGA-III against a surrogate-only problem for n_generations.
      4) On convergence, re-evaluate the final population with the analytical
         model, append to the dataset, and refit. Repeat once for stability.
      5) Return full evaluated archive plus the non-dominated set.
    """
    rng = np.random.default_rng(cfg.seed)
    problem = _AnalyticalProblem(city)

    # Step 1: LHS initial set
    x_init = _lhs(cfg.n_initial_lhs, N_VARS, LOWER, UPPER, cfg.seed)
    f_init = evaluate_batch(x_init, city)

    # Step 2: hold out a small set for surrogate metrics
    x_hold = _lhs(cfg.n_holdout, N_VARS, LOWER, UPPER, cfg.seed + 1)
    f_hold = evaluate_batch(x_hold, city)

    bank = SurrogateBank(n_obj=N_OBJ, rng=rng)
    bank.fit(x_init, f_init)

    # Step 3: surrogate-driven NSGA-III with surrogate-predicted constraint
    class _SurrogateProblem(Problem):
        def __init__(self, surrogate: SurrogateBank):
            super().__init__(n_var=N_VARS, n_obj=N_OBJ, n_ieq_constr=1, xl=LOWER, xu=UPPER)
            self.surrogate = surrogate

        def _evaluate(self, x, out, *args, **kwargs):
            f_pred = self.surrogate.predict(np.asarray(x))
            out["F"] = f_pred
            out["G"] = (f_pred[:, 1] - 1.0).reshape(-1, 1)

    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=8)
    algo = NSGA3(pop_size=cfg.pop_size, ref_dirs=ref_dirs)
    surrogate_problem = _SurrogateProblem(bank)
    res1 = minimize(
        surrogate_problem,
        algo,
        ("n_gen", cfg.n_generations),
        seed=cfg.seed,
        verbose=False,
    )

    # Step 4: re-eval final pop analytically, refit, run a refinement burst
    x_final = res1.X
    f_true_final = evaluate_batch(x_final, city)
    x_train2 = np.vstack([x_init, x_final])
    f_train2 = np.vstack([f_init, f_true_final])
    bank.fit(x_train2, f_train2)

    ref_dirs2 = get_reference_directions("das-dennis", N_OBJ, n_partitions=8)
    algo2 = NSGA3(pop_size=cfg.pop_size, ref_dirs=ref_dirs2)
    surrogate_problem2 = _SurrogateProblem(bank)
    res2 = minimize(
        surrogate_problem2,
        algo2,
        ("n_gen", max(50, cfg.n_generations // 3)),
        seed=cfg.seed + 100,
        verbose=False,
    )
    x_final2 = res2.X
    f_true_final2 = evaluate_batch(x_final2, city)

    # Step 5: pool everything, drop analytically-infeasible designs, then
    # extract Pareto. Constraint: utilization (objective index 1) <= 1.0.
    x_all_raw = np.vstack([x_train2, x_final2])
    f_all_raw = np.vstack([f_train2, f_true_final2])
    feasible = f_all_raw[:, 1] <= 1.0
    x_all = x_all_raw[feasible]
    f_all = f_all_raw[feasible]
    nd_mask = _non_dominated(f_all)

    surrogate_metrics = bank.metrics(x_hold, f_hold)
    n_full_eval = x_all.shape[0]
    # equivalent surrogate-free pop evaluations (NSGA-III without surrogate)
    n_total_pop_evals = cfg.pop_size * cfg.n_generations

    return OptimizeResult(
        x=x_all,
        f=f_all,
        pareto_x=x_all[nd_mask],
        pareto_f=f_all[nd_mask],
        surrogate_metrics=surrogate_metrics,
        n_full_eval=n_full_eval,
        n_total_pop_evals=n_total_pop_evals,
        seed=cfg.seed,
    )
