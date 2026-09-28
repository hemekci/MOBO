"""Verify surrogate accuracy is genuine (no data leakage, no overfitting).

Sanity checks:
  1) 5-fold cross-validation R^2 on the LHS training set
  2) Independent holdout R^2 (different seed than training)
  3) Brand-new uniform random sample (different distribution than LHS)
  4) Inspect learned ARD length scales: long length scales = "ignored" dims
  5) Compare ARD vs isotropic Matern head-to-head on the same data
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mobo_envelope.envelope import LOWER, N_VARS, UPPER, VARIABLE_NAMES
from mobo_envelope.objectives import N_OBJ, OBJECTIVE_NAMES, evaluate_batch
from mobo_envelope.surrogate import SurrogateBank


def lhs(n: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc
    return LOWER + qmc.LatinHypercube(d=N_VARS, seed=seed).random(n) * (UPPER - LOWER)


def uniform(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return LOWER + rng.random((n, N_VARS)) * (UPPER - LOWER)


def kfold_r2(x: np.ndarray, y: np.ndarray, kernel_kind: str) -> list[float]:
    """5-fold CV R^2 per objective."""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    results: list[list[float]] = [[] for _ in range(N_OBJ)]
    for fold_train, fold_eval in kf.split(x):
        bank = SurrogateBank(n_obj=N_OBJ, rng=np.random.default_rng(0))
        # Override kernel for ablation
        original = bank._build_kernel
        if kernel_kind == "isotropic":
            bank._build_kernel = lambda nf: (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
            )
        bank.fit(x[fold_train], y[fold_train])
        m = bank.metrics(x[fold_eval], y[fold_eval])
        for j in range(N_OBJ):
            results[j].append(m["r2"][j])
        bank._build_kernel = original
    return [float(np.mean(r)) for r in results]


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    city = "NYC"

    # ---------------------------------------------------------------------
    # Generate three INDEPENDENT sets — different sampling strategies, different seeds
    # ---------------------------------------------------------------------
    x_train = lhs(200, seed=42)
    y_train = evaluate_batch(x_train, city)

    x_holdout_lhs = lhs(60, seed=999)            # different LHS seed
    y_holdout_lhs = evaluate_batch(x_holdout_lhs, city)

    x_holdout_unif = uniform(120, seed=12345)    # different distribution entirely
    y_holdout_unif = evaluate_batch(x_holdout_unif, city)

    # Confirm no overlap
    overlap = (np.isclose(x_train[:, None, :], x_holdout_lhs[None, :, :]).all(axis=2)).any()
    assert not overlap, "x_train and x_holdout_lhs overlap"

    print("=== Sanity 1: 5-fold CV R^2 on training set (no holdout reuse) ===")
    cv_ard = kfold_r2(x_train, y_train, "ard")
    cv_iso = kfold_r2(x_train, y_train, "isotropic")
    print(f"  ARD (per-dim length scales):  {dict(zip(OBJECTIVE_NAMES, cv_ard))}")
    print(f"  Iso (single length scale):    {dict(zip(OBJECTIVE_NAMES, cv_iso))}")

    print("\n=== Sanity 2: Independent LHS holdout (seed 999) ===")
    bank = SurrogateBank(n_obj=N_OBJ, rng=np.random.default_rng(0))
    bank.fit(x_train, y_train)
    m = bank.metrics(x_holdout_lhs, y_holdout_lhs)
    for j, name in enumerate(OBJECTIVE_NAMES):
        print(f"  {name:14s}  R^2={m['r2'][j]:.3f}  RMSE={m['rmse'][j]:.3f}  MAPE={m['mape_pct'][j]:.2f}%")

    print("\n=== Sanity 3: Uniform random (different distribution from LHS) ===")
    m = bank.metrics(x_holdout_unif, y_holdout_unif)
    for j, name in enumerate(OBJECTIVE_NAMES):
        print(f"  {name:14s}  R^2={m['r2'][j]:.3f}  RMSE={m['rmse'][j]:.3f}  MAPE={m['mape_pct'][j]:.2f}%")

    print("\n=== Sanity 4: ARD length scales (large = irrelevant dim) ===")
    assert bank.models is not None
    for j, name in enumerate(OBJECTIVE_NAMES):
        gp = bank.models[j]
        kernel = gp.kernel_
        # Walk the kernel structure: ConstantKernel * Matern + WhiteKernel
        # In sklearn, kernel_.k1 = ConstantKernel * Matern, k2 = WhiteKernel
        ls = kernel.k1.k2.length_scale
        print(f"  {name:14s}  length scales:")
        for k, var in enumerate(VARIABLE_NAMES):
            mark = "    "
            if ls[k] > 50.0:
                mark = "[--]"  # effectively ignored
            elif ls[k] < 1.0:
                mark = "[**]"  # highly active
            print(f"    {mark}  {var:18s}  ls={ls[k]:.3f}")

    print("\n=== Sanity 5: training set R^2 (should be ~1.0 since deterministic) ===")
    m = bank.metrics(x_train, y_train)
    print(f"  on training:  R^2={[round(v, 4) for v in m['r2']]}")

    print("\n=== Closed-form ground truth: structural depends only on dims 6, 7 ===")
    # Vary only member_depth and framing while holding other dims at midpoint
    mid = (LOWER + UPPER) / 2
    test_xs = []
    for d_val in np.linspace(LOWER[6], UPPER[6], 8):
        for f_idx in [0.5, 1.5, 2.5, 3.5]:  # mid-bin per framing
            x = mid.copy()
            x[6] = d_val
            x[7] = f_idx
            test_xs.append(x)
    test_xs = np.array(test_xs)
    y_true = evaluate_batch(test_xs, city)
    y_pred = bank.predict(test_xs)
    print(f"  structural utilization:")
    print(f"    true range: [{y_true[:, 1].min():.3f}, {y_true[:, 1].max():.3f}]")
    print(f"    pred range: [{y_pred[:, 1].min():.3f}, {y_pred[:, 1].max():.3f}]")
    err = np.abs(y_true[:, 1] - y_pred[:, 1])
    print(f"    abs err:    mean={err.mean():.4f}, max={err.max():.4f}")


if __name__ == "__main__":
    main()
