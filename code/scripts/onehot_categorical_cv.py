"""Compare ordinal vs one-hot encoding of categorical variables in the GP.

Re-fits the Gaussian-process surrogate with a one-hot encoded categorical
representation (4 indicator variables instead of 2 ordinal indices) and
reports 5-fold cross-validation R^2 against the existing ordinal baseline,
per case and per objective.

Output: data/results/onehot_cv_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER
from mobo_envelope.objectives import evaluate_batch
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

# Indices of categorical columns in the design vector
GLAZING_IDX = 5     # 0=double, 1=triple, 2=vacuum
FRAMING_IDX = 7     # 0=steel, 1=aluminum, 2=timber, 3=frp
N_GLAZING_CATS = 3
N_FRAMING_CATS = 4
OBJ_NAMES = ["U_eff", "log1p_util", "neg_sda", "cost"]


def _ordinal_features(X: np.ndarray) -> np.ndarray:
    return X.copy()


def _onehot_features(X: np.ndarray) -> np.ndarray:
    """Replace ordinal indices with one-hot indicators."""
    n = X.shape[0]
    glaz_idx = np.clip(np.round(X[:, GLAZING_IDX]).astype(int), 0, N_GLAZING_CATS - 1)
    fram_idx = np.clip(np.round(X[:, FRAMING_IDX]).astype(int), 0, N_FRAMING_CATS - 1)
    glaz_oh = np.zeros((n, N_GLAZING_CATS))
    fram_oh = np.zeros((n, N_FRAMING_CATS))
    glaz_oh[np.arange(n), glaz_idx] = 1.0
    fram_oh[np.arange(n), fram_idx] = 1.0
    other_idx = [i for i in range(N_VARS) if i not in (GLAZING_IDX, FRAMING_IDX)]
    return np.hstack([X[:, other_idx], glaz_oh, fram_oh])


def _build_kernel(n_features: int):
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
    )


def _cv_r2_per_objective(X: np.ndarray, Y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict:
    """Return per-objective 5-fold CV R^2 for the given feature matrix."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_obj = Y.shape[1]
    r2 = np.zeros(n_obj)
    for j in range(n_obj):
        ss_res, ss_tot = 0.0, 0.0
        y_mean = float(Y[:, j].mean())
        for tr, te in kf.split(X):
            kernel = _build_kernel(X.shape[1])
            gpr = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, alpha=1e-8, n_restarts_optimizer=2
            )
            gpr.fit(X[tr], Y[tr, j])
            y_pred = gpr.predict(X[te])
            ss_res += float(np.sum((Y[te, j] - y_pred) ** 2))
            ss_tot += float(np.sum((Y[te, j] - y_mean) ** 2))
        r2[j] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {OBJ_NAMES[j]: float(r2[j]) for j in range(n_obj)}


def main() -> None:
    summary: dict[str, dict] = {}
    rng = np.random.default_rng(42)
    sampler = qmc.LatinHypercube(d=N_VARS, seed=42)

    for case_key, case in CASES.items():
        # Generate the same 200-sample LHS used in main pipeline (seed=42)
        u = sampler.random(n=200)
        X = qmc.scale(u, LOWER, UPPER)
        Y_raw = evaluate_batch(X, case.city)
        # Match the main pipeline's log1p transform on f_2 (utilisation)
        Y = Y_raw.copy()
        Y[:, 1] = np.log1p(Y_raw[:, 1])

        X_ord = _ordinal_features(X)
        X_oh = _onehot_features(X)

        r2_ord = _cv_r2_per_objective(X_ord, Y)
        r2_oh = _cv_r2_per_objective(X_oh, Y)

        summary[case_key] = {
            "n_samples": int(X.shape[0]),
            "n_features_ordinal": int(X_ord.shape[1]),
            "n_features_onehot": int(X_oh.shape[1]),
            "cv_r2_ordinal": r2_ord,
            "cv_r2_onehot":  r2_oh,
            "delta_r2": {k: r2_oh[k] - r2_ord[k] for k in r2_ord},
        }
        print(f"\n=== {case_key} (n_features ord={X_ord.shape[1]} oh={X_oh.shape[1]}) ===")
        for k in OBJ_NAMES:
            d = r2_oh[k] - r2_ord[k]
            print(f"  {k:>12s}  ord={r2_ord[k]:+.3f}  oh={r2_oh[k]:+.3f}  Δ={d:+.3f}")

    out = RESULTS_DIR / "onehot_cv_comparison.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
