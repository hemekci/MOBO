"""Pick a representative subset of Pareto winners for Path-C verification.

Strategy: a balanced selection covering the four-objective space:
  - Best in each single objective (4 designs)
  - Best-compromise (Euclidean to ideal) on full and on cost-constrained subset (2)
  - Stratified random sample over the front (~14 designs)

Total ~20 designs per case. Selection is reproducible via a fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = REPO_ROOT / "data" / "results"


def _normalize(f: np.ndarray) -> np.ndarray:
    lo = f.min(axis=0)
    hi = f.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (f - lo) / span


def select_subset(
    pareto_df: pd.DataFrame,
    n_total: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a representative subset of Pareto solutions."""
    if len(pareto_df) <= n_total:
        return pareto_df.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Build the 4-objective matrix used for selection
    F = np.column_stack([
        pareto_df["u_eff_w_m2k"].to_numpy(),
        1.0 - pareto_df["structural_reserve"].to_numpy(),  # utilization (minimize)
        -pareto_df["sda_pct"].to_numpy(),                  # negate sDA (minimize)
        pareto_df["cost_usd_m2"].to_numpy(),
    ])
    Fn = _normalize(F)

    selected: list[int] = []

    # 4 single-objective extremes
    for j in range(4):
        idx = int(np.argmin(Fn[:, j]))
        if idx not in selected:
            selected.append(idx)

    # Best compromise: minimum Euclidean distance to ideal point
    ideal = Fn.min(axis=0)
    dist = np.linalg.norm(Fn - ideal, axis=1)
    bc = int(np.argmin(dist))
    if bc not in selected:
        selected.append(bc)

    # Stratified random over remaining: split front into n_strata bins by
    # weighted-rank and sample one per bin
    n_remaining = n_total - len(selected)
    if n_remaining > 0:
        weighted_rank = Fn.sum(axis=1)
        order = np.argsort(weighted_rank)
        # exclude already selected
        remaining = [i for i in order if i not in selected]
        n_strata = min(n_remaining, len(remaining))
        chunks = np.array_split(remaining, n_strata)
        for chunk in chunks:
            if len(chunk) > 0:
                idx = int(rng.choice(chunk))
                if idx not in selected:
                    selected.append(idx)

    selected = sorted(set(selected))
    return pareto_df.iloc[selected].reset_index(drop=True)


def load_subsets(n_per_case: int = 20) -> dict[str, pd.DataFrame]:
    """Convenience: load Pareto fronts for all 3 cases and return subsets."""
    case_files = {
        "A_Houston":     "case_a_houston_pareto.csv",
        "B_NYC":         "case_b_nyc_pareto.csv",
        "C_Minneapolis": "case_c_minneapolis_pareto.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for case, fname in case_files.items():
        df = pd.read_csv(RESULTS_DIR / fname)
        out[case] = select_subset(df, n_total=n_per_case)
    return out
