"""Compute Sobol first-order indices for each objective per case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample as sobol_sample

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER, VARIABLE_NAMES
from mobo_envelope.objectives import evaluate_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    problem = {
        "num_vars": N_VARS,
        "names": VARIABLE_NAMES,
        "bounds": [[lo, hi] for lo, hi in zip(LOWER.tolist(), UPPER.tolist())],
    }
    rows: list[dict] = []
    obj_pretty = ["thermal", "structural", "daylight", "cost"]

    for case_key, case in CASES.items():
        # N=512 base samples -> 512*(2*N_VARS + 2) = 11264 evaluations
        x_sob = sobol_sample(problem, N=512, seed=42)
        y_sob = evaluate_batch(x_sob, case.city)
        # invert daylight sign so larger Si means "more important to sda"
        y_sob_signed = y_sob.copy()
        y_sob_signed[:, 2] = -y_sob_signed[:, 2]

        for j, obj_name in enumerate(obj_pretty):
            try:
                Si = analyze(problem, y_sob_signed[:, j], print_to_console=False, seed=42)
                for var_name, s1 in zip(VARIABLE_NAMES, Si["S1"]):
                    rows.append({
                        "objective": obj_name,
                        "case": case.name,
                        "variable": var_name,
                        "sobol_s1": round(float(s1), 3),
                    })
            except Exception as e:
                print(f"[warn] {case.name} {obj_name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "sensitivity_analysis.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'sensitivity_analysis.csv'} ({len(df)} rows)")
    print("\nTop 5 per objective (averaged across cases):")
    avg = df.groupby(["objective", "variable"]).sobol_s1.mean().reset_index()
    for obj in obj_pretty:
        sub = avg[avg.objective == obj].sort_values("sobol_s1", ascending=False).head(5)
        print(f"\n  {obj}:")
        for _, r in sub.iterrows():
            print(f"    {r.variable:22s}  S1={r.sobol_s1:+.3f}")


if __name__ == "__main__":
    main()
