"""Sobol global sensitivity using Path-B climate-sensitive thermal EUI.

Why a separate script: the existing sensitivity.py computes Sobol on the
Path-A analytical objectives, which are climate-invariant for U_eff and
cost (city only enters multiplicatively in the structural and daylight
proxies). For genuinely climate-dependent feature importance this script
retargets the thermal Sobol to
Path-B annual EUI from the ISO 13790 hourly balance, which IS sensitive
to weather data.

Output: data/results/sensitivity_pathb_thermal.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample as sobol_sample

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import LOWER, N_VARS, UPPER, VARIABLE_NAMES, decode_design
from mobo_envelope.path_b.thermal_iso13790 import annual_eui
from mobo_envelope.path_b.weather import load_weather

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"


def main() -> None:
    problem = {
        "num_vars": N_VARS,
        "names": VARIABLE_NAMES,
        "bounds": [[lo, hi] for lo, hi in zip(LOWER.tolist(), UPPER.tolist())],
    }
    rows: list[dict] = []
    # N=64 left the Saltelli estimator noisier than the effects it measures:
    # 12 of 15 (S_1, S_T) pairs came back with S_T < S_1, which the Sobol
    # decomposition forbids. Path-B costs 0.5 ms per evaluation, so N=512
    # (the Path-A budget) is affordable and removes the artefact.
    n_base = 512

    for case_key, case in CASES.items():
        weather = load_weather(case.city)
        x_sob = sobol_sample(problem, N=n_base, seed=42)
        n_eval = x_sob.shape[0]
        print(f"\n=== {case.name}  ({n_eval} evaluations) ===")

        t0 = time.time()
        eui_total = np.empty(n_eval, dtype=float)
        for i in range(n_eval):
            d = decode_design(x_sob[i])
            res = annual_eui(d, weather)
            eui_total[i] = float(res["total_eui_kwh_m2_yr"])
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{n_eval}  ({time.time() - t0:.1f}s)")

        Si = analyze(problem, eui_total, print_to_console=False, seed=42)
        for var_name, s1, st in zip(VARIABLE_NAMES, Si["S1"], Si["ST"]):
            rows.append({
                "objective": "thermal_total_eui_path_b",
                "case": case.name,
                "variable": var_name,
                "sobol_s1": round(float(s1), 3),
                "sobol_st": round(float(st), 3),
            })
        # Print top-5 for this case
        df_case = pd.DataFrame([r for r in rows if r["case"] == case.name])
        df_case_sorted = df_case.sort_values("sobol_st", ascending=False)
        print(f"  Top 5 by ST:")
        for _, r in df_case_sorted.head(5).iterrows():
            print(f"    {r.variable:22s}  S1={r.sobol_s1:+.3f}  ST={r.sobol_st:+.3f}")
        print(f"  wall-clock: {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "sensitivity_pathb_thermal.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'sensitivity_pathb_thermal.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
