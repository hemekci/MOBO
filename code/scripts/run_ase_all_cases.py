"""ASE_1000,250 and v2 sDA on the Path-C verification subsets, all three cases.

Complements sDA with a second daylight metric. This script
computes IES LM-83 ASE on the same k-medoids verification subset already used
for Path-C, in every case, and re-computes v2 sDA alongside it so that both
metrics come from the identical four-orientation geometry.

v2 sDA previously existed for Houston only (path_c_v2_houston_subset.csv); NYC
and Minneapolis carried v1 (south-only) sDA. Recomputing v2 sDA here puts all
three cases on the same footing, which is what makes the cross-climate
comparison meaningful.

Outputs:
    data/results/ase_v2_all_cases.csv    per-design sDA_v2, ASE, exceed-hours
    data/results/ase_v2_all_cases.json   per-case summary and correlations

Usage:
    source scripts/sim_env.sh
    .venv/bin/python scripts/run_ase_all_cases.py [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_c.radiance_ase_v2 import run_ase_v2
from mobo_envelope.path_c.radiance_runner_v2 import run_radiance_v2
from mobo_envelope.path_c.subset_selector import load_subsets

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

CASE_KEY_TO_ID = {"A_Houston": "A", "B_NYC": "B", "C_Minneapolis": "C"}


def _row_to_design(row: pd.Series) -> EnvelopeDesign:
    return EnvelopeDesign(
        wwr_north=float(row["wwr_north"]),
        wwr_south=float(row["wwr_south"]),
        wwr_east=float(row["wwr_east"]),
        wwr_west=float(row["wwr_west"]),
        insulation_m=float(row["insulation_m"]),
        glazing_type=str(row["glazing_type"]),
        member_depth_m=float(row["member_depth_m"]),
        framing_material=str(row["framing_material"]),
        overhang_ratio=float(row["overhang_ratio"]),
        panel_thickness_m=float(row["panel_thickness_m"]),
    )


def evaluate_one(args: tuple[dict, str, str, int]) -> dict:
    """Compute v2 sDA and ASE for one design. Runs in a worker process."""
    row_dict, city, case_key, solution_id = args
    d = _row_to_design(pd.Series(row_dict))
    t0 = time.time()
    sda = run_radiance_v2(d, city, n_amb=1)
    ase = run_ase_v2(d, city)
    return {
        "case": case_key,
        "solution_id": solution_id,
        "sda_pct_path_a": float(row_dict["sda_pct"]),
        "sda_pct_v2": float(sda.sda_pct) if sda.success else None,
        "ase_pct": float(ase.ase_pct) if ase.success else None,
        "mean_exceed_hours": float(ase.mean_exceed_hours) if ase.success else None,
        "max_exceed_hours": float(ase.max_exceed_hours) if ase.success else None,
        "wwr_mean": float(np.mean([row_dict["wwr_north"], row_dict["wwr_south"],
                                   row_dict["wwr_east"], row_dict["wwr_west"]])),
        "overhang_ratio": float(row_dict["overhang_ratio"]),
        "sda_success": bool(sda.success),
        "ase_success": bool(ase.success),
        "sda_error": sda.error[:200] if not sda.success else "",
        "ase_error": ase.error[:200] if not ase.success else "",
        "runtime_s": time.time() - t0,
    }


def _rho(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="keep low while another simulation job is running")
    ap.add_argument("--n-per-case", type=int, default=20)
    args = ap.parse_args()

    subsets = load_subsets(n_per_case=args.n_per_case)
    payload: list[tuple[dict, str, str, int]] = []
    for case_key, sub in subsets.items():
        city = CASES[CASE_KEY_TO_ID[case_key]].city
        for _, row in sub.iterrows():
            payload.append((row.to_dict(), city, case_key, int(row["solution_id"])))

    print(f"=== ASE + v2 sDA on {len(payload)} designs "
          f"({len(subsets)} cases x {args.n_per_case}) ===", flush=True)

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(evaluate_one, payload))
    else:
        rows = [evaluate_one(p) for p in payload]
    wall = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ase_v2_all_cases.csv", index=False)

    summary: dict[str, dict] = {}
    for case_key, g in df.groupby("case"):
        ok = g[g.sda_success & g.ase_success]
        summary[case_key] = {
            "n": int(len(g)),
            "n_success": int(len(ok)),
            "sda_v2": {
                "n_distinct": int(ok.sda_pct_v2.nunique()),
                "min": float(ok.sda_pct_v2.min()),
                "max": float(ok.sda_pct_v2.max()),
                "std": float(ok.sda_pct_v2.std()),
                "pct_at_100": float((ok.sda_pct_v2 >= 99.5).mean() * 100.0),
            },
            "ase": {
                "n_distinct": int(ok.ase_pct.nunique()),
                "min": float(ok.ase_pct.min()),
                "max": float(ok.ase_pct.max()),
                "std": float(ok.ase_pct.std()),
                "pct_over_lm83_10pct": float((ok.ase_pct > 10.0).mean() * 100.0),
            },
            "exceed_hours": {
                "min": float(ok.mean_exceed_hours.min()),
                "max": float(ok.mean_exceed_hours.max()),
                "std": float(ok.mean_exceed_hours.std()),
            },
            "spearman": {
                "wwr_vs_sda_v2": _rho(ok.wwr_mean, ok.sda_pct_v2),
                "wwr_vs_ase": _rho(ok.wwr_mean, ok.ase_pct),
                "wwr_vs_exceed_hours": _rho(ok.wwr_mean, ok.mean_exceed_hours),
                "overhang_vs_exceed_hours": _rho(ok.overhang_ratio, ok.mean_exceed_hours),
                "path_a_sda_vs_sda_v2": _rho(ok.sda_pct_path_a, ok.sda_pct_v2),
                "path_a_sda_vs_ase": _rho(ok.sda_pct_path_a, ok.ase_pct),
            },
        }

    meta = {"wall_clock_s": wall, "n_designs": int(len(df)), "per_case": summary}
    with open(RESULTS_DIR / "ase_v2_all_cases.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwall clock: {wall / 60:.1f} min\n")
    hdr = f"{'case':16s} {'sDA distinct':>12} {'sDA std':>8} {'ASE range':>14} {'ASE std':>8} {'>LM83':>7}"
    print(hdr); print("-" * len(hdr))
    for k, s in summary.items():
        print(f"{k:16s} {s['sda_v2']['n_distinct']:>12} {s['sda_v2']['std']:>8.2f} "
              f"{s['ase']['min']:>6.0f}-{s['ase']['max']:<7.0f} {s['ase']['std']:>8.2f} "
              f"{s['ase']['pct_over_lm83_10pct']:>6.0f}%")
    print("\nwritten: ase_v2_all_cases.{csv,json}")


if __name__ == "__main__":
    main()
