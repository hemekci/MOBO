"""Experiment C: Path-C v2 verification of the COMPLETE Pareto front.

A 20-design k-medoids subset per case is chosen to *represent the spread* of
the front, not to contain its best-performing members, so it is not a fair
basis for comparison against an independently optimised high-fidelity front.

This script evaluates every design on every Pareto front
at Path-C v2 (the same model used for the independent reference run), so the
two fronts are compared on equal terms.

Outputs (per case):
    data/results/path_c_full_front_{case}.csv     every design, Path-C v2 metrics
    data/results/path_c_full_front_{case}_meta.json

The CSV is checkpointed every 25 designs, so an interrupted run keeps its work
and can be resumed with --resume.

Usage:
    source scripts/sim_env.sh
    .venv/bin/python scripts/run_path_c_full_front.py --case A --workers 5
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mobo_envelope import cost as cost_mod
from mobo_envelope.cases import CASES
from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_b.structural_opensees import utilization_fem
from mobo_envelope.path_c.energyplus_runner_v2 import run_eplus_v2
from mobo_envelope.path_c.radiance_runner_v2 import run_radiance_v2

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"

CASE_FRONT_CSV = {
    "A": ("A_Houston", "case_a_houston_pareto.csv"),
    "B": ("B_NYC", "case_b_nyc_pareto.csv"),
    "C": ("C_Minneapolis", "case_c_minneapolis_pareto.csv"),
}


def _row_to_design(row: dict) -> EnvelopeDesign:
    return EnvelopeDesign(
        wwr_north=float(row["wwr_north"]), wwr_south=float(row["wwr_south"]),
        wwr_east=float(row["wwr_east"]), wwr_west=float(row["wwr_west"]),
        insulation_m=float(row["insulation_m"]), glazing_type=str(row["glazing_type"]),
        member_depth_m=float(row["member_depth_m"]),
        framing_material=str(row["framing_material"]),
        overhang_ratio=float(row["overhang_ratio"]),
        panel_thickness_m=float(row["panel_thickness_m"]),
    )


def evaluate_one(args: tuple[dict, str, str]) -> dict:
    """Path-C v2 evaluation of one Pareto design. Runs in a worker process."""
    row, city, case_key = args
    d = _row_to_design(row)
    t0 = time.time()
    ep = run_eplus_v2(d, city)
    rad = run_radiance_v2(d, city, n_amb=1)
    try:
        util = float(utilization_fem(d, city))
        fem_ok = True
    except Exception:  # noqa: BLE001 - recorded, never aborts the sweep
        util, fem_ok = float("nan"), False
    return {
        "case": case_key,
        "solution_id": int(row["solution_id"]),
        # Path-A (search-space) values, carried through for the comparison
        "u_eff_w_m2k_path_a": float(row["u_eff_w_m2k"]),
        "structural_reserve_path_a": float(row["structural_reserve"]),
        "sda_pct_path_a": float(row["sda_pct"]),
        "cost_usd_m2": float(cost_mod.cost_per_envelope_m2(d)),
        # Path-C v2 (high-fidelity) values
        "total_eui_kwh_m2_yr_path_c_v2": float(ep.total_eui_kwh_m2_yr) if ep.success else None,
        "heating_eui_kwh_m2_yr_path_c_v2": float(ep.heating_eui_kwh_m2_yr) if ep.success else None,
        "cooling_eui_kwh_m2_yr_path_c_v2": float(ep.cooling_eui_kwh_m2_yr) if ep.success else None,
        "sda_pct_path_c_v2": float(rad.sda_pct) if rad.success else None,
        "structural_utilization_path_c": util if fem_ok else None,
        "eplus_success": bool(ep.success), "radiance_success": bool(rad.success),
        "fem_success": fem_ok,
        "success": bool(ep.success and rad.success and fem_ok),
        "eplus_error": ep.error[:200] if not ep.success else "",
        "radiance_error": rad.error[:200] if not rad.success else "",
        "eval_time_s": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=sorted(CASE_FRONT_CSV))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole front")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    case_key, csv_name = CASE_FRONT_CSV[args.case]
    city = CASES[args.case].city
    front = pd.read_csv(RESULTS_DIR / csv_name)
    if args.limit:
        front = front.head(args.limit)

    out_csv = RESULTS_DIR / f"path_c_full_front_{case_key}.csv"
    done: set[int] = set()
    rows: list[dict] = []
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(prev.solution_id.astype(int))
        print(f"resuming: {len(done)} designs already evaluated")

    todo = [r for r in front.to_dict("records") if int(r["solution_id"]) not in done]
    print(f"=== Path-C v2 full-front verification: {case_key} ===")
    print(f"front size {len(front)}, to evaluate {len(todo)}, workers {args.workers}")
    print(f"estimated wall clock ~{len(todo) * 11.2 / args.workers / 60:.0f} min", flush=True)

    t_start = time.time()
    payload = [(r, city, case_key) for r in todo]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(evaluate_one, payload), 1):
            rows.append(rec)
            if i % 25 == 0 or i == len(payload):
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                el = time.time() - t_start
                rate = el / i
                print(f"  {i:>4}/{len(payload)}  {el/60:5.1f} min elapsed, "
                      f"~{(len(payload)-i)*rate/60:5.1f} min left  "
                      f"(fail {sum(1 for r in rows if not r['success'])})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    meta = {
        "case": case_key, "city": city,
        "front_size": int(len(front)),
        "n_evaluated": int(len(df)),
        "n_success": int(df.success.sum()),
        "n_eplus_fail": int((~df.eplus_success).sum()),
        "n_radiance_fail": int((~df.radiance_success).sum()),
        "wall_clock_s": time.time() - t_start,
        "mean_eval_s": float(df.eval_time_s.mean()),
        "model": "Path-C v2 (DOE RefBldg Small Office + 4-orientation Radiance)",
    }
    with open(RESULTS_DIR / f"path_c_full_front_{case_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nevaluated {meta['n_evaluated']} designs, {meta['n_success']} successful, "
          f"{(time.time()-t_start)/60:.1f} min")
    print(f"written: path_c_full_front_{case_key}.csv")


if __name__ == "__main__":
    main()
