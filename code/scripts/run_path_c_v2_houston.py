"""Path-C v2 verification on Houston (Case A) using DOE Reference Small Office
+ multi-orientation Radiance perimeter zone.

Outputs:
    data/results/path_c_v2_houston_subset.csv
    data/results/path_c_v2_houston_metrics.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_b.structural_opensees import utilization_fem
from mobo_envelope.path_c.energyplus_runner_v2 import run_eplus_v2
from mobo_envelope.path_c.radiance_runner_v2 import run_radiance_v2
from mobo_envelope.path_c.subset_selector import load_subsets

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"


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


def _agree(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
        return {"spearman": float("nan"), "pearson": float("nan")}
    rho, _ = spearmanr(a, b)
    r, _ = pearsonr(a, b)
    return {"spearman": float(rho), "pearson": float(r)}


def main() -> None:
    subsets = load_subsets(n_per_case=20)
    case_key = "A_Houston"
    sub = subsets[case_key]
    case = CASES["A"]
    print(f"=== Case A v2 on DOE RefBldg Small Office: {case.name}  ({len(sub)} subset designs) ===")

    rows = []
    t_start = time.time()
    for idx, row in sub.iterrows():
        d = _row_to_design(row)
        t0 = time.time()
        ep = run_eplus_v2(d, case.city)
        t_eplus = time.time() - t0
        t1 = time.time()
        rad = run_radiance_v2(d, case.city, n_amb=1)
        t_rad = time.time() - t1
        util_c = utilization_fem(d, case.city)
        rec = {
            "case": case.name,
            "solution_id": int(row["solution_id"]),
            "u_eff_w_m2k_path_a": float(row["u_eff_w_m2k"]),
            "structural_reserve_path_a": float(row["structural_reserve"]),
            "sda_pct_path_a": float(row["sda_pct"]),
            "cost_usd_m2": float(row["cost_usd_m2"]),
            "heating_eui_kwh_m2_yr_path_c_v2": ep.heating_eui_kwh_m2_yr if ep.success else None,
            "cooling_eui_kwh_m2_yr_path_c_v2": ep.cooling_eui_kwh_m2_yr if ep.success else None,
            "total_eui_kwh_m2_yr_path_c_v2":   ep.total_eui_kwh_m2_yr   if ep.success else None,
            "sda_pct_path_c_v2": rad.sda_pct if rad.success else None,
            "structural_utilization_path_c": float(util_c),
            "structural_reserve_path_c":     1.0 - float(util_c),
            "eplus_runtime_s": ep.runtime_s,
            "radiance_runtime_s": rad.runtime_s,
            "eplus_success": ep.success,
            "radiance_success": rad.success,
            "eplus_error": ep.error[:200] if not ep.success else "",
            "radiance_error": rad.error[:200] if not rad.success else "",
        }
        rows.append(rec)
        ep_str = f"{rec['total_eui_kwh_m2_yr_path_c_v2']:.1f}" if rec['total_eui_kwh_m2_yr_path_c_v2'] is not None else "FAIL"
        rad_str = f"{rec['sda_pct_path_c_v2']:.1f}" if rec['sda_pct_path_c_v2'] is not None else "FAIL"
        print(f"  [{idx + 1:>2}/{len(sub)}]  EUI={ep_str:>6}  sDA={rad_str:>6}  ({t_eplus:.1f}s + {t_rad:.1f}s)")

    df = pd.DataFrame(rows)
    ep_mask = df["eplus_success"]
    rad_mask = df["radiance_success"]
    therm = _agree(df.loc[ep_mask, "u_eff_w_m2k_path_a"], df.loc[ep_mask, "total_eui_kwh_m2_yr_path_c_v2"])
    struct = _agree(1.0 - df["structural_reserve_path_a"], df["structural_utilization_path_c"])
    daylight = _agree(df.loc[rad_mask, "sda_pct_path_a"], df.loc[rad_mask, "sda_pct_path_c_v2"])

    summary = {
        "case": case.name,
        "n_subset": len(df),
        "n_eplus_success": int(ep_mask.sum()),
        "n_radiance_success": int(rad_mask.sum()),
        "wall_clock_s": time.time() - t_start,
        "thermal_path_a_vs_c_v2":   therm,
        "structural_path_a_vs_c":   struct,
        "daylight_path_a_vs_c_v2":  daylight,
        "model_notes": {
            "energyplus_idf": "RefBldgSmallOfficeNew2004_Chicago.idf",
            "floor_area_m2": 511.16,
            "n_zones": 5,
            "radiance_geometry": "8x8x3 m perimeter zone with 4-orientation glazing",
            "radiance_n_sensors": 25,
        },
    }

    df.to_csv(RESULTS_DIR / "path_c_v2_houston_subset.csv", index=False)
    with open(RESULTS_DIR / "path_c_v2_houston_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"thermal Path-A vs Path-C v2:    spearman={therm['spearman']:.3f}  pearson={therm['pearson']:.3f}")
    print(f"daylight Path-A vs Path-C v2:   spearman={daylight['spearman']:.3f}  pearson={daylight['pearson']:.3f}")
    print(f"structural Path-A vs Path-C:    spearman={struct['spearman']:.3f}  pearson={struct['pearson']:.3f}")
    print(f"total wall-clock: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
