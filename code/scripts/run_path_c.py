"""Run Path-C (real EnergyPlus + openseespy) on the representative subset.

Outputs:
  - data/results/path_c_subset.csv  (per-design EUI, structural FEM)
  - data/results/path_c_metrics.json  (Path-A vs B vs C agreement)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_b.structural_opensees import utilization_fem  # already real OpenSees
from mobo_envelope.path_c.energyplus_runner import run_eplus
from mobo_envelope.path_c.radiance_runner import run_radiance
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
    rows: list[dict] = []
    summary = {}

    for case_key, sub in subsets.items():
        case_letter = case_key.split("_")[0]  # "A", "B", "C"
        case = CASES[case_letter]
        print(f"\n=== Case {case_letter}: {case.name}  ({len(sub)} subset designs) ===")

        per_case = []
        t_start = time.time()
        for idx, row in sub.iterrows():
            d = _row_to_design(row)
            t0 = time.time()
            ep = run_eplus(d, case.city)
            t_eplus = time.time() - t0
            t1 = time.time()
            rad = run_radiance(d, case.city, n_amb=3)
            t_rad = time.time() - t1
            util_c = utilization_fem(d, case.city)  # same OpenSees as Path-B for structural
            per_case.append({
                "case": case.name,
                "solution_id": int(row["solution_id"]),
                "u_eff_w_m2k_path_a":          float(row["u_eff_w_m2k"]),
                "structural_reserve_path_a":   float(row["structural_reserve"]),
                "sda_pct_path_a":              float(row["sda_pct"]),
                "cost_usd_m2":                 float(row["cost_usd_m2"]),
                "heating_eui_kwh_m2_yr_path_c": ep.heating_eui_kwh_m2_yr if ep.success else None,
                "cooling_eui_kwh_m2_yr_path_c": ep.cooling_eui_kwh_m2_yr if ep.success else None,
                "total_eui_kwh_m2_yr_path_c":   ep.total_eui_kwh_m2_yr   if ep.success else None,
                "sda_pct_path_c":               rad.sda_pct if rad.success else None,
                "structural_utilization_path_c": float(util_c),
                "structural_reserve_path_c":     1.0 - float(util_c),
                "eplus_runtime_s":               ep.runtime_s,
                "radiance_runtime_s":            rad.runtime_s,
                "eplus_success":                 ep.success,
                "radiance_success":              rad.success,
                "eplus_error":                   ep.error[:200] if not ep.success else "",
                "radiance_error":                rad.error[:200] if not rad.success else "",
            })
            rows.append(per_case[-1])
            print(f"  [{idx + 1:>2}/{len(sub)}]  EUI={per_case[-1]['total_eui_kwh_m2_yr_path_c']!s:>6}  sDA_C={per_case[-1]['sda_pct_path_c']!s:>6}  ({t_eplus:.1f}s + {t_rad:.1f}s)")

        df = pd.DataFrame(per_case)
        ep_mask = df["eplus_success"]
        rad_mask = df["radiance_success"]
        therm_ag = _agree(df.loc[ep_mask, "u_eff_w_m2k_path_a"], df.loc[ep_mask, "total_eui_kwh_m2_yr_path_c"])
        struct_ag = _agree(1.0 - df["structural_reserve_path_a"], df["structural_utilization_path_c"])
        daylight_ag = _agree(df.loc[rad_mask, "sda_pct_path_a"], df.loc[rad_mask, "sda_pct_path_c"])
        summary[case.name] = {
            "n_subset": len(df),
            "n_eplus_success": int(ep_mask.sum()),
            "n_radiance_success": int(rad_mask.sum()),
            "wall_clock_s": time.time() - t_start,
            "thermal_path_a_vs_c":   therm_ag,
            "structural_path_a_vs_c": struct_ag,
            "daylight_path_a_vs_c":   daylight_ag,
        }
        print(f"  thermal Path-A vs Path-C:    spearman={therm_ag['spearman']:.3f}  pearson={therm_ag['pearson']:.3f}")
        print(f"  daylight Path-A vs Path-C:   spearman={daylight_ag['spearman']:.3f}  pearson={daylight_ag['pearson']:.3f}")
        print(f"  structural Path-A vs Path-C: spearman={struct_ag['spearman']:.3f}  pearson={struct_ag['pearson']:.3f}")
        print(f"  total wall-clock: {time.time() - t_start:.1f}s")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "path_c_subset.csv", index=False)
    with open(RESULTS_DIR / "path_c_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {len(rows)} rows to path_c_subset.csv")


if __name__ == "__main__":
    main()
