"""Run Path-B high-fidelity verification on every Pareto winner.

For each case (Houston / NYC / Minneapolis):
  1. Load the Path-A Pareto front from data/results/case_*_pareto.csv
  2. Reconstruct each design from the columns (no need to re-run the optimizer)
  3. Run all three Path-B simulators per design:
       - ISO 13790 hourly thermal -> heating + cooling EUI
       - IES LM-83-12 daylight -> sDA at 300lux/50%
       - openseespy FEM -> mullion utilization with P-delta
  4. Compute analytical-vs-Path-B agreement metrics:
       - R^2 across the front (regression of high-fidelity on analytical)
       - Spearman rank correlation (does ranking survive?)
       - MAPE (mean absolute percentage error)
  5. Write data/results/verification.csv + verification_metrics.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from mobo_envelope.cases import CASES
from mobo_envelope.envelope import EnvelopeDesign
from mobo_envelope.path_b.daylight_lm83 import sda_lm83
from mobo_envelope.path_b.structural_opensees import utilization_fem
from mobo_envelope.path_b.thermal_iso13790 import annual_eui
from mobo_envelope.path_b.weather import load_weather

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


def _agreement_metrics(analytical: np.ndarray, path_b: np.ndarray) -> dict[str, float]:
    """Compute rank-based and scale-invariant agreement metrics.

    Path-A and Path-B can produce different-unit quantities (e.g. U_eff in
    W/m^2.K vs annual EUI in kWh/m^2/yr) so absolute-error metrics like RMSE
    and MAPE are not meaningful. The metrics that matter:

      - Spearman rank correlation (does the optimizer's ranking survive?)
      - Pearson correlation (linear relationship strength)
      - NRMSE_zscore (RMSE on z-scored series, scale invariant)
    """
    a = np.asarray(analytical, dtype=float)
    b = np.asarray(path_b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "nrmse_zscore": float("nan"),
        }

    rho, _ = spearmanr(a, b)
    r, _ = pearsonr(a, b)

    az = (a - a.mean()) / max(1e-12, a.std())
    bz = (b - b.mean()) / max(1e-12, b.std())
    nrmse = float(np.sqrt(np.mean((az - bz) ** 2)))

    return {
        "spearman": float(rho),
        "pearson": float(r),
        "nrmse_zscore": nrmse,
    }


def main() -> None:
    case_files = {
        "A": "case_a_houston_pareto.csv",
        "B": "case_b_nyc_pareto.csv",
        "C": "case_c_minneapolis_pareto.csv",
    }

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for case_key, fname in case_files.items():
        case = CASES[case_key]
        weather = load_weather(case.city)
        df = pd.read_csv(RESULTS_DIR / fname)
        n = len(df)
        print(f"\n=== Case {case_key}: {case.name}  (n={n} Pareto winners) ===")

        per_case_rows: list[dict] = []
        t_start = time.time()
        for idx, row in df.iterrows():
            design = _row_to_design(row)

            t0 = time.time()
            therm = annual_eui(design, weather)
            t_therm = time.time() - t0

            t0 = time.time()
            light = sda_lm83(design, weather)
            t_light = time.time() - t0

            t0 = time.time()
            util_b = utilization_fem(design, case.city)
            t_struct = time.time() - t0

            per_case_rows.append({
                "case": case.name,
                "solution_id": int(row["solution_id"]),
                # analytical (Path A) values (already in CSV)
                "u_eff_w_m2k_path_a":            float(row["u_eff_w_m2k"]),
                "structural_reserve_path_a":     float(row["structural_reserve"]),
                "sda_pct_path_a":                float(row["sda_pct"]),
                "cost_usd_m2":                   float(row["cost_usd_m2"]),
                # Path B values
                "heating_eui_kwh_m2_yr":         therm["heating_eui_kwh_m2_yr"],
                "cooling_eui_kwh_m2_yr":         therm["cooling_eui_kwh_m2_yr"],
                "total_eui_kwh_m2_yr":           therm["total_eui_kwh_m2_yr"],
                "sda_pct_path_b":                light["sda_pct_path_b"],
                "df_mean_pct":                   light["df_mean_pct"],
                "structural_utilization_path_b": util_b,
                "structural_reserve_path_b":     1.0 - util_b,
            })
            if (idx + 1) % 50 == 0:
                done_frac = (idx + 1) / n
                eta = (time.time() - t_start) * (1 - done_frac) / max(1e-3, done_frac)
                print(f"  [{idx + 1:>3}/{n}]  thermal {t_therm * 1000:.0f}ms  daylight {t_light * 1000:.0f}ms  fem {t_struct * 1000:.0f}ms  ETA {eta:.0f}s")

        per_case_df = pd.DataFrame(per_case_rows)
        all_rows.extend(per_case_rows)

        # Agreement metrics for this case
        therm_metrics = _agreement_metrics(
            per_case_df["u_eff_w_m2k_path_a"].to_numpy(),
            per_case_df["total_eui_kwh_m2_yr"].to_numpy(),
        )
        light_metrics = _agreement_metrics(
            per_case_df["sda_pct_path_a"].to_numpy(),
            per_case_df["sda_pct_path_b"].to_numpy(),
        )
        struct_metrics = _agreement_metrics(
            (1.0 - per_case_df["structural_reserve_path_a"]).to_numpy(),
            per_case_df["structural_utilization_path_b"].to_numpy(),
        )

        summary[case.name] = {
            "n_winners": n,
            "wall_clock_s": time.time() - t_start,
            "thermal_path_a_vs_path_b_eui": therm_metrics,
            "daylight_path_a_vs_path_b_sda": light_metrics,
            "structural_path_a_vs_path_b_util": struct_metrics,
            "path_b_means": {
                "heating_eui": float(per_case_df["heating_eui_kwh_m2_yr"].mean()),
                "cooling_eui": float(per_case_df["cooling_eui_kwh_m2_yr"].mean()),
                "total_eui":   float(per_case_df["total_eui_kwh_m2_yr"].mean()),
                "sda_pct":     float(per_case_df["sda_pct_path_b"].mean()),
                "utilization": float(per_case_df["structural_utilization_path_b"].mean()),
            },
        }

        print(f"  thermal U_eff vs Path-B EUI:   spearman={therm_metrics['spearman']:.3f}  pearson={therm_metrics['pearson']:.3f}  NRMSE_z={therm_metrics['nrmse_zscore']:.3f}")
        print(f"  daylight Path-A sDA vs LM-83:  spearman={light_metrics['spearman']:.3f}  pearson={light_metrics['pearson']:.3f}  NRMSE_z={light_metrics['nrmse_zscore']:.3f}")
        print(f"  structural analytical vs FEM:  spearman={struct_metrics['spearman']:.3f}  pearson={struct_metrics['pearson']:.3f}  NRMSE_z={struct_metrics['nrmse_zscore']:.3f}")
        print(f"  total wall-clock for {n} designs: {time.time() - t_start:.1f}s")

    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "verification.csv", index=False)
    with open(RESULTS_DIR / "verification_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {len(all_rows)} rows to verification.csv")


if __name__ == "__main__":
    main()
