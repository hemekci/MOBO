# MOBO-Envelope

Code and data for the article

> S. Emekci, A. Abbas, H. Emekci. **Multi-Objective Optimisation of Building Envelopes:
> A Surrogate-Assisted Three-Tier Verification Framework.** *Building and Environment*, 2026.

MOBO-Envelope optimises four envelope objectives (effective U-value, structural reserve,
spatial daylight autonomy and material cost) and verifies the result at three levels of fidelity:

| Tier | What it does | Engines |
|---|---|---|
| **Path-A** | NSGA-III search accelerated by an ARD Matérn-5/2 Gaussian-process surrogate | scikit-learn, pymoo |
| **Path-B** | Standards-grade verification of the full Pareto front | openseespy (ASCE 7-22 FEA), ISO 13790 hourly thermal, IES LM-83-12 sDA, pvlib |
| **Path-C** | High-fidelity cross-check against external simulator binaries | EnergyPlus 26.1, Radiance 6.0 |

Case studies: Houston (ASHRAE CZ 2A), New York City (4A), Minneapolis (6A).

## Repository layout

```
code/
  src/mobo_envelope/     framework package
    constants.py         published input values (ASHRAE, ASCE, ISO, RSMeans), cited inline
    envelope.py          design vector -> physical envelope
    thermal.py, structural.py, daylight.py, cost.py   Path-A objectives
    surrogate.py         Gaussian-process surrogate
    optimize.py          surrogate-assisted NSGA-III loop
    cases.py             case definitions and baselines
    path_b/              standards-grade Python simulators
    path_c/              EnergyPlus / Radiance runners, k-medoids subset selector
  scripts/               one script per experiment (see below)
  tests/
data/
  raw/weather/           TMY3 EPW files for the three cities
  results/               every result reported in the paper (CSV / JSON)
figures/                 the figures as published
```

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
cd code
uv sync
uv run pytest
```

Path-A and Path-B run with Python only. Path-C additionally needs
[EnergyPlus 26.1](https://energyplus.net) and [Radiance 6.0](https://www.radiance-online.org);
set their install locations in `code/scripts/sim_env.sh` and `source` it before running Path-C scripts.

## Reproducing the results

Run from `code/`. Each script writes to `data/results/`.

| Script | Produces |
|---|---|
| `run_optimization.py` | Path-A Pareto fronts, best-compromise designs, wall-clock cost |
| `compute_compliant_baselines.py` | B1 (as-practised) and B2 (code-compliant) baselines |
| `verify_surrogate.py` | surrogate cross-validation, hold-out and out-of-distribution accuracy |
| `run_multiseed.py` | five-seed hypervolume statistics with bootstrap CIs |
| `verify_pareto.py` | Path-B verification of every Pareto design |
| `run_path_c.py`, `run_path_c_v2_houston.py` | Path-C verification on the k-medoids subset |
| `run_path_c_full_front.py` | Path-C verification of the complete front (all 789 designs) |
| `run_path_c_direct_opt.py` | independently optimised high-fidelity reference front |
| `run_ase_all_cases.py` | ASE<sub>1000,250</sub> and sDA on the verification subsets |
| `algorithmic_baseline.py` | surrogate-assisted pipeline vs vanilla NSGA-II / NSGA-III |
| `sensitivity.py`, `sensitivity_pathb.py` | Sobol indices (Path-A objectives, Path-B annual EUI) |
| `run_sensitivity_package.py` | seed, reference-point and total-order Sobol robustness checks |
| `onehot_categorical_cv.py` | ordinal vs one-hot encoding of categorical variables |
| `regenerate_all_figures.py`, `regenerate_pareto_figures.py`, `regenerate_v2_comparison_figure.py`, `fig_independent_front.py` | figures |

```bash
uv run python scripts/run_optimization.py          # ~1 min per case
uv run python scripts/regenerate_all_figures.py
```

All runs are seeded (default seed 42). Radiance sky matrices are cached in
`data/raw/sky_cache/` on the first Path-C run.

## Cost data

Unit costs are representative RSMeans 2024 values, documented per category in
`code/src/mobo_envelope/constants.py`. Users with their own cost data can substitute
it there without changing the optimisation code.

## Licence

MIT, see [LICENSE](LICENSE). If you use this code, please cite the article (see `CITATION.cff`).
