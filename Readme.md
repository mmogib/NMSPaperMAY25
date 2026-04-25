# SGSD-DEED — Companion Code

[![Julia](https://img.shields.io/badge/Julia-1.10+-purple.svg)](https://julialang.org/)
[![Solver](https://img.shields.io/badge/Ipopt-3.14%20%2B%20MA27-blue.svg)](https://github.com/coin-or/Ipopt)
[![DRDeed.jl](https://img.shields.io/badge/DRDeed.jl-v0.5.0-blue.svg)](https://github.com/mmogib/DRDeed.jl)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

Companion repository for the manuscript:

> **A Stackelberg Game for Multi-Objective Demand Response in Dynamic Economic Emission Dispatch**
>
> Norah Almuraysil, Mohammed Alshahrani (corresponding), Slim Belhaiza
>
> *International Journal of Electrical Power & Energy Systems* (under revision, manuscript ID IJEPES-D-26-00611)

This repository contains the Julia experiment scripts that generate every numerical result, figure, and table reported in the paper. The optimization models themselves live in the companion package [DRDeed.jl](https://github.com/mmogib/DRDeed.jl).

---

## Repository Layout

```
.
├── experiments.jl                # Unified loader / orchestrator
├── experiments/                  # One Julia script per experiment
│   ├── exp_drdeed_comparison.jl
│   ├── exp_scalability.jl
│   ├── exp_scaling_walltime.jl
│   ├── exp_ieee30_validation.jl
│   ├── exp_model_progression.jl
│   ├── exp_sensitivity.jl
│   ├── exp_saudi_case_study.jl
│   ├── exp_pareto_analysis.jl
│   ├── exp_metaheuristic_comparison.jl
│   └── exp_vpl.jl
├── analysis.jl                   # Post-hoc data analysis helpers
├── plotting.jl                   # Plotting utilities
├── utils.jl                      # Shared utilities
├── timing_summary.jl             # Generates the wall-clock summary table
├── verify_sensitivity_stats.jl   # Drift detector for sensitivity headline stats
├── datasets/                     # Input data (PJM, IEEE 30-bus, SEC)
├── results/                      # All output (gitignored — regenerate locally)
└── Project.toml                  # Julia environment manifest
```

---

## Experiment → Script Mapping

| # | Manuscript section | Script | What it produces |
|---|---|---|---|
| 1 | §4.2 — DR-DEED comparison | `experiments/exp_drdeed_comparison.jl` | SC1/SC2 comparison vs DR-DEED across 4 weight schemes |
| 2 | §4.3 — Scalability grid | `experiments/exp_scalability.jl` | Customer × generator grid, runtime + cost statistics |
| 2′ | §4.5.4 — Extended scaling | `experiments/exp_scalability.jl` (mode `:extended`), `experiments/exp_scaling_walltime.jl` | n ∈ {50, 100, 200, 400}, log-log fit T ≈ 0.012·n^1.82 |
| 3 | §4.4 — IEEE 30-bus DC-OPF | `experiments/exp_ieee30_validation.jl` | DC-OPF cost premium under SC1/SC2 demand |
| 4 | §4.5 — Model progression | `experiments/exp_model_progression.jl` | DEED → DR-DEED → SGSD-DEED on SC1/SC2 |
| 5 | §4.6 — Sensitivity sweeps | `experiments/exp_sensitivity.jl` | 5A weights (64-pt simplex), 5B θ, 5C storage, 5D customers, 5F demand multiplier, 5G tariff multiplier; six-parameter tornado |
| 5E | §4.6 — Pareto front | `experiments/exp_pareto_analysis.jl` | ε-constraint Pareto front (cost vs emission) |
| 6 | §4.7 — Saudi Eastern Province | `experiments/exp_saudi_case_study.jl` | SEC-calibrated regional case study |
| 7 | §4.8 — Valve-point loading | `experiments/exp_vpl.jl` | 1 smooth + 10 multi-start VPL solves on SC1 |
| — | Remark in §4 | `experiments/exp_metaheuristic_comparison.jl` | NSGA-II vs Ipopt on the same Stackelberg formulation |
| — | §4.9 — Wall-clock summary | `timing_summary.jl` | Generates `tab:timing_summary` LaTeX from the latest XLSX outputs |

The verifier script `verify_sensitivity_stats.jl` recomputes the headline statistics quoted in §4.6 (CV, peak shifts, phase-transition thresholds) directly from the latest XLSX outputs, to detect drift between the manuscript text and a fresh experimental run.

---

## Requirements

- **Julia** 1.10 or later
- **Ipopt** 3.14 with the **MA27** linear solver (HSL); free MUMPS works but is slower and was not used for the reported timings
- **DRDeed.jl** v0.5.0 (or later) — installed automatically via the `Project.toml` manifest
- A workstation with at least 16 GB RAM for the n = 400 extended-scaling run; smaller experiments fit comfortably in 8 GB

The reported wall-clock numbers were measured on a 13th Gen Intel Core i9-13900H (2.60 GHz) with 16 GB DDR5 RAM.

---

## Quick Start

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("experiments.jl")

# Reproduce every paper experiment (writes to results/):
run_paper_experiments()
```

To run experiments individually:

```julia
include("experiments.jl")

run_drdeed_comparison()         # Experiment 1
run_scalability()               # Experiment 2 + extended scaling
run_ieee30_validation()         # Experiment 3
run_model_progression()         # Experiment 4
run_sensitivity()               # Experiment 5 (5A–5G + tornado)
run_saudi_case_study()          # Experiment 6
run_pareto_analysis()           # Pareto front (5E)
run_vpl_experiment()            # Experiment 7 — see VPL note below
run_metaheuristic_comparison()  # NSGA-II vs Ipopt remark
```

### Experiment 7 (VPL) — local DRDeed.jl required

The valve-point loading variant uses a `vpl=true` flag and multi-start warm-start support added to `DRDeed.jl` for this revision. Until those changes are tagged in a public DRDeed.jl release, run Experiment 7 against a local checkout:

```julia
using Pkg
Pkg.develop(path="../DRDeed.jl")  # path to your local DRDeed.jl clone
Pkg.instantiate()
include("experiments.jl")
run_vpl_experiment()
```

All other experiments (1–6) work directly against the published DRDeed.jl.

---

## Solver Configuration

Ipopt is invoked through JuMP with the following non-default settings:

| Option | Value |
|---|---|
| `tol` | 1e-6 |
| `max_iter` | 3000 |
| `linear_solver` | ma27 |
| `print_level` | 0 (per-experiment override available) |

Each configuration is solved with three independent initializations (default Ipopt + two random perturbations) and the coefficient of variation across runs is reported as a robustness check. For Experiment 7, ten Gaussian-noise warm-starts are used in addition to the smooth-cost baseline.

---

## Results Directory

```
results/
├── drdeed_comparison/    SC1, SC2
├── scalability/          customer × generator grid + extended n
├── ieee30_validation/    SC1, SC2
├── model_progression/    SC1, SC2
├── sensitivity/          5A weights, 5B θ, 5C storage, 5D customers, 5F demand, 5G tariff, tornado
├── saudi_case_study/     ieee30 + progression variants
├── pareto_analysis/      SC1, SC2
├── metaheuristic_comparison/
└── vpl/                  Experiment 7
```

`results/` is gitignored — every output file regenerates from the scripts above.

---

## Citation

When the paper is in print, please cite the published version. In the meantime:

```bibtex
@unpublished{almuraysil2026sgsddeed,
  title  = {A Stackelberg Game for Multi-Objective Demand Response in
            Dynamic Economic Emission Dispatch},
  author = {Almuraysil, Norah and Alshahrani, Mohammed and Belhaiza, Slim},
  note   = {Manuscript IJEPES-D-26-00611, under revision at International
            Journal of Electrical Power \& Energy Systems},
  year   = {2026}
}
```

The optimization package should be cited as:

```bibtex
@software{DRDeed2024,
  title  = {DRDeed.jl: Demand-Response Dynamic Economic Emission Dispatch in Julia},
  author = {Alshahrani, Mohammed},
  year   = {2024},
  url    = {https://github.com/mmogib/DRDeed.jl}
}
```

---

## License

Released under the MIT License. See `LICENSE` for the full text.
