# CLAUDE.md — Experiment Code

**Parent:** `../CLAUDE.md` (Project Orchestrator)

**Status:** ✅ **ALL EXPERIMENTS COMPLETE** (2026-02-06)

---

## Role

Julia experiment scripts generating all numerical results, figures, and tables. Depends on `DRDeed.jl` v0.5.0.

---

## File Inventory

| File | Purpose |
|------|---------|
| `experiments.jl` | Unified loader for all experiments |
| `experiments/exp_drdeed_comparison.jl` | DR-DEED comparison (SC1/SC2) |
| `experiments/exp_scalability.jl` | Scalability analysis |
| `experiments/exp_ieee30_validation.jl` | IEEE 30-Bus DC-OPF |
| `experiments/exp_model_progression.jl` | Model progression |
| `experiments/exp_sensitivity.jl` | Sensitivity analysis |
| `experiments/exp_saudi_case_study.jl` | Saudi case study |
| `experiments/exp_pareto_analysis.jl` | Pareto front (ε-constraint) |
| `experiments/exp_metaheuristic_comparison.jl` | NSGA-II vs Ipopt |
| `analysis.jl` | Deep data analysis |
| `plotting.jl`, `utils.jl` | Support functions |

---

## How to Run

```julia
using Pkg; Pkg.activate(".")
include("experiments.jl")

# Reproduce ALL paper results (for reviewers):
run_paper_experiments()

# Or run individual experiments:
run_drdeed_comparison()        # DR-DEED comparison
run_scalability()              # Scalability analysis
run_ieee30_validation()        # IEEE 30-bus DC-OPF
run_model_progression()        # DEED → DR-DEED → SGSD-DEED
run_sensitivity()              # Sensitivity analysis
run_saudi_case_study()         # Saudi case study
run_pareto_analysis()          # Pareto front (ε-constraint)
run_metaheuristic_comparison() # NSGA-II vs Ipopt
```

**Julia:** 1.10+ | **Solver:** Ipopt v3.14 with MA27

---

## Results Directory

```
results/
├── drdeed_comparison/SC1/, SC2/    # DR-DEED Comparison
├── scalability/                     # Scalability Analysis
├── ieee30_validation/SC1/, SC2/     # IEEE 30-Bus
├── model_progression/SC1/, SC2/     # Model Progression
├── sensitivity/SC1/, SC2/           # Sensitivity Analysis
├── saudi_case_study/ieee30/, progression/
├── pareto_analysis/SC1/, SC2/       # Pareto Analysis
└── metaheuristic_comparison/SC1/, SC2/
```

**Total:** 120+ files (24 XLSX, 10 TEX, 80+ PDF/SVG)

---

## Experiment Results Summary

| Name | Focus | Key Finding |
|------|-------|-------------|
| DR-DEED Comparison | SC1/SC2 comparison | 50-53% cost reduction |
| Scalability | Customer/generator grid | CV 6-37%, optimal c=10,g=4 |
| IEEE 30-Bus | DC-OPF validation | +4-8% cost from transmission |
| Model Progression | DEED→DR-DEED→SGSD-DEED | 46-55% cost reduction |
| Sensitivity | Weights, θ, storage, customers | CV=1.5%, phase transition at 0.5x |
| Saudi Case Study | Regional validation | 46-48% cost, 68-71% emission |
| Pareto Analysis | ε-constraint method | 28% emission ↓ for 4.5% cost ↑ |
| Metaheuristic | NSGA-II vs Ipopt | Ipopt 48× better hypervolume |

---

## Related Sub-Contexts

- `../DRDeed.jl/CLAUDE.md` — optimization package
- `../papertext/CLAUDE.md` — where figures/tables are used
