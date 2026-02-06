# SGSD-DEED Paper Experiments

[![Status](https://img.shields.io/badge/status-ready%20for%20submission-green.svg)]()
[![Julia](https://img.shields.io/badge/Julia-1.10+-purple.svg)](https://julialang.org/)
[![DRDeed.jl](https://img.shields.io/badge/DRDeed.jl-v0.5.0-blue.svg)](https://github.com/mmogib/DRDeed.jl)

> **A Stackelberg Game for Multi-Objective Demand Response in Dynamic Economic Emission Dispatch**
>
> Authors: Norah Almuraysil, Mohammed Alshahrani, Slim Belhaiza

This repository contains the code for reproducing all experiments in the paper.

## Key Results

| Experiment | Key Finding |
|------------|-------------|
| DR-DEED Comparison | 50-53% cost, 63-72% emission, 78-80% loss reduction vs DR-DEED |
| Model Progression | 46-56% cost, 72-77% emission reduction vs baseline DEED |
| IEEE 30-Bus | 3-5% cost premium for network feasibility |
| Saudi Case Study | 46-47% cost, 68-70% emission reduction (regional validation) |
| Pareto Analysis | 28% emission reduction for only 4.5% cost increase |

## Requirements

- Julia 1.10+
- Ipopt solver (v3.14 with MA27 linear solver recommended)
- DRDeed.jl v0.5.0

## Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Reproducing All Paper Results

To reproduce ALL results, plots, and data for the paper:

```julia
include("experiments.jl")
run_paper_experiments()
```

This runs all 8 experiments and saves results to `results/`.

## Individual Experiments

You can also run experiments individually:

```julia
include("experiments.jl")

run_drdeed_comparison()        # Experiment 1: DR-DEED comparison (SC1/SC2)
run_scalability()              # Experiment 2: Scalability analysis
run_ieee30_validation()        # Experiment 3: IEEE 30-bus DC-OPF validation
run_model_progression()        # Experiment 4: DEED -> DR-DEED -> SGSD-DEED
run_sensitivity()              # Experiment 5: Sensitivity & robustness analysis
run_saudi_case_study()         # Experiment 6: Saudi Eastern Province case study
run_pareto_analysis()          # Experiment 5E: Pareto front (epsilon-constraint)
run_metaheuristic_comparison() # Remark: NSGA-II vs Ipopt comparison
```

## Results Directory

```
results/
├── drdeed_comparison/       # Experiment 1: DR-DEED Comparison
├── scalability/             # Experiment 2: Scalability Analysis
├── ieee30_validation/       # Experiment 3: IEEE 30-Bus Validation
├── model_progression/       # Experiment 4: Model Progression
├── sensitivity/             # Experiment 5: Sensitivity Analysis
├── saudi_case_study/        # Experiment 6: Saudi Case Study
├── pareto_analysis/         # Experiment 5E: Pareto Front Analysis
└── metaheuristic_comparison/ # Remark: Metaheuristic Comparison
```

**Total output:** 120+ files (24 XLSX, 10 TEX, 80+ PDF/SVG figures)

## DRDeed.jl Package

The core optimization models are implemented in [DRDeed.jl](https://github.com/mmogib/DRDeed.jl).

```julia
using Pkg
Pkg.add(url="https://github.com/mmogib/DRDeed.jl.git")
```

## Solver Configuration

The experiments use Ipopt with the following settings:
- KKT tolerance: 1e-6
- Maximum iterations: 3000
- Linear solver: MA27 (recommended for performance)

## Data and Tables

Detailed result tables can be found in the `results/` directory after running experiments.

## Citation

If you use this code, please cite:

```bibtex
@article{almuraysil2026stackelberg,
  title={A Stackelberg Game for Multi-Objective Demand Response in Dynamic Economic Emission Dispatch},
  author={Almuraysil, Norah and Alshahrani, Mohammed and Belhaiza, Slim},
  journal={[Under Review]},
  year={2026}
}
```

## License

MIT License
