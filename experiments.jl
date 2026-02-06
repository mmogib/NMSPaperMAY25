"""
    experiments.jl — Unified Experiment Environment

    Include this file to load all dependencies and experiment functions.
    Nothing is executed automatically — you choose what to run.

    Usage:
      using Pkg; Pkg.activate(".")
      include("experiments.jl")

    To reproduce ALL paper results:
      run_paper_experiments()

    Or run individual experiments:
      run_drdeed_comparison()        # DR-DEED comparison (SC1 vs SC2)
      run_scalability()              # Scalability analysis
      run_ieee30_validation()        # IEEE 30-bus DC-OPF
      run_model_progression()        # DEED → DR-DEED → SGSD-DEED
      run_sensitivity()              # Sensitivity analysis
      run_saudi_case_study()         # Saudi case study
      run_pareto_analysis()          # Pareto front (ε-constraint)
      run_metaheuristic_comparison() # NSGA-II vs Ipopt
"""

# ─────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────

using DRDeed
using JuMP
using Random, Dates, LinearAlgebra, Statistics
using TimeZones
using DataFrames
using Plots, StatsPlots
using XLSX

# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────

include("utils.jl")
include("plotting.jl")

# ─────────────────────────────────────────────────────────────────
# Experiment modules
# ─────────────────────────────────────────────────────────────────

include("experiments/exp_drdeed_comparison.jl")
include("experiments/exp_scalability.jl")
include("experiments/exp_ieee30_validation.jl")
include("experiments/exp_model_progression.jl")
include("experiments/exp_sensitivity.jl")
include("experiments/exp_saudi_case_study.jl")
include("experiments/exp_pareto_analysis.jl")
include("experiments/exp_metaheuristic_comparison.jl")

# ─────────────────────────────────────────────────────────────────
# Run all paper experiments
# ─────────────────────────────────────────────────────────────────

"""
    run_paper_experiments()

Run ALL experiments and generate all plots/data needed for the paper.
A reviewer can call this single function to reproduce all results.

Experiments:
  1. DR-DEED Comparison (SC1/SC2)
  2. Scalability Analysis
  3. IEEE 30-Bus DC-OPF Validation
  4. Model Progression (DEED → DR-DEED → SGSD-DEED)
  5. Sensitivity & Robustness Analysis
  6. Saudi Eastern Province Case Study
  7. Pareto Front Analysis (ε-constraint)
  8. Metaheuristic Comparison (NSGA-II vs Ipopt)

Results are saved to the `results/` directory.
"""
function run_paper_experiments()
    setup_plot_defaults()
    init_log("results/paper_experiments_$(Dates.format(now(), "yyyy_mm_dd_HHMMSS")).log")

    logmsg("\n" * "=" ^ 60 * "\n", color=:magenta)
    logmsg("REPRODUCING ALL PAPER EXPERIMENTS\n", color=:magenta)
    logmsg("=" ^ 60 * "\n\n", color=:magenta)

    results = Dict{Symbol,Any}()

    logmsg("[1/8] DR-DEED Comparison\n", color=:magenta)
    results[:drdeed_comparison] = run_drdeed_comparison()

    logmsg("\n[2/8] Scalability Analysis\n", color=:magenta)
    results[:scalability] = run_scalability()

    logmsg("\n[3/8] IEEE 30-Bus Validation\n", color=:magenta)
    results[:ieee30] = run_ieee30_validation()

    logmsg("\n[4/8] Model Progression\n", color=:magenta)
    results[:progression] = run_model_progression()

    logmsg("\n[5/8] Sensitivity Analysis\n", color=:magenta)
    results[:sensitivity] = run_sensitivity()

    logmsg("\n[6/8] Saudi Case Study\n", color=:magenta)
    results[:saudi] = run_saudi_case_study()

    logmsg("\n[7/8] Pareto Front Analysis\n", color=:magenta)
    results[:pareto] = run_pareto_analysis()

    logmsg("\n[8/8] Metaheuristic Comparison\n", color=:magenta)
    results[:metaheuristic] = run_metaheuristic_comparison()

    logmsg("\n" * "=" ^ 60 * "\n", color=:green)
    logmsg("ALL PAPER EXPERIMENTS COMPLETE\n", color=:green)
    logmsg("=" ^ 60 * "\n", color=:green)
    logmsg("Results saved to:\n", color=:green)
    logmsg("  results/drdeed_comparison/       (DR-DEED Comparison)\n", color=:green)
    logmsg("  results/scalability/             (Scalability)\n", color=:green)
    logmsg("  results/ieee30_validation/       (IEEE 30-Bus)\n", color=:green)
    logmsg("  results/model_progression/       (Model Progression)\n", color=:green)
    logmsg("  results/sensitivity/             (Sensitivity)\n", color=:green)
    logmsg("  results/saudi_case_study/        (Saudi Case Study)\n", color=:green)
    logmsg("  results/pareto_analysis/         (Pareto Analysis)\n", color=:green)
    logmsg("  results/metaheuristic_comparison/ (Metaheuristic Comparison)\n", color=:green)

    close_log()
    return results
end

# ─────────────────────────────────────────────────────────────────
# Ready message
# ─────────────────────────────────────────────────────────────────

println()
println("═" ^ 60)
println(" SGSD-DEED Experiment Environment Loaded")
println("═" ^ 60)
println()
println(" To reproduce ALL paper results:")
println("   run_paper_experiments()")
println()
println(" Or run individual experiments:")
println("   run_drdeed_comparison()        DR-DEED comparison (SC1/SC2)")
println("   run_scalability()              Scalability analysis")
println("   run_ieee30_validation()        IEEE 30-bus DC-OPF")
println("   run_model_progression()        Model progression")
println("   run_sensitivity()              Sensitivity analysis")
println("   run_saudi_case_study()         Saudi case study")
println("   run_pareto_analysis()          Pareto front (ε-constraint)")
println("   run_metaheuristic_comparison() NSGA-II vs Ipopt")
println()
