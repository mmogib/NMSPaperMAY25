"""
    experiment1_comparison.jl — DR-DEED Comparison (SC1 vs SC2)

    Compares SGSD-DEED performance across two scenarios with different weight schemes.

    Scenarios:
      SC1: 5 customers, 6 generators, 24 periods
      SC2: 7 customers, 10 generators, 24 periods

    Weight schemes:
      BC: [1/3, 1/3, 1/3] — balanced
      C2: [1, 0, 0]       — cost only
      C3: [0, 1, 0]       — emission only
      C4: [0, 0, 1]       — utility only

    Usage:
      include("experiments.jl")
      run_drdeed_comparison()
"""

# ─────────────────────────────────────────────────────────────────
# Core experiment function
# ─────────────────────────────────────────────────────────────────

"""
    experiment1_single(customers, generators, periods, folder, w, name)

Run a single SGSD-DEED optimization and save results.
"""
function experiment1_single(
    customers::Int,
    generators::Int,
    periods::Int,
    folder::String,
    w::Vector{Float64},
    name::String,
)
    printstyled("Experiment 1 ($name) started.\n", color=:blue)

    gtrmmodels = gtdrdeed(customers, generators, periods)
    gtr_sol = gtrmmodels[:ws](w)

    excel_file = outputfilename("load_profile"; dated=true, root=folder)
    saveModel(gtr_sol, excel_file)

    # Plot 1: Load profile before/after
    Demand = vec(sum(gtr_sol.data.CDemandt', dims=2))
    qsg = vec(sum(gtr_sol.solution.deedSolution.q', dims=2))
    psg = vec(sum(gtr_sol.solution.x', dims=2))

    p1 = Plots.plot(
        [Demand, qsg + psg],
        linetype=:steppre,
        label=["Initial Load Before SGModel" "Final Load After SGModel"],
        title="Generators Load Profile Before and After SGModel",
        xlabel="Time (hour)",
        ylabel="Generator Power (MW)",
    )
    fig1_file = outputfilename("load_profile"; dated=true, root=folder, extension="pdf")
    Plots.pdf(p1, fig1_file)

    # Plot 2: Power curtailed per customer
    χ = gtr_sol.solution.deedSolution.χ
    p2 = Plots.plot(
        χ',
        linetype=:steppre,
        label=reshape((["customer $i" for i = 1:customers]), 1, customers),
        title="Optimal Power Curtailed ($name)",
        xlabel="Time (hour)",
        ylabel="Power Curtailed (MW)",
    )
    fig2_file = outputfilename("power_curtailed"; dated=true, root=folder, extension="pdf")
    Plots.pdf(p2, fig2_file)

    # Plot 3: Incentives per customer
    ω = gtr_sol.solution.deedSolution.ω
    p3 = Plots.plot(
        ω',
        linetype=:steppre,
        label=reshape((["customer $i incentive" for i = 1:customers]), 1, customers),
        title="Optimal Incentive ($name)",
        xlabel="Time (hour)",
        ylabel="Incentive (\$)",
    )
    fig3_file = outputfilename("incentive"; dated=true, root=folder, extension="pdf")
    Plots.pdf(p3, fig3_file)

    gtr_sol
end

# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

"""
    run_drdeed_comparison()

Run Experiment 1: Compare SGSD-DEED across SC1 and SC2 with 4 weight schemes.

Results saved to:
  results/drdeed_comparison/SC1/{BC,C2,C3,C4}/
  results/drdeed_comparison/SC2/{BC,C2,C3,C4}/
"""
function run_drdeed_comparison()
    logmsg("=" ^ 60 * "\n", color=:blue)
    logmsg("EXPERIMENT 1: DR-DEED Comparison\n", color=:blue)
    logmsg("=" ^ 60 * "\n", color=:blue)

    weight_schemes = [
        ("BC", (1 / 3) * ones(3)),
        ("C2", [1.0, 0.0, 0.0]),
        ("C3", [0.0, 1.0, 0.0]),
        ("C4", [0.0, 0.0, 1.0]),
    ]

    # SC1: 5 customers, 6 generators
    logmsg("  Running SC1 (5 customers, 6 generators)...\n", color=:cyan)
    sc1_results = map(weight_schemes) do (name, w)
        experiment1_single(5, 6, 24, "results/drdeed_comparison/SC1/$name", w, "SC1-$name")
    end

    # SC2: 7 customers, 10 generators
    logmsg("  Running SC2 (7 customers, 10 generators)...\n", color=:cyan)
    sc2_results = map(weight_schemes) do (name, w)
        experiment1_single(7, 10, 24, "results/drdeed_comparison/SC2/$name", w, "SC2-$name")
    end

    # Check success
    sc1_ok = all(r -> isa(r, SuccessResult), sc1_results)
    sc2_ok = all(r -> isa(r, SuccessResult), sc2_results)

    if sc1_ok && sc2_ok
        logmsg("EXPERIMENT 1 COMPLETE\n", color=:green)
    else
        logmsg("EXPERIMENT 1 COMPLETED WITH ERRORS\n", color=:yellow)
    end

    logmsg("Results saved to: results/drdeed_comparison/SC1/, results/drdeed_comparison/SC2/\n", color=:green)

    return (SC1=sc1_results, SC2=sc2_results)
end
