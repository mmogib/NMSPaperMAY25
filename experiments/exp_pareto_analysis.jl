"""
    experiment7_pareto.jl — Pareto Front Generation (ε-Constraint Method)

    Generates a 2D Pareto front for Cost vs Emission using the ε-constraint method.
    This addresses reviewer criticism of "trivial weighted-sum scalarization" by
    showing the actual tradeoff frontier.

    The ε-constraint method:
      - Minimize Cost
      - Subject to: Emission ≤ ε
      - Sweep ε from E_min to E_max

    This produces uniformly distributed Pareto points even on non-convex frontiers.

    Usage:
      using Pkg; Pkg.activate(".")
      include("main_phase3.jl")  # loads DRDeed, plotting utilities
      include("experiment7_pareto.jl")

      # Run Pareto front generation:
      run_pareto_analysis()
"""

using Ipopt
using MathOptInterface
const MOI = MathOptInterface

# ─────────────────────────────────────────────────────────────────
# Core: Build GT-DEED model with ε-constraint on Emission
# ─────────────────────────────────────────────────────────────────

"""
    build_epsilon_model(customers, generators, periods, epsilon; data=nothing)

Build SGSD-DEED model that minimizes Cost subject to Emission ≤ epsilon.
Returns the JuMP model ready for optimization.
"""
function build_epsilon_model(
    customers::Int,
    generators::Int,
    periods::Int,
    epsilon::Float64;
    data::Union{Nothing,DRDeed.GTDeedData}=nothing,
)
    gtdata = isnothing(data) ? getGTDRDeedData(customers, generators, periods) : data
    dd = gtdata.deedData

    solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        MOI.Silent() => true,
        "sb" => "yes",
        "max_iter" => 10_000,
    )

    model = Model(solver)

    # Decision variables (same as gtdrdeed.jl)
    @variable(model, q[1:generators, 1:periods] >= 0)
    @variable(model, p[1:customers, 1:periods] >= 0)
    @variable(model, s[1:customers, 0:periods] >= 0)
    @variable(model, χ[1:customers, 1:periods] >= 0)
    @variable(model, ω[1:customers, 1:periods] >= 0)
    @variable(model, x[1:customers, 1:periods] >= 0)
    @variable(model, f[1:customers, 1:periods] >= 0)
    @variable(model, h[1:customers, 1:periods+1] >= 0)
    @variable(model, y[1:customers, 1:generators, 1:periods] >= 0)

    # Expressions for objectives
    @NLexpression(
        model,
        Ci[i in 1:customers, t in 1:periods],
        gtdata.adot_it[i, t] +
        gtdata.ahdot_it[i, t] * h[i, t] +
        gtdata.bdot_it[i, t] * p[i, t] +
        gtdata.cdot_it[i, t] * p[i, t]^2 +
        gtdata.edot_it[i, t] * s[i, t] +
        gtdata.ddot_it[i, t] * s[i, t]^2 +
        gtdata.nudot_it[i, t] * x[i, t] +
        gtdata.afdot_it[i, t] * f[i, t]
    )
    @NLexpression(
        model,
        Cj[j in 1:generators, t in 1:periods],
        dd.a[j] + dd.b[j] * q[j, t] + dd.c[j] * q[j, t]^2
    )
    @NLexpression(
        model,
        Ej[j in 1:generators, t in 1:periods],
        dd.e[j] + dd.f[j] * q[j, t] + dd.g[j] * q[j, t]^2
    )

    @NLexpression(model, Ccust, sum(Ci[i, t] for i = 1:customers for t = 1:periods))
    @NLexpression(model, C, sum(Cj[j, t] for j = 1:generators for t = 1:periods))
    @NLexpression(model, E, sum(Ej[j, t] for j = 1:generators for t = 1:periods))

    @NLexpression(
        model,
        utility,
        sum(dd.λ[i, t] * χ[i, t] - ω[i, t] for i = 1:customers for t = 1:periods) + Ccust
    )

    # Loss model (B-matrix)
    @NLexpression(
        model,
        losst[t in 1:periods],
        sum(q[j, t] * dd.B[j, k] * q[k, t] for j = 1:generators for k = 1:generators)
    )

    # Power balance constraints
    @constraint(
        model,
        powerbalance1[i in 1:customers, t in 1:periods],
        f[i, t] + s[i, t] - s[i, t-1] - p[i, t] + x[i, t] == sum(y[i, j, t] for j = 1:generators)
    )
    @constraint(
        model,
        powerbalance2[i in 1:customers, t in 1:periods],
        gtdata.CDemandt[i, t] + h[i, t] == h[i, t+1] + f[i, t]
    )
    @constraint(
        model,
        powerbalance3[j in 1:generators, t in 1:periods],
        sum(y[i, j, t] for i = 1:customers) == q[j, t]
    )
    @NLconstraint(
        model,
        powerbalance4[t in 1:periods],
        sum(x[i, t] for i = 1:customers) + sum(q[j, t] for j = 1:generators) ==
        sum(gtdata.CDemandt[i, t] for i = 1:customers) + losst[t] - sum(χ[i, t] for i = 1:customers)
    )

    # Operational limits
    @constraint(model, [t in 1:periods-1], gtdata.pimin .<= s[:, t] .<= gtdata.pimax)
    @constraint(model, [t in 1:periods], gtdata.pimin .<= p[:, t] .<= gtdata.pimax)
    @constraint(model, [t in 1:periods], sum(x[:, t]) .<= gtdata.pimax)
    @constraint(model, [t in 1:periods], dd.pjmin .<= q[:, t] .<= dd.pjmax)
    @constraint(model, [t in 1:(periods-1)], -dd.DR .<= (q[:, t+1] - q[:, t]) .<= dd.UR)

    # Individual rationality (benefit ≥ discomfort)
    @NLconstraint(
        model,
        benefit[i in 1:customers, t in 1:periods],
        ω[i, t] - χ[i, t] * (dd.K1[i] * χ[i, t] + (1 - dd.θ[i]) * dd.K2[i]) >= 0
    )
    # Incentive compatibility
    @NLconstraint(
        model,
        benefit2[i in 2:customers, t in 1:periods],
        ω[i, t] - χ[i, t] * (dd.K1[i] * χ[i, t] + (1 - dd.θ[i]) * dd.K2[i]) >=
        ω[i-1, t] - χ[i-1, t] * (dd.K1[i-1] * χ[i-1, t] + (1 - dd.θ[i-1]) * dd.K2[i-1])
    )

    # Budget and other constraints
    @constraint(model, budget, sum(ω[i, t] for i = 1:customers for t = 1:periods) <= dd.UB)
    @constraint(model, load[i in 1:customers], sum(χ[i, t] for t = 1:periods) <= dd.CM[i])
    @constraint(model, storage1[i in 1:customers], s[i, 0] == 0)
    @constraint(model, storage2[i in 1:customers], s[i, periods] == 0.0)
    @constraint(model, shifted_load[i in 1:customers], h[i, 1] == 0)
    @constraint(model, balance_chi[i in 1:customers, t in 1:periods], χ[i, t] <= h[i, t] + s[i, t])

    # ε-CONSTRAINT: Emission ≤ epsilon
    @NLconstraint(model, emission_bound, E <= epsilon)

    # OBJECTIVE: Minimize Cost only
    @NLobjective(model, Min, C)

    return model, gtdata
end

"""
    solve_epsilon_model(customers, generators, periods, epsilon; data=nothing)

Solve SGSD-DEED with ε-constraint. Returns (Cost, Emission, Utility, Loss) or nothing.
"""
function solve_epsilon_model(
    customers::Int,
    generators::Int,
    periods::Int,
    epsilon::Float64;
    data::Union{Nothing,DRDeed.GTDeedData}=nothing,
)
    model, gtdata = build_epsilon_model(customers, generators, periods, epsilon; data=data)
    optimize!(model)

    if has_values(model)
        return (
            Cost=value(model[:C]),
            Emission=value(model[:E]),
            Utility=value(model[:utility]),
            Loss=sum(value.(model[:losst])),
        )
    else
        return nothing
    end
end

# ─────────────────────────────────────────────────────────────────
# Find emission range via lexicographic optimization
# ─────────────────────────────────────────────────────────────────

"""
    find_emission_range(customers, generators, periods; data=nothing)

Find the range of Pareto-optimal emissions by solving:
  - E_min: Minimize Emission (w = [0, 1, 0])
  - E_max: Minimize Cost (w = [1, 0, 0]), then report emission

Returns (E_min, E_max, C_at_Emin, C_at_Emax).
"""
function find_emission_range(
    customers::Int,
    generators::Int,
    periods::Int;
    data::Union{Nothing,DRDeed.GTDeedData}=nothing,
)
    logmsg("  Finding emission range...\n", color=:cyan)

    gtdata = isnothing(data) ? getGTDRDeedData(customers, generators, periods) : data

    # Minimize emission (maximum emission reduction)
    result_e, _ = solve_with_retry(; max_attempts=3) do
        gtdrdeed(customers, generators, periods; data=gtdata)[:ws]([0.0, 1.0, 0.0])
    end
    E_min = isnothing(result_e) ? NaN : result_e.solution.deedSolution.Emission
    C_at_Emin = isnothing(result_e) ? NaN : result_e.solution.deedSolution.Cost

    # Minimize cost (gives maximum emission on Pareto front)
    result_c, _ = solve_with_retry(; max_attempts=3) do
        gtdrdeed(customers, generators, periods; data=gtdata)[:ws]([1.0, 0.0, 0.0])
    end
    E_max = isnothing(result_c) ? NaN : result_c.solution.deedSolution.Emission
    C_at_Emax = isnothing(result_c) ? NaN : result_c.solution.deedSolution.Cost

    logmsg("    E_min = $(round(E_min, digits=2)) (Cost = $(round(C_at_Emin, digits=2)))\n", color=:cyan)
    logmsg("    E_max = $(round(E_max, digits=2)) (Cost = $(round(C_at_Emax, digits=2)))\n", color=:cyan)

    return (E_min, E_max, C_at_Emin, C_at_Emax, gtdata)
end

# ─────────────────────────────────────────────────────────────────
# Main Pareto front generation
# ─────────────────────────────────────────────────────────────────

"""
    generate_pareto_front(customers, generators, periods, folder; n_points=25)

Generate Pareto front using ε-constraint method with n_points uniformly spaced
epsilon values between E_min and E_max.
"""
function generate_pareto_front(
    customers::Int,
    generators::Int,
    periods::Int,
    folder::String;
    n_points::Int=25,
)
    setup_plot_defaults()
    logmsg("Experiment 7 — Pareto Front (ε-Constraint)\n", color=:blue)
    logmsg("  Configuration: $customers customers, $generators generators, $periods periods\n", color=:cyan)
    logmsg("  Target: $n_points Pareto points\n", color=:cyan)
    mkpath(folder)

    Random.seed!(7001)

    # Step 1: Find emission range
    E_min, E_max, C_at_Emin, C_at_Emax, gtdata = find_emission_range(customers, generators, periods)

    if isnan(E_min) || isnan(E_max)
        logmsg("  ERROR: Could not determine emission range\n", color=:red)
        return nothing
    end

    # Step 2: Generate epsilon grid (slightly inside bounds for numerical stability)
    margin = 0.01 * (E_max - E_min)
    epsilons = range(E_min + margin, E_max - margin, length=n_points)

    logmsg("  Sweeping ε from $(round(minimum(epsilons), digits=2)) to $(round(maximum(epsilons), digits=2))...\n", color=:cyan)

    # Step 3: Solve for each epsilon
    pareto_points = []

    for (idx, eps) in enumerate(epsilons)
        result = solve_epsilon_model(customers, generators, periods, eps; data=gtdata)
        if !isnothing(result)
            push!(pareto_points, (epsilon=eps, result...))
            if idx % 5 == 0
                logmsg("    $idx/$n_points complete\n", color=:cyan)
            end
        else
            logmsg("    Point $idx (ε=$(round(eps, digits=2))): infeasible\n", color=:yellow)
        end
    end

    # Step 4: Add extreme points
    push!(pareto_points, (epsilon=E_max, Cost=C_at_Emax, Emission=E_max, Utility=NaN, Loss=NaN))
    push!(pareto_points, (epsilon=E_min, Cost=C_at_Emin, Emission=E_min, Utility=NaN, Loss=NaN))

    df = DataFrame(pareto_points)
    sort!(df, :Emission)

    # Remove dominated points (keep only Pareto-optimal)
    df = filter_pareto_optimal(df)

    logmsg("  Generated $(nrow(df)) Pareto points\n", color=:green)

    # Step 5: Save data
    excel_file = outputfilename("pareto_front"; dated=false, root=folder)
    XLSX.writetable("$(excel_file).xlsx", "PARETO" => df, overwrite=true)
    logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

    # Step 6: Create Pareto front figure
    create_pareto_figure(df, folder; customers=customers, generators=generators)

    return df
end

"""
    filter_pareto_optimal(df)

Remove dominated points from the Cost-Emission Pareto front.
A point (C1, E1) is dominated if there exists (C2, E2) with C2 ≤ C1 and E2 ≤ E1 and at least one strict.
"""
function filter_pareto_optimal(df::DataFrame)
    n = nrow(df)
    is_pareto = trues(n)

    for i in 1:n
        for j in 1:n
            if i != j
                # j dominates i if Cj ≤ Ci and Ej ≤ Ei with at least one strict
                if df.Cost[j] <= df.Cost[i] && df.Emission[j] <= df.Emission[i]
                    if df.Cost[j] < df.Cost[i] || df.Emission[j] < df.Emission[i]
                        is_pareto[i] = false
                        break
                    end
                end
            end
        end
    end

    return df[is_pareto, :]
end

"""
    create_pareto_figure(df, folder; kwargs...)

Create publication-quality Pareto front figure.
"""
function create_pareto_figure(
    df::DataFrame,
    folder::String;
    customers::Int=5,
    generators::Int=6,
)
    # Sort by emission for proper line connection
    sorted_df = sort(df, :Emission)

    # Main Pareto front plot
    p1 = plot(
        sorted_df.Emission,
        sorted_df.Cost,
        marker=:circle,
        markersize=6,
        linewidth=2,
        color=:steelblue,
        xlabel="Total Emission (lb)",
        ylabel="Total Cost (\$)",
        title="Cost-Emission Pareto Front\n(SGSD-DEED, $customers customers, $generators generators)",
        legend=false,
        grid=true,
        minorgrid=true,
        size=(700, 600),
    )

    # Add extreme point annotations with relative offsets
    min_cost_idx = argmin(sorted_df.Cost)
    min_emit_idx = argmin(sorted_df.Emission)

    # Use relative offsets based on data range
    E_range = maximum(sorted_df.Emission) - minimum(sorted_df.Emission)
    C_range = maximum(sorted_df.Cost) - minimum(sorted_df.Cost)
    e_offset = 0.05 * E_range
    c_offset = 0.05 * C_range

    # Min Cost: at high emission (right), low cost (bottom) → label above-left
    # Min Emission: at low emission (left), high cost (top) → label below-right
    annotate!(p1, [
        (sorted_df.Emission[min_cost_idx] - e_offset, sorted_df.Cost[min_cost_idx] + c_offset,
            text("Min Cost", 8, :right)),
        (sorted_df.Emission[min_emit_idx] + e_offset, sorted_df.Cost[min_emit_idx] - c_offset,
            text("Min Emission", 8, :left)),
    ])

    save_figure(p1, "pareto_front_cost_emission"; folder=folder)

    # Normalized Pareto front (for comparison across scenarios)
    C_range = maximum(sorted_df.Cost) - minimum(sorted_df.Cost)
    E_range = maximum(sorted_df.Emission) - minimum(sorted_df.Emission)

    C_norm = (sorted_df.Cost .- minimum(sorted_df.Cost)) ./ max(C_range, 1e-6)
    E_norm = (sorted_df.Emission .- minimum(sorted_df.Emission)) ./ max(E_range, 1e-6)

    p2 = plot(
        E_norm,
        C_norm,
        marker=:circle,
        markersize=6,
        linewidth=2,
        color=:coral,
        xlabel="Normalized Emission",
        ylabel="Normalized Cost",
        title="Normalized Pareto Front",
        legend=false,
        grid=true,
        xlims=(-0.05, 1.05),
        ylims=(-0.05, 1.05),
        size=(550, 500),
        aspect_ratio=1,
    )

    # Add ideal and nadir point markers
    scatter!(p2, [0], [0], marker=:star, markersize=10, color=:green, label="")
    scatter!(p2, [1], [1], marker=:x, markersize=10, color=:red, label="")
    annotate!(p2, [
        (0.05, -0.08, text("Ideal", 8, :left)),
        (0.95, 1.08, text("Nadir", 8, :right)),
    ])

    save_figure(p2, "pareto_front_normalized"; folder=folder)

    # Trade-off analysis: Marginal Rate of Substitution (MRS)
    if nrow(sorted_df) >= 3
        dC = diff(sorted_df.Cost)
        dE = diff(sorted_df.Emission)
        mrs = abs.(dC ./ max.(abs.(dE), 1e-6))  # |dC/dE|
        mid_E = (sorted_df.Emission[1:end-1] .+ sorted_df.Emission[2:end]) ./ 2

        p3 = plot(
            mid_E,
            mrs,
            marker=:diamond,
            markersize=5,
            linewidth=1.5,
            color=:purple,
            xlabel="Emission (lb)",
            ylabel="|dCost/dEmission| (\$/lb)",
            title="Marginal Rate of Substitution",
            legend=false,
            grid=true,
            size=(600, 400),
        )
        save_figure(p3, "pareto_front_mrs"; folder=folder)
    end

    # Combined figure for paper
    combined = plot(p1, p2, layout=(1, 2), size=(1200, 500))
    save_figure(combined, "pareto_front_combined"; folder=folder)

    return p1
end

# ─────────────────────────────────────────────────────────────────
# Compare weighted-sum vs ε-constraint (optional)
# ─────────────────────────────────────────────────────────────────

"""
    compare_ws_vs_epsilon(customers, generators, periods, folder; n_ws=20, n_eps=25)

Generate Pareto front using both methods and overlay for comparison.
This shows the advantage of ε-constraint for uniform coverage.
"""
function compare_ws_vs_epsilon(
    customers::Int,
    generators::Int,
    periods::Int,
    folder::String;
    n_ws::Int=20,
    n_eps::Int=25,
)
    setup_plot_defaults()
    logmsg("Comparison: Weighted-Sum vs ε-Constraint\n", color=:blue)
    mkpath(folder)

    Random.seed!(7002)
    gtdata = getGTDRDeedData(customers, generators, periods)

    # Weighted-sum points
    ws_points = []
    logmsg("  Generating weighted-sum points...\n", color=:cyan)
    for α in range(0, 1, length=n_ws)
        w = [α, 1 - α, 0.0]  # sweep between cost and emission only
        result, _ = solve_with_retry(; max_attempts=3) do
            gtdrdeed(customers, generators, periods; data=gtdata)[:ws](w)
        end
        if !isnothing(result)
            sol = result.solution.deedSolution
            push!(ws_points, (Cost=sol.Cost, Emission=sol.Emission, Method="Weighted-Sum"))
        end
    end

    # ε-constraint points
    eps_points = []
    logmsg("  Generating ε-constraint points...\n", color=:cyan)
    E_min, E_max, _, _, _ = find_emission_range(customers, generators, periods; data=gtdata)

    if !isnan(E_min) && !isnan(E_max)
        margin = 0.01 * (E_max - E_min)
        for eps in range(E_min + margin, E_max - margin, length=n_eps)
            result = solve_epsilon_model(customers, generators, periods, eps; data=gtdata)
            if !isnothing(result)
                push!(eps_points, (Cost=result.Cost, Emission=result.Emission, Method="Epsilon-Constraint"))
            end
        end
    end

    # Combine and plot
    ws_df = DataFrame(ws_points)
    eps_df = DataFrame(eps_points)

    p = plot(
        ws_df.Emission,
        ws_df.Cost,
        seriestype=:scatter,
        marker=:circle,
        markersize=7,
        color=:steelblue,
        label="Weighted-Sum",
        xlabel="Total Emission (lb)",
        ylabel="Total Cost (\$)",
        title="Pareto Front: Weighted-Sum vs Epsilon-Constraint",
        size=(700, 500),
    )
    plot!(p,
        eps_df.Emission,
        eps_df.Cost,
        seriestype=:scatter,
        marker=:diamond,
        markersize=5,
        color=:coral,
        label="Epsilon-Constraint",
    )

    # Connect ε-constraint points
    sorted_eps = sort(eps_df, :Emission)
    plot!(p, sorted_eps.Emission, sorted_eps.Cost, linewidth=1.5, linestyle=:dash, color=:coral, label="")

    save_figure(p, "pareto_ws_vs_epsilon"; folder=folder)

    # Save data
    all_df = vcat(ws_df, eps_df)
    excel_file = outputfilename("pareto_comparison"; dated=false, root=folder)
    XLSX.writetable("$(excel_file).xlsx", "DATA" => all_df, overwrite=true)

    return (ws=ws_df, eps=eps_df)
end

# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

"""
    run_pareto_analysis(; customers=5, generators=6, n_points=30)

Run Pareto front generation for SC1 configuration.
"""
function run_pareto_analysis(; customers::Int=5, generators::Int=6, n_points::Int=30)
    base_folder = "results/pareto_analysis"

    logmsg("="^60 * "\n", color=:blue)
    logmsg("EXPERIMENT 7: Pareto Front Generation\n", color=:blue)
    logmsg("="^60 * "\n", color=:blue)

    # Main Pareto front
    pareto_df = generate_pareto_front(customers, generators, 24, "$base_folder/SC1"; n_points=n_points)

    # Optional: comparison plot
    compare_ws_vs_epsilon(customers, generators, 24, "$base_folder/comparison")

    logmsg("\n" * "="^60 * "\n", color=:green)
    logmsg("EXPERIMENT 7 COMPLETE\n", color=:green)
    logmsg("="^60 * "\n", color=:green)
    logmsg("Results saved to: $base_folder/\n", color=:green)
    logmsg("  - pareto_front_cost_emission.pdf  (main figure)\n", color=:green)
    logmsg("  - pareto_front_normalized.pdf     (normalized)\n", color=:green)
    logmsg("  - pareto_front_mrs.pdf            (trade-off analysis)\n", color=:green)
    logmsg("  - pareto_ws_vs_epsilon.pdf        (method comparison)\n", color=:green)

    return pareto_df
end

"""
    run_pareto_analysis_both(; n_points=30)

Run Pareto front for both SC1 (5c,6g) and SC2 (7c,10g).
"""
function run_pareto_analysis_both(; n_points::Int=30)
    base_folder = "results/pareto_analysis"

    logmsg("="^60 * "\n", color=:blue)
    logmsg("EXPERIMENT 7: Pareto Fronts for SC1 and SC2\n", color=:blue)
    logmsg("="^60 * "\n", color=:blue)

    # SC1: 5 customers, 6 generators
    pareto_sc1 = generate_pareto_front(5, 6, 24, "$base_folder/SC1"; n_points=n_points)

    # SC2: 7 customers, 10 generators
    pareto_sc2 = generate_pareto_front(7, 10, 24, "$base_folder/SC2"; n_points=n_points)

    # Combined comparison figure
    if !isnothing(pareto_sc1) && !isnothing(pareto_sc2)
        create_scenario_comparison(pareto_sc1, pareto_sc2, base_folder)
    end

    return (SC1=pareto_sc1, SC2=pareto_sc2)
end

"""
    create_scenario_comparison(df1, df2, folder)

Create overlay figure comparing Pareto fronts of SC1 and SC2.
"""
function create_scenario_comparison(df1::DataFrame, df2::DataFrame, folder::String)
    setup_plot_defaults()

    # Sort both
    s1 = sort(df1, :Emission)
    s2 = sort(df2, :Emission)

    p = plot(
        s1.Emission,
        s1.Cost,
        marker=:circle,
        markersize=5,
        linewidth=2,
        color=:steelblue,
        label="SC1 (5c, 6g)",
        xlabel="Total Emission (lb)",
        ylabel="Total Cost (\$)",
        title="Cost-Emission Pareto Fronts: SC1 vs SC2",
        size=(700, 500),
        legend=:topright,
    )
    plot!(p,
        s2.Emission,
        s2.Cost,
        marker=:diamond,
        markersize=5,
        linewidth=2,
        color=:coral,
        label="SC2 (7c, 10g)",
    )

    save_figure(p, "pareto_sc1_vs_sc2"; folder=folder)
    return p
end
