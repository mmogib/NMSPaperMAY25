"""
    experiment8_metaheuristics.jl — Metaheuristic Comparison (NSGA-II vs Ipopt)

    Compares the SGSD-DEED solution quality and computation time using:
      - NSGA-II (multi-objective evolutionary algorithm)
      - Ipopt (interior-point NLP solver) — our baseline

    This addresses Reviewer 3's concern about weak comparisons — specifically that
    the paper only compares against reference [1] and not recent metaheuristic
    methods like [12] and [13].

    Usage:
      using Pkg; Pkg.activate(".")
      include("main_phase3.jl")  # loads DRDeed, plotting utilities
      include("experiment8_metaheuristics.jl")

      # Run comparison:
      run_metaheuristic_comparison()
"""

using Random
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

# ─────────────────────────────────────────────────────────────────
# Check for Metaheuristics.jl availability
# ─────────────────────────────────────────────────────────────────

const HAS_METAHEURISTICS = try
    # Import Metaheuristics qualified to avoid conflict with JuMP.optimize!
    @eval import Metaheuristics
    true
catch
    @warn "Metaheuristics.jl not found. Install with: Pkg.add(\"Metaheuristics\")"
    false
end

# ─────────────────────────────────────────────────────────────────
# Problem definition for metaheuristics
# ─────────────────────────────────────────────────────────────────

"""
    SGSDProblem

Holds the problem data and dimensions for the SGSD-DEED optimization.
"""
struct SGSDProblem
    gtdata::DRDeed.GTDeedData
    customers::Int
    generators::Int
    periods::Int
    n_vars::Int  # total decision variables
    bounds_lower::Vector{Float64}
    bounds_upper::Vector{Float64}
end

"""
    create_sgsd_problem(customers, generators, periods)

Create the problem structure with variable bounds.
"""
function create_sgsd_problem(customers::Int, generators::Int, periods::Int)
    gtdata = getGTDRDeedData(customers, generators, periods)
    dd = gtdata.deedData

    # Variable layout (flattened):
    # q[j,t]: generators × periods
    # p[i,t]: customers × periods
    # s[i,t]: customers × (periods+1)  (s[i,0] to s[i,T])
    # χ[i,t]: customers × periods
    # ω[i,t]: customers × periods
    # x[i,t]: customers × periods
    # f[i,t]: customers × periods
    # h[i,t]: customers × (periods+1)  (h[i,1] to h[i,T+1])
    # y[i,j,t]: customers × generators × periods

    n_q = generators * periods
    n_p = customers * periods
    n_s = customers * (periods + 1)
    n_χ = customers * periods
    n_ω = customers * periods
    n_x = customers * periods
    n_f = customers * periods
    n_h = customers * (periods + 1)
    n_y = customers * generators * periods

    n_vars = n_q + n_p + n_s + n_χ + n_ω + n_x + n_f + n_h + n_y

    # Build bounds
    lower = zeros(n_vars)
    upper = fill(Inf, n_vars)

    idx = 1

    # q bounds
    for j in 1:generators, t in 1:periods
        lower[idx] = dd.pjmin[j]
        upper[idx] = dd.pjmax[j]
        idx += 1
    end

    # p bounds
    for i in 1:customers, t in 1:periods
        lower[idx] = gtdata.pimin[i]
        upper[idx] = gtdata.pimax[i]
        idx += 1
    end

    # s bounds
    for i in 1:customers, t in 0:periods
        lower[idx] = gtdata.pimin[i]
        upper[idx] = gtdata.pimax[i]
        idx += 1
    end

    # χ bounds (curtailment)
    for i in 1:customers, t in 1:periods
        lower[idx] = 0.0
        upper[idx] = dd.CM[i]  # max curtailable
        idx += 1
    end

    # ω bounds (incentive)
    for i in 1:customers, t in 1:periods
        lower[idx] = 0.0
        upper[idx] = dd.UB / (customers * periods)  # proportional budget
        idx += 1
    end

    # x bounds
    for i in 1:customers, t in 1:periods
        lower[idx] = 0.0
        upper[idx] = gtdata.pimax[i]
        idx += 1
    end

    # f bounds
    for i in 1:customers, t in 1:periods
        lower[idx] = 0.0
        upper[idx] = gtdata.CDemandt[i, t] * 2  # reasonable upper
        idx += 1
    end

    # h bounds
    for i in 1:customers, t in 1:(periods+1)
        lower[idx] = 0.0
        upper[idx] = sum(gtdata.CDemandt[i, :])  # total demand as upper
        idx += 1
    end

    # y bounds
    for i in 1:customers, j in 1:generators, t in 1:periods
        lower[idx] = 0.0
        upper[idx] = dd.pjmax[j]  # can't exceed generator capacity
        idx += 1
    end

    return SGSDProblem(gtdata, customers, generators, periods, n_vars, lower, upper)
end

"""
    decode_variables(x, prob)

Extract individual variable arrays from flattened decision vector.
"""
function decode_variables(x::Vector{Float64}, prob::SGSDProblem)
    c, g, T = prob.customers, prob.generators, prob.periods
    idx = 1

    q = reshape(x[idx:idx+g*T-1], g, T)
    idx += g * T

    p = reshape(x[idx:idx+c*T-1], c, T)
    idx += c * T

    s = reshape(x[idx:idx+c*(T+1)-1], c, T+1)  # s[i, 0:T]
    idx += c * (T + 1)

    χ = reshape(x[idx:idx+c*T-1], c, T)
    idx += c * T

    ω = reshape(x[idx:idx+c*T-1], c, T)
    idx += c * T

    xvar = reshape(x[idx:idx+c*T-1], c, T)
    idx += c * T

    f = reshape(x[idx:idx+c*T-1], c, T)
    idx += c * T

    h = reshape(x[idx:idx+c*(T+1)-1], c, T+1)  # h[i, 1:T+1]
    idx += c * (T + 1)

    y = reshape(x[idx:idx+c*g*T-1], c, g, T)

    return (q=q, p=p, s=s, χ=χ, ω=ω, x=xvar, f=f, h=h, y=y)
end

"""
    evaluate_objectives(x, prob)

Compute the three objectives: (Cost, Emission, -Utility)
Note: Metaheuristics.jl minimizes all objectives, so we negate utility.
Returns (objectives, constraint_violation).
"""
function evaluate_objectives(x::Vector{Float64}, prob::SGSDProblem)
    vars = decode_variables(x, prob)
    gtdata = prob.gtdata
    dd = gtdata.deedData
    c, g, T = prob.customers, prob.generators, prob.periods

    # Compute Cost (generator)
    cost_gen = 0.0
    for j in 1:g, t in 1:T
        cost_gen += dd.a[j] + dd.b[j] * vars.q[j, t] + dd.c[j] * vars.q[j, t]^2
    end

    # Compute Emission
    emission = 0.0
    for j in 1:g, t in 1:T
        emission += dd.e[j] + dd.f[j] * vars.q[j, t] + dd.g[j] * vars.q[j, t]^2
    end

    # Compute Utility (customer)
    # Cost_customer + λ*χ - ω
    cost_cust = 0.0
    for i in 1:c, t in 1:T
        cost_cust += gtdata.adot_it[i, t] +
                     gtdata.ahdot_it[i, t] * vars.h[i, t] +
                     gtdata.bdot_it[i, t] * vars.p[i, t] +
                     gtdata.cdot_it[i, t] * vars.p[i, t]^2 +
                     gtdata.edot_it[i, t] * vars.s[i, t] +
                     gtdata.ddot_it[i, t] * vars.s[i, t]^2 +
                     gtdata.nudot_it[i, t] * vars.x[i, t] +
                     gtdata.afdot_it[i, t] * vars.f[i, t]
    end

    utility_benefit = 0.0
    for i in 1:c, t in 1:T
        utility_benefit += dd.λ[i, t] * vars.χ[i, t] - vars.ω[i, t]
    end
    utility = utility_benefit + cost_cust

    # Compute constraint violations
    violations = Float64[]

    # Loss (for power balance)
    loss = zeros(T)
    for t in 1:T
        for j in 1:g, k in 1:g
            loss[t] += vars.q[j, t] * dd.B[j, k] * vars.q[k, t]
        end
    end

    # Power balance constraints
    for i in 1:c, t in 1:T
        # powerbalance1: f + s[t] - s[t-1] - p + x = sum(y)
        s_prev = t == 1 ? vars.s[i, 1] : vars.s[i, t]  # s[i,0] is s[i,1] in 1-indexed
        lhs = vars.f[i, t] + vars.s[i, t+1] - s_prev - vars.p[i, t] + vars.x[i, t]
        rhs = sum(vars.y[i, j, t] for j in 1:g)
        push!(violations, abs(lhs - rhs))
    end

    for i in 1:c, t in 1:T
        # powerbalance2: CDemand + h[t] = h[t+1] + f
        lhs = gtdata.CDemandt[i, t] + vars.h[i, t]
        rhs = vars.h[i, t+1] + vars.f[i, t]
        push!(violations, abs(lhs - rhs))
    end

    for j in 1:g, t in 1:T
        # powerbalance3: sum(y) = q
        lhs = sum(vars.y[i, j, t] for i in 1:c)
        rhs = vars.q[j, t]
        push!(violations, abs(lhs - rhs))
    end

    for t in 1:T
        # powerbalance4: sum(x) + sum(q) = sum(CDemand) + loss - sum(χ)
        lhs = sum(vars.x[i, t] for i in 1:c) + sum(vars.q[j, t] for j in 1:g)
        rhs = sum(gtdata.CDemandt[i, t] for i in 1:c) + loss[t] - sum(vars.χ[i, t] for i in 1:c)
        push!(violations, abs(lhs - rhs))
    end

    # Individual rationality
    for i in 1:c, t in 1:T
        benefit = vars.ω[i, t] - vars.χ[i, t] * (dd.K1[i] * vars.χ[i, t] + (1 - dd.θ[i]) * dd.K2[i])
        if benefit < 0
            push!(violations, -benefit)
        end
    end

    # Budget constraint
    total_incentive = sum(vars.ω)
    if total_incentive > dd.UB
        push!(violations, total_incentive - dd.UB)
    end

    # Ramp constraints
    for j in 1:g, t in 1:(T-1)
        delta = vars.q[j, t+1] - vars.q[j, t]
        if delta > dd.UR[j]
            push!(violations, delta - dd.UR[j])
        end
        if delta < -dd.DR[j]
            push!(violations, -dd.DR[j] - delta)
        end
    end

    # Storage boundary: s[0] = 0, s[T] = 0
    for i in 1:c
        push!(violations, abs(vars.s[i, 1]))  # s[i,0]
        push!(violations, abs(vars.s[i, T+1]))  # s[i,T]
    end

    # h[1] = 0
    for i in 1:c
        push!(violations, abs(vars.h[i, 1]))
    end

    total_violation = sum(violations)

    # Return objectives: (Cost, Emission, -Utility)
    # We minimize all, so negate utility
    return ([cost_gen, emission, -utility], total_violation)
end

# ─────────────────────────────────────────────────────────────────
# NSGA-II wrapper
# ─────────────────────────────────────────────────────────────────

"""
    run_nsga2(prob; pop_size=100, n_generations=200)

Run NSGA-II on the SGSD-DEED problem.
Returns the Pareto front approximation.
"""
function run_nsga2(prob::SGSDProblem; pop_size::Int=100, n_generations::Int=200)
    if !HAS_METAHEURISTICS
        error("Metaheuristics.jl not available")
    end

    # Define bounds matrix (2 × n_vars): row 1 = lower, row 2 = upper
    bounds = Array{Float64}(undef, 2, prob.n_vars)
    bounds[1, :] = prob.bounds_lower
    bounds[2, :] = prob.bounds_upper

    # Objective function for Metaheuristics.jl
    # Must return tuple (f::Vector, g::Vector, h::Vector) where:
    #   f = objective function values (minimize all)
    #   g = inequality constraints (g ≤ 0 is feasible)
    #   h = equality constraints (h = 0 is feasible)
    function objective(x)
        objs, violation = evaluate_objectives(x, prob)
        # Convert total violation to inequality constraint format (g ≤ 0)
        # violation ≥ 0 always, so g = violation means g ≤ 0 when feasible
        gx = [violation]  # single aggregated inequality constraint
        hx = Float64[]    # no equality constraints (already in violation)
        return objs, gx, hx
    end

    # Run NSGA-II
    # Options must be passed inside the algorithm constructor
    options = Metaheuristics.Options(;
        iterations=n_generations,
        seed=UInt(8001),
        verbose=false,
    )
    # p_cr = crossover probability, p_m = mutation probability (default 1/D)
    algorithm = Metaheuristics.NSGA2(;
        N=pop_size,
        p_cr=0.9,
        p_m=1.0/prob.n_vars,
        options=options,
    )

    result = Metaheuristics.optimize(objective, bounds, algorithm)

    return result
end

# ─────────────────────────────────────────────────────────────────
# Ipopt baseline
# ─────────────────────────────────────────────────────────────────

"""
    run_ipopt_weighted_sum(prob, weights)

Run Ipopt with weighted-sum scalarization.
Returns (Cost, Emission, Utility, Loss, time).
"""
function run_ipopt_weighted_sum(prob::SGSDProblem, weights::Vector{Float64})
    t_start = time()

    result, _ = solve_with_retry(; max_attempts=3) do
        gtdrdeed(prob.customers, prob.generators, prob.periods; data=prob.gtdata)[:ws](weights)
    end

    t_elapsed = time() - t_start

    if isnothing(result)
        return nothing
    end

    sol = result.solution.deedSolution
    return (
        Cost=sol.Cost,
        Emission=sol.Emission,
        Utility=sol.Utility,
        Loss=sol.Losst,
        Time=t_elapsed,
    )
end

"""
    generate_ipopt_pareto(prob; n_points=20)

Generate Pareto front approximation using Ipopt with weighted-sum.
"""
function generate_ipopt_pareto(prob::SGSDProblem; n_points::Int=20)
    points = []

    t_total = 0.0
    for i in 1:n_points
        α = (i - 1) / (n_points - 1)
        # Sweep between cost and emission, with some utility weight
        w = [α, 1 - α, 0.1]
        w = w ./ sum(w)  # normalize

        result = run_ipopt_weighted_sum(prob, w)
        if !isnothing(result)
            push!(points, result)
            t_total += result.Time
        end
    end

    return (points=points, total_time=t_total)
end

# ─────────────────────────────────────────────────────────────────
# Comparison and analysis
# ─────────────────────────────────────────────────────────────────

"""
    compute_hypervolume(pareto_points, ref_point)

Compute hypervolume indicator for 2D Pareto front (Cost, Emission).
Uses the standard sweep-line algorithm for 2D minimization problems.
"""
function compute_hypervolume(points::Vector, ref_point::Tuple{Float64,Float64})
    # Filter points: must have valid (non-NaN) values and be dominated by reference
    valid = filter(points) do p
        !isnan(p.Cost) && !isnan(p.Emission) &&
        p.Cost < ref_point[1] && p.Emission < ref_point[2]
    end

    if isempty(valid)
        return 0.0
    end

    # Sort by Cost (x-axis) ascending
    sorted = sort(valid, by=x -> x.Cost)
    n = length(sorted)

    hv = 0.0
    for i in 1:n
        pt = sorted[i]
        # Width: from this point's x to next point's x (or reference for last point)
        if i < n
            width = sorted[i+1].Cost - pt.Cost
        else
            width = ref_point[1] - pt.Cost
        end
        # Height: from this point's y to reference y
        height = ref_point[2] - pt.Emission
        hv += width * height
    end

    return hv
end

"""
    compare_methods(prob, folder; nsga2_pop=100, nsga2_gen=200, ipopt_points=20)

Run both NSGA-II and Ipopt, compare results.
"""
function compare_methods(
    prob::SGSDProblem,
    folder::String;
    nsga2_pop::Int=100,
    nsga2_gen::Int=200,
    ipopt_points::Int=20,
)
    setup_plot_defaults()
    mkpath(folder)

    logmsg("  Running Ipopt (weighted-sum, $ipopt_points points)...\n", color=:cyan)
    t_ipopt_start = time()
    ipopt_result = generate_ipopt_pareto(prob; n_points=ipopt_points)
    t_ipopt = time() - t_ipopt_start

    ipopt_points_data = ipopt_result.points
    logmsg("    Ipopt: $(length(ipopt_points_data)) points, $(round(t_ipopt, digits=2))s\n", color=:cyan)

    nsga2_points_data = []
    t_nsga2 = NaN

    if HAS_METAHEURISTICS
        logmsg("  Running NSGA-II (pop=$nsga2_pop, gen=$nsga2_gen)...\n", color=:cyan)
        t_nsga2_start = time()

        try
            nsga2_result = run_nsga2(prob; pop_size=nsga2_pop, n_generations=nsga2_gen)
            t_nsga2 = time() - t_nsga2_start

            # Extract Pareto front from NSGA-II result
            pf = Metaheuristics.pareto_front(nsga2_result)
            for row in eachrow(pf)
                push!(nsga2_points_data, (Cost=row[1], Emission=row[2], Utility=-row[3]))
            end

            logmsg("    NSGA-II: $(length(nsga2_points_data)) points, $(round(t_nsga2, digits=2))s\n", color=:cyan)
        catch e
            # Only log error type, not full data structures
            if e isa MethodError
                logmsg("    NSGA-II failed: MethodError for $(e.f) - check API\n", color=:red)
            else
                logmsg("    NSGA-II failed: $(typeof(e))\n", color=:red)
            end
        end
    else
        logmsg("  Skipping NSGA-II (Metaheuristics.jl not installed)\n", color=:yellow)
    end

    # Compute metrics
    hv_ipopt = NaN
    hv_nsga2 = NaN

    if !isempty(ipopt_points_data)
        # Reference point for hypervolume (worst case + margin)
        # Filter out NaN values when computing reference point
        all_points = vcat(ipopt_points_data, nsga2_points_data)
        valid_costs = filter(!isnan, [pt.Cost for pt in all_points])
        valid_emissions = filter(!isnan, [pt.Emission for pt in all_points])

        if !isempty(valid_costs) && !isempty(valid_emissions)
            ref_cost = maximum(valid_costs) * 1.1
            ref_emission = maximum(valid_emissions) * 1.1
            ref_point = (ref_cost, ref_emission)

            hv_ipopt = compute_hypervolume(ipopt_points_data, ref_point)
            hv_nsga2 = isempty(nsga2_points_data) ? NaN : compute_hypervolume(nsga2_points_data, ref_point)

            logmsg("  Hypervolume (Ipopt): $(round(hv_ipopt, digits=2))\n", color=:green)
            if !isnan(hv_nsga2) && hv_nsga2 > 0
                logmsg("  Hypervolume (NSGA-II): $(round(hv_nsga2, digits=2))\n", color=:green)
                logmsg("  Ratio (Ipopt/NSGA-II): $(round(hv_ipopt/hv_nsga2, digits=3))\n", color=:green)
            else
                logmsg("  Hypervolume (NSGA-II): N/A (no valid Pareto points)\n", color=:yellow)
            end
        else
            logmsg("  Hypervolume: N/A (no valid points)\n", color=:yellow)
        end
    end

    # Create comparison table
    comparison_df = DataFrame(
        Method=String[],
        Points=Int[],
        Time_s=Float64[],
        Min_Cost=Float64[],
        Min_Emission=Float64[],
        Max_Utility=Float64[],
        Hypervolume=Float64[],
    )

    if !isempty(ipopt_points_data)
        push!(comparison_df, (
            Method="Ipopt",
            Points=length(ipopt_points_data),
            Time_s=round(t_ipopt, digits=2),
            Min_Cost=minimum(pt.Cost for pt in ipopt_points_data),
            Min_Emission=minimum(pt.Emission for pt in ipopt_points_data),
            Max_Utility=maximum(pt.Utility for pt in ipopt_points_data),
            Hypervolume=hv_ipopt,
        ))
    end

    if !isempty(nsga2_points_data)
        push!(comparison_df, (
            Method="NSGA-II",
            Points=length(nsga2_points_data),
            Time_s=round(t_nsga2, digits=2),
            Min_Cost=minimum(pt.Cost for pt in nsga2_points_data),
            Min_Emission=minimum(pt.Emission for pt in nsga2_points_data),
            Max_Utility=maximum(pt.Utility for pt in nsga2_points_data),
            Hypervolume=hv_nsga2,
        ))
    end

    # Save results
    excel_file = outputfilename("metaheuristic_comparison"; dated=false, root=folder)
    XLSX.writetable("$(excel_file).xlsx",
        "COMPARISON" => comparison_df,
        "IPOPT" => DataFrame(ipopt_points_data),
        overwrite=true,
    )
    if !isempty(nsga2_points_data)
        # Append NSGA-II sheet
        XLSX.openxlsx("$(excel_file).xlsx", mode="rw") do xf
            sheet = XLSX.addsheet!(xf, "NSGA2")
            XLSX.writetable!(sheet, DataFrame(nsga2_points_data))
        end
    end
    logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

    # Create comparison plot
    create_comparison_figure(ipopt_points_data, nsga2_points_data, folder)

    return (ipopt=ipopt_points_data, nsga2=nsga2_points_data, comparison=comparison_df)
end

"""
    create_comparison_figure(ipopt_pts, nsga2_pts, folder)

Create overlay figure comparing Pareto fronts from both methods.
"""
function create_comparison_figure(
    ipopt_pts::Vector,
    nsga2_pts::Vector,
    folder::String,
)
    setup_plot_defaults()

    # Sort by emission for line connection
    ipopt_sorted = sort(ipopt_pts, by=x -> x.Emission)

    p = plot(
        [pt.Emission for pt in ipopt_sorted],
        [pt.Cost for pt in ipopt_sorted],
        marker=:circle,
        markersize=6,
        linewidth=2,
        color=:steelblue,
        label="Ipopt (Interior-Point)",
        xlabel="Total Emission (lb)",
        ylabel="Total Cost (\$)",
        title="Pareto Front Comparison: Ipopt vs NSGA-II",
        size=(700, 500),
        legend=:topright,
    )

    if !isempty(nsga2_pts)
        nsga2_sorted = sort(nsga2_pts, by=x -> x.Emission)
        plot!(p,
            [pt.Emission for pt in nsga2_sorted],
            [pt.Cost for pt in nsga2_sorted],
            marker=:diamond,
            markersize=5,
            linewidth=1.5,
            linestyle=:dash,
            color=:coral,
            label="NSGA-II (Evolutionary)",
        )
    end

    save_figure(p, "pareto_ipopt_vs_nsga2"; folder=folder)

    # Utility comparison
    p2 = plot(
        [pt.Emission for pt in ipopt_sorted],
        [pt.Utility for pt in ipopt_sorted],
        marker=:circle,
        markersize=6,
        linewidth=2,
        color=:steelblue,
        label="Ipopt",
        xlabel="Total Emission (lb)",
        ylabel="Total Utility (\$)",
        title="Utility vs Emission: Ipopt vs NSGA-II",
        size=(700, 500),
        legend=:topright,
    )

    if !isempty(nsga2_pts)
        nsga2_sorted = sort(nsga2_pts, by=x -> x.Emission)
        plot!(p2,
            [pt.Emission for pt in nsga2_sorted],
            [pt.Utility for pt in nsga2_sorted],
            marker=:diamond,
            markersize=5,
            linewidth=1.5,
            linestyle=:dash,
            color=:coral,
            label="NSGA-II",
        )
    end

    save_figure(p2, "utility_ipopt_vs_nsga2"; folder=folder)

    return p
end

# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

"""
    run_metaheuristic_comparison(; scenario=:SC1, nsga2_pop=100, nsga2_gen=200)

Run metaheuristic comparison for specified scenario.
"""
function run_metaheuristic_comparison(;
    scenario::Symbol=:SC1,
    nsga2_pop::Int=100,
    nsga2_gen::Int=200,
    ipopt_points::Int=20,
)
    base_folder = "results/metaheuristic_comparison"
    mkpath(base_folder)

    # Initialize log file
    log_file = "$base_folder/experiment8_$(scenario)_$(Dates.format(now(), "yyyy_mm_dd_HHMMSS")).log"
    init_log(log_file)

    result = nothing
    try
        logmsg("=" ^ 60 * "\n", color=:blue)
        logmsg("EXPERIMENT 8: Metaheuristic Comparison (NSGA-II vs Ipopt)\n", color=:blue)
        logmsg("=" ^ 60 * "\n", color=:blue)

        if scenario == :SC1
            customers, generators = 5, 6
        elseif scenario == :SC2
            customers, generators = 7, 10
        else
            error("Unknown scenario: $scenario. Use :SC1 or :SC2")
        end

        periods = 24
        folder = "$base_folder/$scenario"

        logmsg("  Scenario: $scenario ($customers customers, $generators generators)\n", color=:cyan)
        logmsg("  NSGA-II: pop=$nsga2_pop, generations=$nsga2_gen\n", color=:cyan)
        logmsg("  Ipopt: $ipopt_points weighted-sum points\n", color=:cyan)

        prob = create_sgsd_problem(customers, generators, periods)
        logmsg("  Problem size: $(prob.n_vars) decision variables\n", color=:cyan)

        result = compare_methods(prob, folder;
            nsga2_pop=nsga2_pop,
            nsga2_gen=nsga2_gen,
            ipopt_points=ipopt_points,
        )

        logmsg("\n" * "=" ^ 60 * "\n", color=:green)
        logmsg("EXPERIMENT 8 ($scenario) COMPLETE\n", color=:green)
        logmsg("=" ^ 60 * "\n", color=:green)
        logmsg("Log saved to: $log_file\n", color=:green)
    finally
        close_log()
    end

    return result
end

"""
    run_metaheuristic_comparison_both(; kwargs...)

Run metaheuristic comparison for both SC1 and SC2.
"""
function run_metaheuristic_comparison_both(;
    nsga2_pop::Int=100,
    nsga2_gen::Int=200,
    ipopt_points::Int=20,
)
    base_folder = "results/metaheuristic_comparison"
    mkpath(base_folder)

    # Initialize combined log file
    log_file = "$base_folder/experiment8_all_$(Dates.format(now(), "yyyy_mm_dd_HHMMSS")).log"
    init_log(log_file)

    result = nothing
    try
        logmsg("=" ^ 60 * "\n", color=:blue)
        logmsg("EXPERIMENT 8: Metaheuristic Comparison (Both Scenarios)\n", color=:blue)
        logmsg("=" ^ 60 * "\n", color=:blue)

        # Run SC1 (without separate log - uses the combined log)
        logmsg("\n[1/2] Running SC1...\n", color=:magenta)
        result_sc1 = _run_metaheuristic_core(;
            scenario=:SC1, nsga2_pop, nsga2_gen, ipopt_points, base_folder)

        # Run SC2
        logmsg("\n[2/2] Running SC2...\n", color=:magenta)
        result_sc2 = _run_metaheuristic_core(;
            scenario=:SC2, nsga2_pop, nsga2_gen, ipopt_points, base_folder)

        logmsg("\n" * "=" ^ 60 * "\n", color=:green)
        logmsg("EXPERIMENT 8 (ALL SCENARIOS) COMPLETE\n", color=:green)
        logmsg("=" ^ 60 * "\n", color=:green)
        logmsg("Results saved to: $base_folder/\n", color=:green)
        logmsg("Log saved to: $log_file\n", color=:green)

        result = (SC1=result_sc1, SC2=result_sc2)
    finally
        close_log()
    end

    return result
end

"""
    _run_metaheuristic_core(; scenario, nsga2_pop, nsga2_gen, ipopt_points, base_folder)

Internal function to run experiment 8 without separate logging (for use by both_scenarios).
"""
function _run_metaheuristic_core(;
    scenario::Symbol,
    nsga2_pop::Int,
    nsga2_gen::Int,
    ipopt_points::Int,
    base_folder::String,
)
    if scenario == :SC1
        customers, generators = 5, 6
    elseif scenario == :SC2
        customers, generators = 7, 10
    else
        error("Unknown scenario: $scenario. Use :SC1 or :SC2")
    end

    periods = 24
    folder = "$base_folder/$scenario"

    logmsg("  Scenario: $scenario ($customers customers, $generators generators)\n", color=:cyan)
    logmsg("  NSGA-II: pop=$nsga2_pop, generations=$nsga2_gen\n", color=:cyan)
    logmsg("  Ipopt: $ipopt_points weighted-sum points\n", color=:cyan)

    prob = create_sgsd_problem(customers, generators, periods)
    logmsg("  Problem size: $(prob.n_vars) decision variables\n", color=:cyan)

    result = compare_methods(prob, folder;
        nsga2_pop=nsga2_pop,
        nsga2_gen=nsga2_gen,
        ipopt_points=ipopt_points,
    )

    logmsg("  $scenario complete.\n", color=:green)

    return result
end
