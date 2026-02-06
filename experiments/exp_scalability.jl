"""
    experiment2_scalability.jl — Scalability Analysis

    Tests SGSD-DEED scalability across a grid of customer and generator counts.

    Default grid:
      Customers:  10, 15, 20
      Generators: 4, 6, 8
      Trials: 3 per configuration

    Usage:
      include("experiments.jl")
      run_scalability()
      run_scalability(; customers=10:10:50, generators=4:2:10)
"""

# ─────────────────────────────────────────────────────────────────
# Core experiment function
# ─────────────────────────────────────────────────────────────────

"""
    experiment2_grid(customers, generators, folder; w, trials)

Run SGSD-DEED across a grid of (customers × generators) configurations.
Returns a DataFrame with results.
"""
function experiment2_grid(
    customers::Vector{Int},
    generators::Vector{Int},
    folder::String;
    w::Vector{Float64}=(1 / 3) * ones(3),
    trials::Int=3,
)
    periods = 24
    nc, ng = length(customers), length(generators)
    total_runs = nc * ng * trials

    logmsg("  Grid: $(nc) customer counts × $(ng) generator counts × $(trials) trials = $(total_runs) runs\n", color=:cyan)

    results = Matrix{Float64}(undef, total_runs, 11)
    row = 0

    for c in customers
        for g in generators
            for trial in 1:trials
                row += 1
                results[row, 1:3] = [c, g, periods]

                attempt = 0
                success = false

                while attempt < 5 && !success
                    attempt += 1
                    try
                        timed_sol = @timed begin
                            gtrmmodels = gtdrdeed(c, g, periods)
                            gtrmmodels[:ws](w)
                        end

                        sol = timed_sol.value
                        elapsed = timed_sol.time

                        if isa(sol, SuccessResult)
                            results[row, 4] = sol.solution.deedSolution.Cost
                            results[row, 5] = sol.solution.deedSolution.Emission
                            results[row, 6] = sol.solution.deedSolution.Utility
                            results[row, 7] = sum(sol.solution.deedSolution.Losst)
                            results[row, 8] = sum(sol.data.CDemandt)
                            results[row, 9] = sum(sol.solution.deedSolution.q) + sum(sol.solution.x)
                            results[row, 10] = elapsed
                            results[row, 11] = 1.0
                            success = true
                            logmsg("    c=$c, g=$g, trial=$trial: OK ($(round(elapsed, digits=2))s)\n", color=:cyan)
                        end
                    catch e
                        logmsg("    c=$c, g=$g, trial=$trial, attempt=$attempt: FAILED\n", color=:yellow)
                    end
                end

                if !success
                    results[row, 4:10] .= NaN
                    results[row, 11] = 0.0
                end
            end
        end
    end

    # Create DataFrame
    col_names = [:c, :g, :T, :cost, :emission, :utility, :loss, :demand, :power_generated, :time, :success]
    df = DataFrame(results, col_names)

    # Save to Excel
    mkpath(folder)
    excel_file = outputfilename("solutions"; dated=true, root=folder)
    XLSX.writetable("$excel_file.xlsx", "SOLUTIONS" => df, overwrite=true)
    logmsg("  Saved: $excel_file.xlsx\n", color=:green)

    return df
end

"""
    experiment2_analyze(df, folder)

Analyze experiment 2 results: compute means, create plots.
"""
function experiment2_analyze(df::DataFrame, folder::String)
    # Group by (c, g) and compute statistics
    grouped = combine(
        groupby(df, [:c, :g]),
        :cost => mean => :cost_avg,
        :cost => std => :cost_std,
        :emission => mean => :emission_avg,
        :emission => std => :emission_std,
        :utility => mean => :utility_avg,
        :loss => mean => :loss_avg,
        :demand => mean => :demand_avg,
        :power_generated => mean => :power_generated_avg,
        :time => mean => :time_avg,
    )

    # Add load reduction metric
    transform!(grouped,
        [:demand_avg, :power_generated_avg] =>
            ByRow((d, p) -> 100 * (d - p) / d) => :load_reduction
    )

    # Create plots
    customers = sort(unique(df.c))
    generators = sort(unique(df.g))

    plot_metrics = [
        (:cost_avg, "Cost (\$)"),
        (:emission_avg, "Emission (lb)"),
        (:utility_avg, "Utility (\$)"),
        (:load_reduction, "Load Reduction (%)"),
        (:loss_avg, "Loss (MW)"),
        (:time_avg, "CPU Time (s)"),
    ]

    for (metric, ylabel) in plot_metrics
        experiment2_plot(grouped, customers, generators, metric, ylabel, folder)
    end

    # Save summary table
    summary_df = select(grouped,
        :c => :Customers,
        :g => :Generators,
        :cost_avg => :Cost,
        :emission_avg => :Emission,
        :utility_avg => :Utility,
        :load_reduction => :Load_Reduction_Pct,
        :loss_avg => :Loss,
        :time_avg => :Time_s,
    )

    tex_file = "$folder/results.tex"
    open(tex_file, "w") do f
        show(f, MIME("text/latex"), summary_df)
    end
    logmsg("  Saved: $tex_file\n", color=:green)

    return grouped
end

"""
    experiment2_plot(df, customers, generators, metric, ylabel, folder)

Create a line plot for the given metric.
"""
function experiment2_plot(
    df::DataFrame,
    customers::AbstractVector{<:Real},
    generators::AbstractVector{<:Real},
    metric::Symbol,
    ylabel::String,
    folder::String,
)
    markers = [:circle, :square, :diamond, :utriangle, :star5]

    p = plot(
        xlabel="Number of Customers",
        ylabel=ylabel,
        legend=:outerright,
        size=(700, 450),
    )

    for (i, g) in enumerate(generators)
        subset = filter(r -> r.g == g, df)
        sort!(subset, :c)
        plot!(p, subset.c, subset[!, metric],
            marker=markers[mod1(i, length(markers))],
            markersize=6,
            linewidth=2,
            label="$g generators",
        )
    end

    filename = outputfilename(String(metric); dated=false, root=folder)
    savefig(p, "$(filename).pdf")
    savefig(p, "$(filename).svg")
end

# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

"""
    run_scalability(; customers=10:5:20, generators=4:2:8, trials=3)

Run Experiment 2: Scalability analysis across customer/generator grid.

Results saved to: results/scalability/
"""
function run_scalability(;
    customers::Union{StepRange,Vector{Int}}=10:5:20,
    generators::Union{StepRange,Vector{Int}}=4:2:8,
    trials::Int=3,
)
    logmsg("=" ^ 60 * "\n", color=:blue)
    logmsg("EXPERIMENT 2: Scalability Analysis\n", color=:blue)
    logmsg("=" ^ 60 * "\n", color=:blue)

    folder = "results/scalability"
    customers_vec = collect(customers)
    generators_vec = collect(generators)

    # Run grid
    df = experiment2_grid(customers_vec, generators_vec, folder; trials=trials)

    # Analyze and plot
    logmsg("  Analyzing results...\n", color=:cyan)
    summary = experiment2_analyze(df, folder)

    logmsg("EXPERIMENT 2 COMPLETE\n", color=:green)
    logmsg("Results saved to: $folder/\n", color=:green)

    return (raw=df, summary=summary)
end

# ─────────────────────────────────────────────────────────────────
# Utility: Read existing results
# ─────────────────────────────────────────────────────────────────

"""
    read_experiment2_results(filename, sheet="SOLUTIONS")

Read experiment 2 results from an Excel file.
"""
function read_experiment2_results(filename::String, sheet::String="SOLUTIONS")
    xlsx = XLSX.readxlsx(filename)
    df = DataFrame(XLSX.eachtablerow(xlsx[sheet]))
    return df
end
