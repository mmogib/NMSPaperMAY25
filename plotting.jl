"""
    plotting.jl — Shared publication-quality plotting utilities for Phase 3 experiments.

    Provides: setup_plot_defaults, save_figure, comparison_bar_chart, tornado_diagram

    Plots.jl v1.40.4, StatsPlots.jl v0.15.7
"""

using Plots: mm, px, pct  # Import measure units directly from Plots

# ─────────────────────────────────────────────────────────────────
# File logging: dual output to console (colored) + log file (plain)
# ─────────────────────────────────────────────────────────────────

const LOG_IO = Ref{Union{Nothing,IOStream}}(nothing)

function init_log(filepath::String)
  mkpath(dirname(filepath))
  LOG_IO[] = open(filepath, "w")
  logmsg("Log started: $(Dates.now())\n")
end

function close_log()
  if !isnothing(LOG_IO[])
    logmsg("Log ended: $(Dates.now())\n")
    close(LOG_IO[])
    LOG_IO[] = nothing
  end
end

function logmsg(msg::String; color::Symbol=:default)
  printstyled(msg, color=color)
  if !isnothing(LOG_IO[])
    print(LOG_IO[], msg)
    flush(LOG_IO[])
  end
end

"""Convert metrics Dict{String,Float64} to a NamedTuple for safe splatting."""
metrics_nt(m::Dict{String,Float64}) = (; (Symbol(k) => v for (k, v) in m)...)

function setup_plot_defaults()
  # GR backend built-in fonts: "sans-serif", "times", "helvetica", "courier"
  # "times" is a safe serif option available in GR
  default(;
    fontfamily="times",
    titlefontsize=12,
    guidefontsize=10,
    tickfontsize=8,
    legendfontsize=8,
    linewidth=2,
    markersize=5,
    dpi=300,
    size=(600, 400),
    margin=5mm,
  )
end

"""
    save_figure(p, name; folder, formats)

Save plot `p` to multiple formats (default: PDF).
Uses `outputfilename` from utils.jl for path generation.
"""
function save_figure(p, name::String; folder::String=".", formats=["pdf"], dated::Bool=false)
  mkpath(folder)
  for fmt in formats
    filepath = outputfilename(name; dated=dated, root=folder, extension=fmt)
    savefig(p, filepath)
  end
  p
end

"""
    comparison_bar_chart(categories, values, group_names; kwargs...)

Grouped bar chart for comparing models/scenarios.

- `categories`: Vector of category labels (x-axis)
- `values`: Matrix (rows=categories, cols=groups)
- `group_names`: Vector of group labels (legend)
"""
function comparison_bar_chart(
  categories::Vector{String},
  values::Matrix{<:Number},
  group_names::Vector{String};
  ylabel::String="",
  title::String="",
  kwargs...,
)
  groupedbar(
    categories,
    values;
    bar_position=:dodge,
    label=reshape(group_names, 1, :),
    ylabel=ylabel,
    title=title,
    alpha=0.8,
    bar_width=0.7,
    kwargs...,
  )
end

"""
    tornado_diagram(param_names, low_values, high_values, base_value; kwargs...)

Horizontal tornado diagram for sensitivity analysis.
Parameters are sorted by impact range (most sensitive at top).

- `param_names`: Vector of parameter names
- `low_values`: Metric value when parameter is at low end
- `high_values`: Metric value when parameter is at high end
- `base_value`: Metric value at base case
"""
function tornado_diagram(
  param_names::Vector{String},
  low_values::Vector{Float64},
  high_values::Vector{Float64},
  base_value::Float64;
  xlabel::String="",
  title::String="",
  kwargs...,
)
  n = length(param_names)
  ranges = abs.(high_values .- low_values)
  order = sortperm(ranges)  # ascending: smallest at bottom, largest at top

  sorted_names = param_names[order]
  sorted_low = low_values[order]
  sorted_high = high_values[order]

  p = plot(;
    yticks=(1:n, sorted_names),
    xlabel=xlabel,
    title=title,
    legend=:topright,
    size=(700, max(300, n * 80)),
    left_margin=15mm,
    xlims=(minimum(sorted_low) - 0.05 * abs(base_value), maximum(sorted_high) + 0.05 * abs(base_value)),
    kwargs...,
  )

  bw = 0.3  # half-width of each bar
  for i in 1:n
    plot!(p, Shape([base_value, sorted_low[i], sorted_low[i], base_value],
                   [i - bw, i - bw, i + bw, i + bw]);
      color=:steelblue, alpha=0.7, linewidth=0.5,
      label=(i == n ? "Low" : ""))
    plot!(p, Shape([base_value, sorted_high[i], sorted_high[i], base_value],
                   [i - bw, i - bw, i + bw, i + bw]);
      color=:coral, alpha=0.7, linewidth=0.5,
      label=(i == n ? "High" : ""))
  end

  vline!(p, [base_value], color=:black, linewidth=1.5, linestyle=:dash, label="Base")
  p
end

"""
    metrics_lineplot(x_values, metrics_dict, xlabel; folder, name)

Multi-panel line plot for sensitivity sweeps.
- `x_values`: parameter values on x-axis
- `metrics_dict`: Dict("Metric Name" => Vector of values)
"""
function metrics_lineplot(
  x_values::Vector{<:Number},
  metrics_dict::Dict{String,Vector{Float64}},
  xlabel::String;
  folder::Union{Nothing,String}=nothing,
  name::String="sensitivity",
  title_prefix::String="",
)
  plots_list = []
  for (metric_name, values) in sort(collect(metrics_dict))
    p = plot(
      x_values, values,
      marker=:circle,
      xlabel=xlabel,
      ylabel=metric_name,
      title="$(title_prefix)$(metric_name)",
      legend=false,
    )
    push!(plots_list, p)
  end

  n = length(plots_list)
  ncols = min(n, 2)
  nrows = ceil(Int, n / ncols)
  combined = plot(plots_list..., layout=(nrows, ncols), size=(600 * ncols, 400 * nrows))

  if !isnothing(folder)
    save_figure(combined, name; folder=folder)
  end
  combined
end

"""
    metrics_lineplot_with_errorbars(x_values, means, stds, xlabel; kwargs...)

Line plot with error bars for multi-trial experiments.
"""
function metrics_lineplot_with_errorbars(
  x_values::Vector{<:Number},
  metrics_dict::Dict{String,Tuple{Vector{Float64},Vector{Float64}}},
  xlabel::String;
  folder::Union{Nothing,String}=nothing,
  name::String="sensitivity",
  title_prefix::String="",
)
  plots_list = []
  for (metric_name, (means, stds)) in sort(collect(metrics_dict))
    p = plot(
      x_values, means;
      ribbon=stds,
      fillalpha=0.2,  # ribbon uses fillalpha specifically
      marker=:circle,
      xlabel=xlabel,
      ylabel=metric_name,
      title="$(title_prefix)$(metric_name)",
      legend=false,
    )
    push!(plots_list, p)
  end

  n = length(plots_list)
  ncols = min(n, 2)
  nrows = ceil(Int, n / ncols)
  combined = plot(plots_list..., layout=(nrows, ncols), size=(600 * ncols, 400 * nrows))

  if !isnothing(folder)
    save_figure(combined, name; folder=folder)
  end
  combined
end

"""
    results_to_latex(df, filename)

Write a DataFrame as a LaTeX table to file.
"""
function results_to_latex(df::DataFrame, filename::String)
  open(filename, "w") do f
    show(f, MIME("text/latex"), df)
  end
end

# ─────────────────────────────────────────────────────────────────
# Shared experiment helpers
# ─────────────────────────────────────────────────────────────────

"""
    solve_with_retry(solve_fn; max_attempts=5)

Run `solve_fn()` with retry logic. Returns (result, time) or (nothing, 0.0).
"""
function solve_with_retry(solve_fn; max_attempts::Int=5)
  for attempt in 1:max_attempts
    try
      timed = @timed solve_fn()
      if isa(timed.value, SuccessResult)
        return (timed.value, timed.time)
      else
        logmsg("    Attempt $attempt: solver returned FailResult\n", color=:yellow)
      end
    catch e
      logmsg("    Attempt $attempt failed: $e\n", color=:red)
    end
  end
  return (nothing, 0.0)
end

"""
    extract_metrics_deed(result, time) → Dict

Extract standard metrics from a DEED SuccessResult.
"""
function extract_metrics_deed(result, time::Float64)
  sol = result.solution
  Dict{String,Float64}(
    "Cost" => sol.Cost,
    "Emission" => sol.Emission,
    "Loss" => sum(sol.Loss),
    "Utility" => 0.0,
    "Time" => time,
  )
end

"""
    extract_metrics_drdeed(result, time) → Dict

Extract standard metrics from a DR-DEED SuccessResult.
"""
function extract_metrics_drdeed(result, time::Float64)
  sol = result.solution
  Dict{String,Float64}(
    "Cost" => sol.Cost,
    "Emission" => sol.Emission,
    "Loss" => sum(sol.Losst),
    "Utility" => sol.Utility,
    "Time" => time,
  )
end

"""
    extract_metrics_gtdeed(result, time) → Dict

Extract standard metrics from a GT-DEED (SGSD-DEED) SuccessResult.
"""
function extract_metrics_gtdeed(result, time::Float64)
  sol = result.solution.deedSolution
  Dict{String,Float64}(
    "Cost" => sol.Cost,
    "Emission" => sol.Emission,
    "Loss" => sum(sol.Losst),
    "Utility" => sol.Utility,
    "Time" => time,
  )
end
