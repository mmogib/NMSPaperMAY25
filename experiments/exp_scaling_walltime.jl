"""
    exp_scaling_walltime.jl — Log-log wall-clock plot for the extended scaling experiment.

    Reads the most recent solutions.xlsx from `results/scalability/<YYYY_MM_DD>/`,
    aggregates wall-clock by customer count (mean across trials), fits a log-log
    polynomial T_wall(n) ≈ α · n^γ, and writes `scaling_walltime.pdf` into
    `results/scalability/` (alongside the existing `time_avg.pdf`). Stays inside
    `papercode/` per project convention; copy the produced file to
    `papertext/imgs/scaling_walltime.pdf` separately when ready to compile.

    This is post-processing — no new experiment is run. The data must already exist
    (produce it with `run_scalability(customers=[50,100,200,400], generators=[6])`).

    Usage:
      using Pkg; Pkg.activate(".")
      include("experiments.jl")
      plot_scaling_walltime()                  # auto-detect latest dated folder
      plot_scaling_walltime(xlsx_path = "results/scalability/2026_04_25/solutions.xlsx")
      plot_scaling_walltime(output_path = "results/scalability/scaling_walltime.pdf")

    Returns a NamedTuple `(data, α, γ, R2, plot)` for further inspection.
"""

# ─────────────────────────────────────────────────────────────────
# Internal: locate the most recent dated XLSX
# ─────────────────────────────────────────────────────────────────

"""
    _latest_scalability_xlsx(; root="results/scalability") → String

Find the most recent `<YYYY_MM_DD>/solutions.xlsx` under `root`. Errors if none.
"""
function _latest_scalability_xlsx(; root::String="results/scalability")
  isdir(root) || error("Folder not found: $root")
  dated = filter(d -> isdir(joinpath(root, d)) &&
                       occursin(r"^\d{4}_\d{2}_\d{2}$", d),
                 readdir(root))
  isempty(dated) && error("No dated subfolders matching YYYY_MM_DD in $root")
  latest = sort(dated)[end]
  path = joinpath(root, latest, "solutions.xlsx")
  isfile(path) || error("solutions.xlsx not found in $root/$latest")
  return path
end

# ─────────────────────────────────────────────────────────────────
# Main: produce the log-log wall-clock plot with polynomial fit
# ─────────────────────────────────────────────────────────────────

"""
    plot_scaling_walltime(; xlsx_path=nothing,
                           output_path="results/scalability/scaling_walltime.pdf",
                           sheet="SOLUTIONS")

Generate `scaling_walltime.pdf` from the most recent (or specified) scalability run.

Performs log-log linear regression on (n, mean wall-clock) per customer count,
overlays the fit on the empirical points, and saves a publication-ready PDF
within `results/scalability/`. The author then copies the file into
`papertext/imgs/` as a separate, deliberate step before compiling the manuscript.
"""
function plot_scaling_walltime(;
  xlsx_path::Union{String,Nothing}=nothing,
  output_path::String="results/scalability/scaling_walltime.pdf",
  sheet::String="SOLUTIONS",
)
  setup_plot_defaults()

  xlsx_path = something(xlsx_path, _latest_scalability_xlsx())
  logmsg("=" ^ 60 * "\n", color=:blue)
  logmsg("Plotting scaling wall-clock from $xlsx_path\n", color=:blue)
  logmsg("=" ^ 60 * "\n", color=:blue)

  # Load (use the iterator pattern already adopted elsewhere in this repo,
  # see read_experiment2_results in exp_scalability.jl) and filter to
  # converged trials only.
  xlsx_book = XLSX.readxlsx(xlsx_path)
  df = DataFrame(XLSX.eachtablerow(xlsx_book[sheet]))
  valid = filter(r -> !ismissing(r.success) && r.success == 1.0 &&
                       !ismissing(r.time)    && !isnan(r.time),
                  df)
  nrow(valid) == 0 && error("No converged rows in $xlsx_path")

  # Aggregate by customer count
  agg = combine(groupby(valid, :c),
    :time => mean => :time_mean,
    :time => std  => :time_std,
    nrow  => :n_trials,
  )
  for col in names(agg)
    if endswith(col, "_std")
      agg[!, col] = coalesce.(agg[!, col], 0.0)
    end
  end
  sort!(agg, :c)

  ns_data = Float64.(agg.c)
  ts_data = Float64.(agg.time_mean)

  # Log-log linear regression: log T = log α + γ · log n
  X = log10.(ns_data)
  Y = log10.(ts_data)
  X̄, Ȳ = mean(X), mean(Y)
  γ = sum((X .- X̄) .* (Y .- Ȳ)) / sum((X .- X̄) .^ 2)
  α = 10 ^ (Ȳ - γ * X̄)
  ŷ = Ȳ .+ γ .* (X .- X̄)
  R2 = 1 - sum((Y .- ŷ) .^ 2) / sum((Y .- Ȳ) .^ 2)

  logmsg("  Data points (n, T̄):  $(collect(zip(Int.(ns_data), round.(ts_data; digits=2))))\n",
         color=:cyan)
  logmsg("  Fit:  T ≈ $(round(α; sigdigits=3)) · n^$(round(γ; digits=3))    " *
         "(R² = $(round(R2; digits=4)))\n", color=:green)

  # Smooth fit curve over the data range (slightly extended)
  n_min, n_max = minimum(ns_data), maximum(ns_data)
  ns_fit = exp10.(range(log10(n_min * 0.85), log10(n_max * 1.10), length=100))
  ts_fit = α .* ns_fit .^ γ

  # Plot — keep legend labels short to avoid GR backend bbox quirks; put the
  # full fit (with R²) in the title so it's always visible without crowding.
  trials = maximum(agg.n_trials)
  α_str  = string(round(α; sigdigits=2))
  γ_str  = string(round(γ; digits=2))
  R2_str = string(round(R2; digits=4))

  fit_label    = "Fit: T ≈ $(α_str) n^$(γ_str)"
  empirical_lbl = "Mean of $trials trials"
  title_str    = "Wall-clock vs. n  (log-log fit, R² = $(R2_str))"

  p = scatter(ns_data, ts_data,
    xscale = :log10, yscale = :log10,
    title  = title_str,
    xlabel = "Number of customers, n  (log scale)",
    ylabel = "Wall-clock per trial (s, log scale)",
    label  = empirical_lbl,
    marker = :circle, markersize = 7, markerstrokewidth = 1.2,
    color  = :steelblue,
    legend = :topleft,
    foreground_color_legend = nothing,             # avoid GR legend-border overflow
    background_color_legend = RGBA(1, 1, 1, 0.85), # readable but semi-transparent
    size   = (640, 440),
  )
  plot!(p, ns_fit, ts_fit,
    label     = fit_label,
    linewidth = 2, linestyle = :dash, color = :crimson,
  )

  # Save
  mkpath(dirname(output_path))
  savefig(p, output_path)
  logmsg("  Saved: $output_path\n", color=:green)

  return (data=agg, α=α, γ=γ, R2=R2, plot=p)
end
