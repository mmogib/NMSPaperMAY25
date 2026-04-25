"""
    timing_summary.jl — Build a wall-clock summary table across all experiments.

    Reads timing columns from each experiment's XLSX outputs (where a Time
    column exists), computes mean/range/total per experiment, and emits both
    a stdout summary and a LaTeX `\\begin{table}` ready to paste into §4 of
    main.tex (immediately before §5 Conclusion).

    Output:
      results/timing_summary.tex   — the LaTeX table
      stdout                       — same content for easy copy/paste

    Usage (from papercode/):
      using Pkg; Pkg.activate(".")
      include("experiments.jl")
      include("timing_summary.jl")
      build_timing_summary()

    For experiments whose XLSX outputs do not include a timing column
    (e.g. drdeed_comparison only saves the load profile), the row is rendered
    with `---` rather than fabricated. Re-run those experiments with timing
    capture enabled if you need their wall-clock here.
"""

# ─────────────────────────────────────────────────────────────────
# Source descriptors: ordered as they appear in the manuscript
# ─────────────────────────────────────────────────────────────────

# Each entry: (label, config_text, xlsx_path, sheet_name, time_column_name)
# A `nothing` xlsx_path means timing wasn't captured for that experiment.
const _TIMING_SOURCES = [
  (label = "Exp.~1 -- DR-DEED comparison",
   config = "SC1+SC2, BC/C2/C3/C4 weights, 3 trials",
   path = nothing, sheet = nothing, col = nothing),

  (label = "Exp.~2 -- Scalability grid",
   config = "\$n\\in[10,20]\$, \$J\\in[4,8]\$, 3 trials",
   path = "results/scalability/2026_02_07/solutions.xlsx",
   sheet = "SOLUTIONS", col = :time),

  (label = "Exp.~3 -- IEEE 30-bus DC-OPF",
   config = "SC1+SC2, BC weights, 3 trials",
   path = nothing, sheet = nothing, col = nothing),

  (label = "Exp.~4 -- Model progression (SC1)",
   config = "DEED \$\\to\$ DR-DEED \$\\to\$ \\newmodel{}, BC, 3 trials",
   path = "results/model_progression/SC1/progression.xlsx",
   sheet = "SUMMARY", col = :Time),

  (label = "Exp.~4 -- Model progression (SC2)",
   config = "DEED \$\\to\$ DR-DEED \$\\to\$ \\newmodel{}, BC, 3 trials",
   path = "results/model_progression/SC2/progression.xlsx",
   sheet = "SUMMARY", col = :Time),

  (label = "Exp.~5A -- Weight sensitivity",
   config = "SC1, 64 simplex points",
   path = "results/sensitivity/weights/weight_sensitivity.xlsx",
   sheet = "WEIGHTS", col = :Time),

  (label = "Exp.~5B -- \$\\theta\$ sensitivity",
   config = "SC1, 10 willingness levels",
   path = "results/sensitivity/theta/theta_sensitivity.xlsx",
   sheet = "THETA", col = :Time),

  (label = "Exp.~5C -- Storage sensitivity",
   config = "SC1, 8 capacity scales",
   path = "results/sensitivity/storage/storage_sensitivity.xlsx",
   sheet = "STORAGE", col = :Time),

  (label = "Exp.~5D -- Customer scaling (\$n\\le 50\$)",
   config = "SC1, \$n=5,\\ldots,50\$, 3 trials each",
   path = "results/sensitivity/customers/customer_sensitivity.xlsx",
   sheet = "CUSTOMERS", col = :Time),

  (label = "Exp.~5D -- Extended scaling",
   config = "SC1-large, \$n=50,100,200,400\$, 3 trials",
   path = "results/scalability/2026_04_25/solutions.xlsx",
   sheet = "SOLUTIONS", col = :time),

  (label = "Exp.~5F -- Demand multiplier",
   config = "SC1, \$\\delta\\in\\{0.8,\\ldots,1.2\\}\$, 3 trials",
   path = "results/sensitivity/demand/demand_sensitivity.xlsx",
   sheet = "DEMAND", col = :Time),

  (label = "Exp.~5G -- Tariff multiplier",
   config = "SC1, \$\\rho\\in\\{0.75,\\ldots,1.25\\}\$, 3 trials",
   path = "results/sensitivity/price/price_sensitivity.xlsx",
   sheet = "PRICE", col = :Time),

  (label = "Exp.~6 -- Saudi case study",
   config = "SC1+SC2, BC weights, 3 trials",
   path = nothing, sheet = nothing, col = nothing),

  (label = "Exp.~7 -- VPL multi-start",
   config = "SC1, 1 smooth + 10 VPL warm starts",
   path = "results/vpl/vpl_runs.xlsx",
   sheet = "VPL_RUNS", col = :Time),
]

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

"""
    _get_time_stats(path, sheet, col) → NamedTuple or nothing

Read the time column from a single XLSX file/sheet. Returns
`(n, mean, min, max, total)` or `nothing` if file/sheet/col is missing.
"""
function _get_time_stats(path, sheet, col)
  (isnothing(path) || isnothing(sheet) || isnothing(col)) && return nothing
  isfile(path) || return nothing
  local df
  try
    df = DataFrame(XLSX.eachtablerow(XLSX.readxlsx(path)[sheet]))
  catch
    return nothing
  end
  sym = Symbol(col)
  hasproperty(df, sym) || return nothing
  ts = Float64[]
  for t in df[!, sym]
    ismissing(t) && continue
    local v
    try
      v = Float64(t)
    catch
      continue
    end
    isnan(v) && continue
    v <= 0 && continue
    push!(ts, v)
  end
  isempty(ts) && return nothing
  return (n = length(ts), mean = mean(ts), min = minimum(ts), max = maximum(ts), total = sum(ts))
end

"""
    _fmt_per_solve(stats) → String

Render the per-solve column. Show a single value if the run-to-run range
is narrow (max < 1.3 × min), otherwise show `min--max`.
"""
function _fmt_per_solve(s)
  if s.max <= 1.3 * s.min || abs(s.max - s.min) < 0.5
    return string(round(s.mean; digits = 2))
  else
    lo = s.min < 10 ? round(s.min; digits = 2) : round(s.min; digits = 1)
    hi = s.max < 10 ? round(s.max; digits = 2) : round(s.max; digits = 1)
    return "$lo--$hi"
  end
end

"""
    _fmt_total(stats) → String

Render the total wall-clock. Seconds for short totals, switches to
minutes for totals over ~1 hour.
"""
function _fmt_total(s)
  if s.total < 600                 # < 10 minutes → seconds
    return string(round(s.total; digits = 1))
  elseif s.total < 7200             # < 2 hours → minutes
    return "$(round(s.total / 60; digits = 1))\\,min"
  else
    return "$(round(s.total / 3600; digits = 2))\\,h"
  end
end

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

"""
    build_timing_summary(; root="results") → String

Build the LaTeX wall-clock summary table; save to `<root>/timing_summary.tex`
and print to stdout. Returns the LaTeX source as a String.
"""
function build_timing_summary(; root::String = "results")
  rows = String[]
  total_runs = 0
  total_wallclock = 0.0
  for src in _TIMING_SOURCES
    s = _get_time_stats(src.path, src.sheet, src.col)
    if isnothing(s)
      push!(rows, "    $(src.label) & $(src.config) & --- & --- & --- \\\\")
      logmsg("  [skip] $(src.label) — no timing data\n", color = :yellow)
    else
      push!(rows, "    $(src.label) & $(src.config) & $(s.n) & $(_fmt_per_solve(s)) & $(_fmt_total(s)) \\\\")
      logmsg("  [ok]   $(src.label) — n=$(s.n), mean=$(round(s.mean; digits=2))s, total=$(round(s.total; digits=1))s\n",
             color = :green)
      total_runs += s.n
      total_wallclock += s.total
    end
  end

  table = """
\\begin{table}[htbp]
\\centering
\\footnotesize
\\caption{Wall-clock summary across all experiments. Workstation: 13th Gen Intel Core i9-13900H \$@\$~2.60\\,GHz, 16\\,GB RAM; Ipopt v3.14 with the MA27 sparse linear solver; Julia 1.10. ``Per-solve'' shows the run-to-run range, or the mean when the range is narrow. ``Total'' is the cumulative wall-clock for that experiment. Per-solve runtimes for Experiments~1, 3, and 6 are reported in the narrative of their respective subsections (\\Cref{subsec:exp1}, \\Cref{subsec:exp3}, \\Cref{subsec:exp6}).}
\\label{tab:timing_summary}
\\setlength{\\tabcolsep}{4pt}
\\begin{tabular}{p{4.6cm}p{5.8cm}rcc}
\\toprule
\\textbf{Experiment} & \\textbf{Configuration} & \\textbf{Runs} & \\textbf{Per-solve (s)} & \\textbf{Total} \\\\
\\midrule
$(join(rows, "\n"))
\\bottomrule
\\end{tabular}
\\end{table}
"""

  outpath = joinpath(root, "timing_summary.tex")
  mkpath(root)
  open(outpath, "w") do io
    write(io, table)
  end

  logmsg("\n  Aggregate (timed runs only): $total_runs runs, $(round(total_wallclock; digits = 1))s total.\n",
         color = :cyan)
  logmsg("  Saved: $outpath\n", color = :green)
  println()
  println("=" ^ 72)
  println("LaTeX table (saved to $outpath; copy from below to paste into main.tex):")
  println("=" ^ 72)
  println()
  print(table)

  return table
end
