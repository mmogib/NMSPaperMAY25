"""
    verify_sensitivity_stats.jl — Recompute the §4.5 5A–5D headline statistics
    from the most recent sensitivity XLSX outputs and print to stdout. Use to
    detect drift between the manuscript text and the latest run; nothing is
    written to disk.

    Usage:
      using Pkg; Pkg.activate(".")
      include("experiments.jl")             # to load XLSX, DataFrames, Statistics
      include("verify_sensitivity_stats.jl")
      verify_sensitivity_stats()
"""

function _read_sheet(xlsx_path::String, sheet::String)
  isfile(xlsx_path) || error("Not found: $xlsx_path")
  xb = XLSX.readxlsx(xlsx_path)
  return DataFrame(XLSX.eachtablerow(xb[sheet]))
end

function verify_sensitivity_stats(; root::String="results/sensitivity")
  println("=" ^ 70)
  println("Verifying §4.5 5A–5D headline numbers against fresh XLSX outputs")
  println("Root: $root")
  println("=" ^ 70)

  # ----- 5A: Weight Factor Sensitivity -----
  println("\n[5A]  Weight Factor Sensitivity")
  df_w = _read_sheet("$root/weights/weight_sensitivity.xlsx", "WEIGHTS")
  valid_w = filter(r -> !ismissing(r.Cost) && !isnan(r.Cost), df_w)
  println("  n_valid                        = $(nrow(valid_w)) / $(nrow(df_w))")
  cost_range = maximum(valid_w.Cost) - minimum(valid_w.Cost)
  cost_cv    = std(valid_w.Cost) / mean(valid_w.Cost) * 100
  println("  Cost range (max - min)         = \$$(round(cost_range; digits=0))    [paper: \$8,264]")
  println("  Cost CV                        = $(round(cost_cv; digits=2))%        [paper: 1.5%]")
  println("  cor(Cost, Emission)            = $(round(cor(valid_w.Cost, valid_w.Emission); digits=2))           [paper: -0.74]")
  println("  cor(Cost, Utility)             = $(round(cor(valid_w.Cost, valid_w.Utility); digits=2))           [paper: +0.53]")

  # ----- 5B: Theta Sensitivity -----
  println("\n[5B]  Customer Willingness θ Sensitivity")
  df_t = _read_sheet("$root/theta/theta_sensitivity.xlsx", "THETA")
  valid_t = filter(r -> !ismissing(r.Cost) && !isnan(r.Cost), df_t)
  sort!(valid_t, :theta)
  println("  θ values: $(round.(Float64.(valid_t.theta); digits=2))")
  for r in eachrow(valid_t)
    println("    θ=$(round(Float64(r.theta); digits=2)): cost=\$$(round(r.Cost; digits=0)), util=\$$(round(r.Utility; digits=0))")
  end
  c_lo, c_hi = valid_t.Cost[1], valid_t.Cost[end]
  u_lo, u_hi = valid_t.Utility[1], valid_t.Utility[end]
  println("  Cost change θ=$(valid_t.theta[1])→$(valid_t.theta[end]):  \$$(round(c_lo; digits=0)) → \$$(round(c_hi; digits=0))    [paper: \$145,494 → \$141,183]")
  println("  Cost reduction                 = $(round((c_lo - c_hi)/c_lo * 100; digits=1))%       [paper: 3.0%]")
  println("  Utility change                 = $(round((u_hi - u_lo)/u_lo * 100; digits=1))%       [paper: +13.3%]")

  # ----- 5C: Storage Sensitivity -----
  println("\n[5C]  Storage Capacity Sensitivity")
  df_s = _read_sheet("$root/storage/storage_sensitivity.xlsx", "STORAGE")
  valid_s = filter(r -> !ismissing(r.Cost) && !isnan(r.Cost), df_s)
  sort!(valid_s, :scale)
  for r in eachrow(valid_s)
    println("    scale=$(round(Float64(r.scale); digits=2))×:  cost=\$$(round(r.Cost; digits=0)), util=\$$(round(r.Utility; digits=0)), loss=$(round(r.Loss; digits=2))")
  end
  println("  Phase transition expected at 0.5×–0.75× (cost~\$30,480, util~\$5,086, loss=0 below threshold)")

  # ----- 5D: Customer Count Sensitivity -----
  println("\n[5D]  Customer Count Sensitivity")
  df_c = _read_sheet("$root/customers/customer_sensitivity.xlsx", "CUSTOMERS")
  valid_c = filter(r -> !ismissing(r.Cost) && !isnan(r.Cost), df_c)
  agg = combine(groupby(valid_c, :customers),
    :Cost     => mean => :Cost_mean,
    :Cost     => std  => :Cost_std,
    :Emission => mean => :Em_mean,
    :Loss     => mean => :Loss_mean,
    :Utility  => mean => :Util_mean,
    :Time     => mean => :T_mean,
  )
  for col in names(agg); endswith(col, "_std") && (agg[!, col] = coalesce.(agg[!, col], 0.0)); end
  sort!(agg, :customers)
  for r in eachrow(agg)
    println("    n=$(r.customers):  cost=\$$(round(r.Cost_mean; digits=0)) ±$(round(r.Cost_std; digits=0)),  em=$(round(r.Em_mean; digits=0)),  time=$(round(r.T_mean; digits=2))s")
  end
  c_first = agg.Cost_mean[1]; n_first = agg.customers[1]
  c_last  = agg.Cost_mean[end]; n_last = agg.customers[end]
  e_first, e_last = agg.Em_mean[1], agg.Em_mean[end]
  t_first, t_last = agg.T_mean[1], agg.T_mean[end]
  println("  Cost reduction n=$n_first → n=$n_last:  $(round((c_first - c_last)/c_first * 100; digits=1))%      [paper: 15.1%]")
  println("  Emission reduction               = $(round((e_first - e_last)/e_first * 100; digits=1))%      [paper: 23.3%]")
  println("  Time scaling                     = $(round(t_first; digits=2))s → $(round(t_last; digits=2))s     [paper: 1.4s → 12s]")
  println("  Per-customer cost                = \$$(round(c_first/n_first; digits=0)) → \$$(round(c_last/n_last; digits=0))     [paper: \$28,470 → \$2,417]")

  println("\n" * "=" ^ 70)
  println("Done. Compare bracketed [paper:...] values to the recomputed numbers.")
  println("=" ^ 70)
  return nothing
end
