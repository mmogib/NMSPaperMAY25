"""
    experiment4_progression.jl — Model Progression: DEED → DR-DEED → SGSD-DEED

    Demonstrates that each extension adds value.
    Uses shared data (DeedData from GTDeedData) for fair comparison.

    Shared helpers (solve_with_retry, extract_metrics_*) are in plotting.jl.
"""

"""
    experiment4_progression(customers, generators, periods, folder, w, name; trials=3)

Run DEED, DR-DEED, and SGSD-DEED with shared data and compare metrics.
Saves XLSX results, grouped bar chart, and LaTeX table.
"""
function experiment4_progression(
  customers::Int,
  generators::Int,
  periods::Int,
  folder::String,
  w::Vector{Float64},
  name::String;
  trials::Int=3,
)
  setup_plot_defaults()
  logmsg("Experiment 4 ($name) — Model Progression\n", color=:blue)
  mkpath(folder)

  # Normalize weights for DEED (2 objectives only: cost, emission)
  w_sum = w[1] + w[2]
  w_deed = w_sum > 0 ? [w[1] / w_sum, w[2] / w_sum] : [0.5, 0.5]

  model_names = ["DEED", "DR-DEED", "SGSD-DEED"]
  metrics = ["Cost", "Emission", "Loss", "Utility", "Time"]

  # Collect results: model → trial → metrics dict
  all_results = Dict(m => Vector{Dict{String,Float64}}() for m in model_names)

  for trial in 1:trials
    logmsg("  Trial $trial/$trials\n", color=:cyan)
    Random.seed!(1000 + trial)

    # Generate shared data once per trial
    gtdata = getGTDRDeedData(customers, generators, periods)
    deedData = gtdata.deedData

    # --- DEED ---
    result, t = solve_with_retry() do
      deed(generators, periods; data=deedData)[:ws](w_deed)
    end
    if !isnothing(result)
      push!(all_results["DEED"], extract_metrics_deed(result, t))
      logmsg("    DEED: Cost=$(round(result.solution.Cost, digits=2))\n", color=:green)
    end

    # --- DR-DEED ---
    result, t = solve_with_retry() do
      drdeed(customers, generators, periods; data=deedData)[:ws](w)
    end
    if !isnothing(result)
      push!(all_results["DR-DEED"], extract_metrics_drdeed(result, t))
      logmsg("    DR-DEED: Cost=$(round(result.solution.Cost, digits=2))\n", color=:green)
    end

    # --- SGSD-DEED ---
    result, t = solve_with_retry() do
      gtdrdeed(customers, generators, periods; data=gtdata)[:ws](w)
    end
    if !isnothing(result)
      push!(all_results["SGSD-DEED"], extract_metrics_gtdeed(result, t))
      logmsg("    SGSD-DEED: Cost=$(round(result.solution.deedSolution.Cost, digits=2))\n", color=:green)
    end
  end

  # --- Aggregate results ---
  summary_rows = []
  for m in model_names
    results_vec = all_results[m]
    if isempty(results_vec)
      push!(summary_rows, (
        Model=m, Cost=NaN, Emission=NaN, Loss=NaN, Utility=NaN, Time=NaN,
        Cost_std=NaN, Emission_std=NaN, Loss_std=NaN, Utility_std=NaN, Time_std=NaN,
      ))
    else
      push!(summary_rows, (
        Model=m,
        Cost=mean(r["Cost"] for r in results_vec),
        Emission=mean(r["Emission"] for r in results_vec),
        Loss=mean(r["Loss"] for r in results_vec),
        Utility=mean(r["Utility"] for r in results_vec),
        Time=mean(r["Time"] for r in results_vec),
        Cost_std=length(results_vec) > 1 ? std([r["Cost"] for r in results_vec]) : 0.0,
        Emission_std=length(results_vec) > 1 ? std([r["Emission"] for r in results_vec]) : 0.0,
        Loss_std=length(results_vec) > 1 ? std([r["Loss"] for r in results_vec]) : 0.0,
        Utility_std=length(results_vec) > 1 ? std([r["Utility"] for r in results_vec]) : 0.0,
        Time_std=length(results_vec) > 1 ? std([r["Time"] for r in results_vec]) : 0.0,
      ))
    end
  end
  summary_df = DataFrame(summary_rows)

  # --- Compute % improvement relative to DEED baseline ---
  deed_row = summary_df[summary_df.Model.=="DEED", :]
  if nrow(deed_row) > 0
    base_cost = deed_row.Cost[1]
    base_emission = deed_row.Emission[1]
    base_loss = deed_row.Loss[1]
    summary_df.Cost_pct = map(eachrow(summary_df)) do r
      r.Model == "DEED" ? 0.0 :
      (isnan(base_cost) || base_cost == 0 ? NaN : 100 * (base_cost - r.Cost) / abs(base_cost))
    end
    summary_df.Emission_pct = map(eachrow(summary_df)) do r
      r.Model == "DEED" ? 0.0 :
      (isnan(base_emission) || base_emission == 0 ? NaN : 100 * (base_emission - r.Emission) / abs(base_emission))
    end
    summary_df.Loss_pct = map(eachrow(summary_df)) do r
      r.Model == "DEED" ? 0.0 :
      (isnan(base_loss) || base_loss == 0 ? NaN : 100 * (base_loss - r.Loss) / abs(base_loss))
    end
  end

  # --- Save to Excel ---
  excel_file = outputfilename("progression"; dated=false, root=folder)
  XLSX.writetable("$(excel_file).xlsx", "SUMMARY" => summary_df, overwrite=true)
  logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

  # --- Grouped bar chart ---
  bar_metrics = ["Cost", "Emission", "Loss"]
  for metric in bar_metrics
    vals = summary_df[!, Symbol(metric)]
    stds = summary_df[!, Symbol("$(metric)_std")]
    p = bar(
      model_names, vals,
      yerr=stds,
      ylabel=metric,
      title="$name — $metric Comparison",
      legend=false,
      fillalpha=0.7,
      color=[:steelblue :darkorange :forestgreen],
      bar_width=0.6,
    )
    save_figure(p, "progression_$(lowercase(metric))"; folder=folder)
  end

  # Utility comparison (only DR-DEED and SGSD-DEED have utility)
  utility_models = ["DR-DEED", "SGSD-DEED"]
  utility_df = filter(r -> r.Model in utility_models, summary_df)
  if nrow(utility_df) > 0
    p_util = bar(
      utility_df.Model, utility_df.Utility,
      yerr=utility_df.Utility_std,
      ylabel="Utility",
      title="$name — Utility Comparison",
      legend=false,
      fillalpha=0.7,
      color=[:darkorange, :forestgreen],
      bar_width=0.5,
    )
    save_figure(p_util, "progression_utility"; folder=folder)
  end

  # --- LaTeX table ---
  display_df = select(summary_df,
    :Model,
    :Cost => ByRow(x -> round(x, digits=2)) => :Cost,
    :Emission => ByRow(x -> round(x, digits=2)) => :Emission,
    :Loss => ByRow(x -> round(x, digits=4)) => :Loss,
    :Utility => ByRow(x -> round(x, digits=2)) => :Utility,
    :Time => ByRow(x -> round(x, digits=3)) => :Time_s,
  )
  tex_file = outputfilename("progression_table"; dated=false, root=folder, extension="tex")
  results_to_latex(display_df, tex_file)
  logmsg("  Saved: $tex_file\n", color=:green)

  return summary_df
end

"""
    run_model_progression()

Run Experiment 4 for both scenarios (SC1 and SC2) with balanced weights.
"""
function run_model_progression()
  w_bc = (1 / 3) * ones(3)

  logmsg("=" ^ 60 * "\n", color=:blue)
  logmsg("EXPERIMENT 4: Model Progression Comparison\n", color=:blue)
  logmsg("=" ^ 60 * "\n", color=:blue)

  sc1 = experiment4_progression(5, 6, 24, "results/model_progression/SC1", w_bc, "SC1 (5c, 6g)")
  sc2 = experiment4_progression(7, 10, 24, "results/model_progression/SC2", w_bc, "SC2 (7c, 10g)")

  return (sc1, sc2)
end
