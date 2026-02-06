"""
    experiment6_saudi.jl — Saudi Eastern Province Case Study on IEEE 30-Bus

    Validates SGSD-DEED on a realistic Middle Eastern grid context:
      - Saudi summer demand profile (afternoon AC peak)
      - Gas/oil generators (CCGT, SCGT, oil-fired)
      - SEC TOU tariff pricing (4:1 peak/off-peak ratio)

    Two sub-experiments:
      6A: IEEE 30-bus network effect (with/without DC-OPF)
      6B: Model progression (DEED → DR-DEED → SGSD-DEED)
"""

"""
    experiment6_saudi_ieee30(customers, generators, periods, folder, w, name; trials=3)

Run SGSD-DEED with Saudi data ± IEEE 30-bus DC-OPF constraints.
Compares cost, emission, utility, loss, solve time.
"""
function experiment6_saudi_ieee30(
  customers::Int,
  generators::Int,
  periods::Int,
  folder::String,
  w::Vector{Float64},
  name::String;
  trials::Int=3,
)
  setup_plot_defaults()
  logmsg("Experiment 6A ($name) — Saudi IEEE 30-Bus Benchmark\n", color=:blue)
  mkpath(folder)

  config_names = ["No Network", "IEEE 30-Bus DC-OPF"]
  metrics_keys = ["Cost", "Emission", "Loss", "Utility", "Time"]

  all_results = Dict(c => Vector{Dict{String,Float64}}() for c in config_names)
  branch_flows_all = []

  for trial in 1:trials
    logmsg("  Trial $trial/$trials\n", color=:cyan)
    Random.seed!(3000 + trial)

    # Generate shared Saudi data
    gtdata = getSaudiGTDeedData(customers, generators, periods)
    network = getIEEE30Data(generators, customers)

    # --- Without network constraints ---
    result_no_net, t = solve_with_retry() do
      gtdrdeed(customers, generators, periods; data=gtdata)[:ws](w)
    end
    if !isnothing(result_no_net)
      push!(all_results["No Network"], extract_metrics_gtdeed(result_no_net, t))
      logmsg("    No Network: Cost=$(round(result_no_net.solution.deedSolution.Cost, digits=2))\n", color=:green)
    end

    # --- With IEEE 30-bus DC-OPF ---
    result_dc, t = solve_with_retry() do
      gtdrdeed(customers, generators, periods; data=gtdata, network=network)[:ws](w)
    end
    if !isnothing(result_dc)
      push!(all_results["IEEE 30-Bus DC-OPF"], extract_metrics_gtdeed(result_dc, t))
      logmsg("    DC-OPF: Cost=$(round(result_dc.solution.deedSolution.Cost, digits=2))\n", color=:green)

      # Extract branch flows
      try
        Pf_vals = JuMP.value.(result_dc.model[:Pf])
        theta_vals = JuMP.value.(result_dc.model[:θ])
        push!(branch_flows_all, (Pf=Pf_vals, theta=theta_vals, network=network))
      catch e
        logmsg("    Warning: Could not extract branch flows: $e\n", color=:yellow)
      end
    end
  end

  # --- Aggregate results ---
  summary_rows = []
  for cfg in config_names
    results_vec = all_results[cfg]
    if isempty(results_vec)
      push!(summary_rows, (
        Config=cfg, Cost=NaN, Emission=NaN, Loss=NaN, Utility=NaN, Time=NaN,
        Cost_std=NaN, Emission_std=NaN, Loss_std=NaN, Utility_std=NaN, Time_std=NaN,
      ))
    else
      push!(summary_rows, (
        Config=cfg,
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

  # --- Save to Excel ---
  excel_file = outputfilename("saudi_ieee30_comparison"; dated=false, root=folder)
  XLSX.writetable("$(excel_file).xlsx", "SUMMARY" => summary_df, overwrite=true)
  logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

  # --- Comparison bar charts ---
  for metric in ["Cost", "Emission", "Loss", "Utility", "Time"]
    vals = summary_df[!, Symbol(metric)]
    stds = summary_df[!, Symbol("$(metric)_std")]
    p = bar(
      config_names, vals,
      yerr=stds,
      ylabel=metric,
      title="Saudi $name — $metric: Network Effect",
      legend=false,
      fillalpha=0.7,
      color=[:steelblue :coral],
      bar_width=0.5,
    )
    save_figure(p, "saudi_ieee30_$(lowercase(metric))"; folder=folder)
  end

  # --- Branch flow analysis ---
  if !isempty(branch_flows_all)
    plot_branch_flows(branch_flows_all[end], folder, "Saudi $name")
  end

  # --- LaTeX table ---
  display_df = select(summary_df,
    :Config,
    :Cost => ByRow(x -> round(x, digits=2)) => :Cost,
    :Emission => ByRow(x -> round(x, digits=2)) => :Emission,
    :Loss => ByRow(x -> round(x, digits=4)) => :Loss,
    :Utility => ByRow(x -> round(x, digits=2)) => :Utility,
    :Time => ByRow(x -> round(x, digits=3)) => :Time_s,
  )
  tex_file = outputfilename("saudi_ieee30_table"; dated=false, root=folder, extension="tex")
  results_to_latex(display_df, tex_file)
  logmsg("  Saved: $tex_file\n", color=:green)

  return summary_df
end

"""
    experiment6_saudi_progression(customers, generators, periods, folder, w, name; trials=3)

Run DEED → DR-DEED → SGSD-DEED model progression with Saudi data.
Computes % improvement vs baseline DEED.
"""
function experiment6_saudi_progression(
  customers::Int,
  generators::Int,
  periods::Int,
  folder::String,
  w::Vector{Float64},
  name::String;
  trials::Int=3,
)
  setup_plot_defaults()
  logmsg("Experiment 6B ($name) — Saudi Model Progression\n", color=:blue)
  mkpath(folder)

  # Normalize weights for DEED (2 objectives only: cost, emission)
  w_sum = w[1] + w[2]
  w_deed = w_sum > 0 ? [w[1] / w_sum, w[2] / w_sum] : [0.5, 0.5]

  model_names = ["DEED", "DR-DEED", "SGSD-DEED"]
  metrics = ["Cost", "Emission", "Loss", "Utility", "Time"]

  all_results = Dict(m => Vector{Dict{String,Float64}}() for m in model_names)

  for trial in 1:trials
    logmsg("  Trial $trial/$trials\n", color=:cyan)
    Random.seed!(3100 + trial)

    # Generate shared Saudi data once per trial
    gtdata = getSaudiGTDeedData(customers, generators, periods)
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
  excel_file = outputfilename("saudi_progression"; dated=false, root=folder)
  XLSX.writetable("$(excel_file).xlsx", "SUMMARY" => summary_df, overwrite=true)
  logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

  # --- Grouped bar charts ---
  bar_metrics = ["Cost", "Emission", "Loss"]
  for metric in bar_metrics
    vals = summary_df[!, Symbol(metric)]
    stds = summary_df[!, Symbol("$(metric)_std")]
    p = bar(
      model_names, vals,
      yerr=stds,
      ylabel=metric,
      title="Saudi $name — $metric Comparison",
      legend=false,
      fillalpha=0.7,
      color=[:steelblue :darkorange :forestgreen],
      bar_width=0.6,
    )
    save_figure(p, "saudi_progression_$(lowercase(metric))"; folder=folder)
  end

  # Utility comparison (only DR-DEED and SGSD-DEED have utility)
  utility_models = ["DR-DEED", "SGSD-DEED"]
  utility_df = filter(r -> r.Model in utility_models, summary_df)
  if nrow(utility_df) > 0
    p_util = bar(
      utility_df.Model, utility_df.Utility,
      yerr=utility_df.Utility_std,
      ylabel="Utility",
      title="Saudi $name — Utility Comparison",
      legend=false,
      fillalpha=0.7,
      color=[:darkorange, :forestgreen],
      bar_width=0.5,
    )
    save_figure(p_util, "saudi_progression_utility"; folder=folder)
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
  tex_file = outputfilename("saudi_progression_table"; dated=false, root=folder, extension="tex")
  results_to_latex(display_df, tex_file)
  logmsg("  Saved: $tex_file\n", color=:green)

  return summary_df
end

"""
    run_saudi_case_study()

Run Experiment 6 (Saudi Eastern Province case study) for both scenarios.
  SC1: 5 customers, 6 generators, 24 periods
  SC2: 7 customers, 10 generators, 24 periods
"""
function run_saudi_case_study()
  w_bc = (1 / 3) * ones(3)

  logmsg("=" ^ 60 * "\n", color=:blue)
  logmsg("EXPERIMENT 6: Saudi Eastern Province Case Study\n", color=:blue)
  logmsg("=" ^ 60 * "\n", color=:blue)

  # 6A: IEEE 30-bus network effect with Saudi data
  logmsg("\n[6A] Saudi IEEE 30-Bus Benchmark\n", color=:magenta)
  ieee30_sc1 = experiment6_saudi_ieee30(5, 6, 24, "results/saudi_case_study/ieee30/SC1", w_bc, "SC1 (5c, 6g)")
  ieee30_sc2 = experiment6_saudi_ieee30(7, 10, 24, "results/saudi_case_study/ieee30/SC2", w_bc, "SC2 (7c, 10g)")

  # 6B: Model progression with Saudi data
  logmsg("\n[6B] Saudi Model Progression\n", color=:magenta)
  prog_sc1 = experiment6_saudi_progression(5, 6, 24, "results/saudi_case_study/progression/SC1", w_bc, "SC1 (5c, 6g)")
  prog_sc2 = experiment6_saudi_progression(7, 10, 24, "results/saudi_case_study/progression/SC2", w_bc, "SC2 (7c, 10g)")

  logmsg("\nExperiment 6 complete.\n", color=:green)
  logmsg("Results saved to results/saudi_case_study/\n", color=:green)

  return (ieee30=(sc1=ieee30_sc1, sc2=ieee30_sc2), progression=(sc1=prog_sc1, sc2=prog_sc2))
end
