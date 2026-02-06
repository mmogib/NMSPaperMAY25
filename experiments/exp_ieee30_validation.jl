"""
    experiment3_ieee30.jl — IEEE 30-Bus Benchmark with DC-OPF

    Compares SGSD-DEED with and without transmission network constraints.
    Addresses Reviewer 3's concern about missing AC/DC power flow.
"""

"""
    experiment3_ieee30(customers, generators, periods, folder, w, name; trials=3)

Run SGSD-DEED with and without IEEE 30-bus network constraints.
Compares cost, emission, utility, loss, solve time.
Extracts branch flows for congestion analysis.
"""
function experiment3_ieee30(
  customers::Int,
  generators::Int,
  periods::Int,
  folder::String,
  w::Vector{Float64},
  name::String;
  trials::Int=3,
)
  setup_plot_defaults()
  logmsg("Experiment 3 ($name) — IEEE 30-Bus Benchmark\n", color=:blue)
  mkpath(folder)

  config_names = ["No Network", "IEEE 30-Bus DC-OPF"]
  metrics_keys = ["Cost", "Emission", "Loss", "Utility", "Time"]

  all_results = Dict(c => Vector{Dict{String,Float64}}() for c in config_names)
  branch_flows_all = []  # store branch flow data from DC-OPF runs

  for trial in 1:trials
    logmsg("  Trial $trial/$trials\n", color=:cyan)
    Random.seed!(2000 + trial)

    # Generate shared data
    gtdata = getGTDRDeedData(customers, generators, periods)
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

      # Extract branch flows (variables are in JuMP model, not solution struct)
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
  excel_file = outputfilename("ieee30_comparison"; dated=false, root=folder)
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
      title="$name — $metric: Network Effect",
      legend=false,
      fillalpha=0.7,
      color=[:steelblue :coral],
      bar_width=0.5,
    )
    save_figure(p, "ieee30_$(lowercase(metric))"; folder=folder)
  end

  # --- Branch flow analysis (if available) ---
  if !isempty(branch_flows_all)
    plot_branch_flows(branch_flows_all[end], folder, name)
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
  tex_file = outputfilename("ieee30_table"; dated=false, root=folder, extension="tex")
  results_to_latex(display_df, tex_file)
  logmsg("  Saved: $tex_file\n", color=:green)

  return summary_df
end

"""
    plot_branch_flows(bf_data, folder, name)

Plot branch flow utilization from DC-OPF results.
Shows how close each branch is to its thermal limit.
"""
function plot_branch_flows(bf_data, folder::String, name::String)
  network = bf_data.network
  Pf = bf_data.Pf
  nbranches = length(network.branches)

  if ndims(Pf) == 2
    # Pf is (branches, periods) — plot average utilization
    periods = size(Pf, 2)
    avg_flow = vec(mean(abs.(Pf), dims=2))
    ratings = [b.rating / network.base_mva for b in network.branches]  # per-unit
    utilization = avg_flow ./ ratings .* 100

    branch_labels = ["$(b.from)-$(b.to)" for b in network.branches]

    p = bar(
      1:nbranches, utilization,
      xlabel="Branch",
      ylabel="Avg. Utilization (%)",
      title="$name — Branch Flow Utilization (DC-OPF)",
      legend=false,
      fillalpha=0.7,
      color=:steelblue,
      xticks=(1:nbranches, branch_labels),
      xrotation=90,
      size=(max(800, nbranches * 20), 450),
      bottom_margin=10mm,
    )
    hline!(p, [100], color=:red, linewidth=2, linestyle=:dash, label="Thermal Limit")
    save_figure(p, "ieee30_branch_utilization"; folder=folder)
  elseif ndims(Pf) == 1
    # Pf is (branches,) — single period or flat
    ratings = [b.rating / network.base_mva for b in network.branches]
    utilization = abs.(Pf) ./ ratings .* 100
    branch_labels = ["$(b.from)-$(b.to)" for b in network.branches]

    p = bar(
      1:nbranches, utilization,
      xlabel="Branch",
      ylabel="Utilization (%)",
      title="$name — Branch Flow Utilization (DC-OPF)",
      legend=false,
      fillalpha=0.7,
      color=:steelblue,
      xticks=(1:nbranches, branch_labels),
      xrotation=90,
      size=(max(800, nbranches * 20), 450),
      bottom_margin=10mm,
    )
    hline!(p, [100], color=:red, linewidth=2, linestyle=:dash, label="Thermal Limit")
    save_figure(p, "ieee30_branch_utilization"; folder=folder)
  end
end

"""
    run_ieee30_validation()

Run Experiment 3 for both scenarios.
"""
function run_ieee30_validation()
  w_bc = (1 / 3) * ones(3)

  logmsg("=" ^ 60 * "\n", color=:blue)
  logmsg("EXPERIMENT 3: IEEE 30-Bus Benchmark\n", color=:blue)
  logmsg("=" ^ 60 * "\n", color=:blue)

  sc1 = experiment3_ieee30(5, 6, 24, "results/ieee30_validation/SC1", w_bc, "SC1 (5c, 6g)")
  sc2 = experiment3_ieee30(7, 10, 24, "results/ieee30_validation/SC2", w_bc, "SC2 (7c, 10g)")

  return (sc1, sc2)
end
