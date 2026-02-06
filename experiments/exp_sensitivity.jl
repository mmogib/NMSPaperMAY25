"""
    experiment5_sensitivity.jl — Sensitivity Analysis

    5A: Weight factors (w1, w2, w3) sweep on simplex
    5B: Customer willingness (θ) sweep
    5C: Storage capacity (pimax) scaling
    5D: Customer count with error bars
    5E: Tornado diagrams aggregating all analyses
"""

# ─────────────────────────────────────────────────────────────────
# Helper: construct modified GTDeedData (immutable structs)
# ─────────────────────────────────────────────────────────────────

"""
    with_theta(gtdata, θ_val)

Construct new GTDeedData with uniform θ for all customers.
"""
function with_theta(gtdata, θ_val::Float64)
  dd = gtdata.deedData
  new_dd = DeedData(
    dd.customers, dd.generators, dd.periods,
    dd.Dt, dd.B, dd.pjmin, dd.pjmax, dd.DR, dd.UR,
    dd.K1, dd.K2, fill(θ_val, dd.customers), dd.CM, dd.UB,
    dd.a, dd.b, dd.c, dd.e, dd.f, dd.g, dd.λ,
  )
  DRDeed.GTDeedData(
    new_dd, gtdata.ahdot_it, gtdata.adot_it, gtdata.bdot_it,
    gtdata.cdot_it, gtdata.ddot_it, gtdata.edot_it, gtdata.nudot_it,
    gtdata.afdot_it, gtdata.CDemandt, gtdata.pimin, gtdata.pimax,
  )
end

"""
    with_pimax_scale(gtdata, scale)

Construct new GTDeedData with scaled pimax (storage capacity).
"""
function with_pimax_scale(gtdata, scale::Float64)
  DRDeed.GTDeedData(
    gtdata.deedData, gtdata.ahdot_it, gtdata.adot_it, gtdata.bdot_it,
    gtdata.cdot_it, gtdata.ddot_it, gtdata.edot_it, gtdata.nudot_it,
    gtdata.afdot_it, gtdata.CDemandt, gtdata.pimin, scale .* gtdata.pimax,
  )
end

# ─────────────────────────────────────────────────────────────────
# 5A: Weight Factor Sensitivity
# ─────────────────────────────────────────────────────────────────

"""
    simplex_grid(step) → Vector{Tuple{Float64,Float64,Float64}}

Generate (w1, w2, w3) grid on the unit simplex with given step size.
"""
function simplex_grid(step::Float64=0.1)
  points = Tuple{Float64,Float64,Float64}[]
  w1_range = 0.0:step:1.0
  for w1 in w1_range
    for w2 in 0.0:step:(1.0-w1)
      w3 = 1.0 - w1 - w2
      if w3 >= -1e-10  # numerical tolerance
        push!(points, (w1, w2, max(0.0, w3)))
      end
    end
  end
  points
end

"""
    sensitivity_weights(customers, generators, periods, folder; step=0.1)

Sweep weights on the simplex. For each (w1,w2,w3), solve SGSD-DEED.
Produces heatmaps colored by cost, emission, utility, loss.
"""
function sensitivity_weights(
  customers::Int,
  generators::Int,
  periods::Int,
  folder::String;
  step::Float64=0.1,
)
  setup_plot_defaults()
  logmsg("Experiment 5A — Weight Sensitivity\n", color=:blue)
  mkpath(folder)

  Random.seed!(5001)
  gtdata = getGTDRDeedData(customers, generators, periods)

  points = simplex_grid(step)
  n = length(points)
  logmsg("  Solving $n weight combinations...\n", color=:cyan)

  results_data = []

  for (idx, (w1, w2, w3)) in enumerate(points)
    w = [w1, w2, w3]
    result, t = solve_with_retry(; max_attempts=3) do
      gtdrdeed(customers, generators, periods; data=gtdata)[:ws](w)
    end
    if !isnothing(result)
      m = extract_metrics_gtdeed(result, t)
      push!(results_data, (; w1=w1, w2=w2, w3=w3, metrics_nt(m)...))
      if idx % 10 == 0
        logmsg("    $idx/$n complete\n", color=:cyan)
      end
    else
      push!(results_data, (w1=w1, w2=w2, w3=w3, Cost=NaN, Emission=NaN, Loss=NaN, Utility=NaN, Time=NaN))
    end
  end

  df = DataFrame(results_data)

  # Save raw data
  excel_file = outputfilename("weight_sensitivity"; dated=false, root=folder)
  XLSX.writetable("$(excel_file).xlsx", "WEIGHTS" => df, overwrite=true)
  logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

  # Heatmaps: w1 (x) vs w2 (y), w3 = 1-w1-w2, colored by metric
  valid = filter(r -> !isnan(r.Cost), df)
  for metric in ["Cost", "Emission", "Loss", "Utility"]
    p = scatter(
      valid.w1, valid.w2,
      zcolor=valid[!, Symbol(metric)],
      markershape=:hexagon,
      markersize=8,
      markerstrokewidth=0,
      xlabel="w1 (Cost weight)",
      ylabel="w2 (Emission weight)",
      title="$metric vs. Weight Factors",
      colorbar_title=metric,
      aspect_ratio=1,
      xlims=(-0.05, 1.05),
      ylims=(-0.05, 1.05),
      size=(550, 500),
    )
    # Draw simplex boundary
    plot!(p, [0, 1, 0, 0], [0, 0, 1, 0], color=:black, linewidth=1, label="")
    save_figure(p, "weight_heatmap_$(lowercase(metric))"; folder=folder)
  end

  return df
end

# ─────────────────────────────────────────────────────────────────
# 5B: Customer Willingness (θ) Sensitivity
# ─────────────────────────────────────────────────────────────────

"""
    sensitivity_theta(customers, generators, periods, folder, w; theta_range=0.1:0.1:1.0)

Sweep uniform θ from 0.1 to 1.0. For each value, construct new data and solve.
"""
function sensitivity_theta(
  customers::Int,
  generators::Int,
  periods::Int,
  folder::String,
  w::Vector{Float64};
  theta_range=0.1:0.1:1.0,
)
  setup_plot_defaults()
  logmsg("Experiment 5B — Theta Sensitivity\n", color=:blue)
  mkpath(folder)

  Random.seed!(5002)
  base_gtdata = getGTDRDeedData(customers, generators, periods)

  results_data = []

  for θ_val in theta_range
    logmsg("  θ = $θ_val\n", color=:cyan)
    modified = with_theta(base_gtdata, θ_val)

    result, t = solve_with_retry(; max_attempts=5) do
      gtdrdeed(customers, generators, periods; data=modified)[:ws](w)
    end
    if !isnothing(result)
      m = extract_metrics_gtdeed(result, t)
      push!(results_data, (; theta=θ_val, metrics_nt(m)...))
    else
      push!(results_data, (theta=θ_val, Cost=NaN, Emission=NaN, Loss=NaN, Utility=NaN, Time=NaN))
    end
  end

  df = DataFrame(results_data)

  # Save raw data
  excel_file = outputfilename("theta_sensitivity"; dated=false, root=folder)
  XLSX.writetable("$(excel_file).xlsx", "THETA" => df, overwrite=true)
  logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

  # Multi-panel line plot
  valid = filter(r -> !isnan(r.Cost), df)
  if nrow(valid) > 0
    metrics_dict = Dict{String,Vector{Float64}}(
      "Cost" => valid.Cost,
      "Emission" => valid.Emission,
      "Loss" => valid.Loss,
      "Utility" => valid.Utility,
    )
    metrics_lineplot(
      Float64.(valid.theta), metrics_dict, "Customer Willingness (θ)";
      folder=folder, name="theta_sensitivity", title_prefix="θ Sensitivity — ",
    )
  end

  return df
end

# ─────────────────────────────────────────────────────────────────
# 5C: Storage Capacity Sensitivity
# ─────────────────────────────────────────────────────────────────

"""
    sensitivity_storage(customers, generators, periods, folder, w;
                        scales=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])

Scale pimax by given factors. For each, construct new data and solve.
"""
function sensitivity_storage(
  customers::Int,
  generators::Int,
  periods::Int,
  folder::String,
  w::Vector{Float64};
  scales::Vector{Float64}=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
)
  setup_plot_defaults()
  logmsg("Experiment 5C — Storage Capacity Sensitivity\n", color=:blue)
  mkpath(folder)

  Random.seed!(5003)
  base_gtdata = getGTDRDeedData(customers, generators, periods)

  results_data = []

  for scale in scales
    logmsg("  Storage scale = $(scale)x\n", color=:cyan)
    modified = with_pimax_scale(base_gtdata, scale)

    result, t = solve_with_retry(; max_attempts=5) do
      gtdrdeed(customers, generators, periods; data=modified)[:ws](w)
    end
    if !isnothing(result)
      m = extract_metrics_gtdeed(result, t)
      push!(results_data, (; scale=scale, metrics_nt(m)...))
    else
      push!(results_data, (scale=scale, Cost=NaN, Emission=NaN, Loss=NaN, Utility=NaN, Time=NaN))
    end
  end

  df = DataFrame(results_data)

  # Save raw data
  excel_file = outputfilename("storage_sensitivity"; dated=false, root=folder)
  XLSX.writetable("$(excel_file).xlsx", "STORAGE" => df, overwrite=true)
  logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

  # Multi-panel line plot
  valid = filter(r -> !isnan(r.Cost), df)
  if nrow(valid) > 0
    metrics_dict = Dict{String,Vector{Float64}}(
      "Cost" => valid.Cost,
      "Emission" => valid.Emission,
      "Loss" => valid.Loss,
      "Utility" => valid.Utility,
    )
    metrics_lineplot(
      Float64.(valid.scale), metrics_dict, "Storage Capacity Scale Factor";
      folder=folder, name="storage_sensitivity", title_prefix="Storage Sensitivity — ",
    )
  end

  return df
end

# ─────────────────────────────────────────────────────────────────
# 5D: Customer Count Sensitivity (with error bars)
# ─────────────────────────────────────────────────────────────────

"""
    sensitivity_customers(generators, folder, w;
                          customer_range=5:5:50, trials=3)

Sweep customer count with multiple trials. Track metrics + solve time.
"""
function sensitivity_customers(
  generators::Int,
  folder::String,
  w::Vector{Float64};
  customer_range=5:5:50,
  trials::Int=3,
)
  setup_plot_defaults()
  logmsg("Experiment 5D — Customer Count Sensitivity\n", color=:blue)
  mkpath(folder)

  periods = 24
  results_data = []

  for c in customer_range
    logmsg("  Customers = $c\n", color=:cyan)
    for trial in 1:trials
      Random.seed!(5004 + c * 100 + trial)

      result, t = solve_with_retry(; max_attempts=5) do
        gtdrdeed(c, generators, periods)[:ws](w)
      end
      if !isnothing(result)
        m = extract_metrics_gtdeed(result, t)
        push!(results_data, (; customers=c, trial=trial, metrics_nt(m)...))
      else
        push!(results_data, (customers=c, trial=trial, Cost=NaN, Emission=NaN, Loss=NaN, Utility=NaN, Time=NaN))
      end
    end
  end

  df = DataFrame(results_data)

  # Save raw data
  excel_file = outputfilename("customer_sensitivity"; dated=false, root=folder)
  XLSX.writetable("$(excel_file).xlsx", "CUSTOMERS" => df, overwrite=true)
  logmsg("  Saved: $(excel_file).xlsx\n", color=:green)

  # Aggregate by customer count
  valid = filter(r -> !isnan(r.Cost), df)
  if nrow(valid) > 0
    agg = combine(
      groupby(valid, :customers),
      :Cost => mean => :Cost_mean, :Cost => std => :Cost_std,
      :Emission => mean => :Emission_mean, :Emission => std => :Emission_std,
      :Loss => mean => :Loss_mean, :Loss => std => :Loss_std,
      :Utility => mean => :Utility_mean, :Utility => std => :Utility_std,
      :Time => mean => :Time_mean, :Time => std => :Time_std,
    )

    # Replace missing std (single trial) with 0
    for col in names(agg)
      if endswith(col, "_std")
        agg[!, col] = coalesce.(agg[!, col], 0.0)
      end
    end

    # Line plot with error bars
    metrics_err = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}(
      "Cost" => (agg.Cost_mean, agg.Cost_std),
      "Emission" => (agg.Emission_mean, agg.Emission_std),
      "Loss" => (agg.Loss_mean, agg.Loss_std),
      "Utility" => (agg.Utility_mean, agg.Utility_std),
      "Time (s)" => (agg.Time_mean, agg.Time_std),
    )
    metrics_lineplot_with_errorbars(
      Float64.(agg.customers), metrics_err, "Number of Customers";
      folder=folder, name="customer_sensitivity", title_prefix="Customer Sensitivity — ",
    )

    # Save LaTeX table
    display_df = select(agg,
      :customers => :Customers,
      :Cost_mean => ByRow(x -> round(x, digits=2)) => :Cost,
      :Emission_mean => ByRow(x -> round(x, digits=2)) => :Emission,
      :Loss_mean => ByRow(x -> round(x, digits=4)) => :Loss,
      :Utility_mean => ByRow(x -> round(x, digits=2)) => :Utility,
      :Time_mean => ByRow(x -> round(x, digits=3)) => :Time_s,
    )
    tex_file = outputfilename("customer_table"; dated=false, root=folder, extension="tex")
    results_to_latex(display_df, tex_file)
  end

  return df
end

# ─────────────────────────────────────────────────────────────────
# 5E: Tornado Diagrams
# ─────────────────────────────────────────────────────────────────

"""
    generate_tornado_diagrams(theta_df, storage_df, weight_df, customer_df, folder;
                              base_theta=0.5, base_scale=1.0)

Generate tornado diagrams showing parameter sensitivity for each metric.
Computes the metric range [low, high] for each parameter around the base case.
"""
function generate_tornado_diagrams(
  theta_df::DataFrame,
  storage_df::DataFrame,
  weight_df::DataFrame,
  customer_df::DataFrame,
  folder::String;
  base_theta::Float64=0.5,
  base_scale::Float64=1.0,
)
  setup_plot_defaults()
  logmsg("Experiment 5E — Tornado Diagrams\n", color=:blue)
  mkpath(folder)

  # Extract ranges for each parameter
  param_names = ["Weight Factors", "Customer Willingness (θ)", "Storage Capacity", "Customer Count"]

  for metric in ["Cost", "Emission", "Loss", "Utility"]
    sym = Symbol(metric)
    low_vals = Float64[]
    high_vals = Float64[]

    # Weights: min/max across all simplex points
    valid_w = filter(r -> !isnan(r[sym]), weight_df)
    if nrow(valid_w) > 0
      push!(low_vals, minimum(valid_w[!, sym]))
      push!(high_vals, maximum(valid_w[!, sym]))
    else
      push!(low_vals, NaN)
      push!(high_vals, NaN)
    end

    # Theta
    valid_t = filter(r -> !isnan(r[sym]), theta_df)
    if nrow(valid_t) > 0
      push!(low_vals, minimum(valid_t[!, sym]))
      push!(high_vals, maximum(valid_t[!, sym]))
    else
      push!(low_vals, NaN)
      push!(high_vals, NaN)
    end

    # Storage
    valid_s = filter(r -> !isnan(r[sym]), storage_df)
    if nrow(valid_s) > 0
      push!(low_vals, minimum(valid_s[!, sym]))
      push!(high_vals, maximum(valid_s[!, sym]))
    else
      push!(low_vals, NaN)
      push!(high_vals, NaN)
    end

    # Customer count (aggregate means)
    if metric in names(customer_df)
      valid_c = filter(r -> !isnan(r[sym]), customer_df)
      if nrow(valid_c) > 0
        agg_c = combine(groupby(valid_c, :customers), sym => mean => :val)
        push!(low_vals, minimum(agg_c.val))
        push!(high_vals, maximum(agg_c.val))
      else
        push!(low_vals, NaN)
        push!(high_vals, NaN)
      end
    else
      push!(low_vals, NaN)
      push!(high_vals, NaN)
    end

    # Find base value (use midpoint or specific base case)
    base_value = (minimum(filter(!isnan, low_vals)) + maximum(filter(!isnan, high_vals))) / 2

    # Filter out NaN entries
    valid_mask = .!isnan.(low_vals) .& .!isnan.(high_vals)
    if any(valid_mask)
      p = tornado_diagram(
        param_names[valid_mask],
        low_vals[valid_mask],
        high_vals[valid_mask],
        base_value;
        xlabel=metric,
        title="Sensitivity — $metric",
      )
      save_figure(p, "tornado_$(lowercase(metric))"; folder=folder)
    end
  end
end

# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

"""
    run_sensitivity(; customers=5, generators=6)

Run all sensitivity analyses and generate tornado diagrams.
"""
function run_sensitivity(; customers::Int=5, generators::Int=6)
  w_bc = (1 / 3) * ones(3)
  periods = 24
  base_folder = "results/sensitivity"

  logmsg("=" ^ 60 * "\n", color=:blue)
  logmsg("EXPERIMENT 5: Sensitivity Analysis\n", color=:blue)
  logmsg("=" ^ 60 * "\n", color=:blue)

  # 5A: Weight sensitivity
  weight_df = sensitivity_weights(customers, generators, periods, "$base_folder/weights")

  # 5B: Theta sensitivity
  theta_df = sensitivity_theta(customers, generators, periods, "$base_folder/theta", w_bc)

  # 5C: Storage sensitivity
  storage_df = sensitivity_storage(customers, generators, periods, "$base_folder/storage", w_bc)

  # 5D: Customer count sensitivity
  customer_df = sensitivity_customers(generators, "$base_folder/customers", w_bc)

  # 5E: Tornado diagrams
  generate_tornado_diagrams(theta_df, storage_df, weight_df, customer_df, "$base_folder/tornado")

  return (weight_df, theta_df, storage_df, customer_df)
end
