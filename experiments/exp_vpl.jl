"""
    exp_vpl.jl — Experiment 7: Valve-Point Loading Sensitivity (multi-start)

    Solves the smooth-cost SGSD-DEED baseline once, then solves the VPL variant
    `n_starts` times with different warm-start dispatches (smooth optimum +
    Gaussian noise, clipped to [pjmin, pjmax]). Reports best/mean/worst stats and
    a dispatch comparison plot.

    All output stays inside `papercode/results/vpl/` per project convention; copy
    `vpl_dispatch.pdf` into `papertext/imgs/` separately when ready to compile.

    Usage:
      using Pkg; Pkg.activate(".")
      include("experiments.jl")
      run_vpl_experiment()                                    # SC1, 10 starts
      run_vpl_experiment(n_starts=20, w=[0.5, 0.3, 0.2])      # custom

    Returns NamedTuple `(smooth, runs_df, summary_df, q_smooth, q_vpl_best)`.
"""

# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────

"""
    run_vpl_experiment(; customers=5, generators=6, periods=24,
                        n_starts=10, w=ones(3)/3,
                        vpl_α=nothing, vpl_β=nothing,
                        seed=7001, noise_scale=0.15)

Run Experiment 7. The defaults reproduce the SC1 SGSD-DEED VPL evaluation
referenced by §4.5 / Section~\\ref{subsec:exp_vpl} of the manuscript.

Arguments
---------
- `customers, generators, periods`: scenario dimensions (default SC1 = 5×6×24).
- `n_starts`: number of multi-start runs of the VPL variant.
- `w`: weights for the Stackelberg leader objective `(w1, w2, w3)`.
- `vpl_α, vpl_β`: optional per-generator VPL amplitude/frequency vectors
  (length must equal `generators`). Defaults: `α=100`, `β=0.05` per generator.
- `seed`: RNG seed for reproducible warm-start perturbations.
- `noise_scale`: Gaussian noise scale, as a fraction of each generator's
  capacity range `(pjmax - pjmin)`. Larger ⇒ broader exploration.

Outputs (`papercode/results/vpl/`)
---------------------------------
- `vpl_runs.xlsx`     — per-start raw metrics (smooth baseline as row 0)
- `vpl_summary.xlsx`  — best/mean/worst summary across the multi-start runs
- `vpl_dispatch.pdf`  — dispatch trajectories: smooth (solid) vs VPL best (dashed)
"""
function run_vpl_experiment(;
  customers::Int=5,
  generators::Int=6,
  periods::Int=24,
  n_starts::Int=10,
  w::Vector{Float64}=(1 / 3) * ones(3),
  vpl_α::Union{Nothing,Vector{Float64}}=nothing,
  vpl_β::Union{Nothing,Vector{Float64}}=nothing,
  seed::Int=7001,
  noise_scale::Float64=0.15,
)
  setup_plot_defaults()
  folder = "results/vpl"
  mkpath(folder)
  Random.seed!(seed)

  logmsg("=" ^ 60 * "\n", color=:blue)
  logmsg("EXPERIMENT 7: Valve-Point Loading Sensitivity\n", color=:blue)
  logmsg("=" ^ 60 * "\n", color=:blue)
  logmsg("  c=$(customers), g=$(generators), T=$(periods), n_starts=$(n_starts), w=$(round.(w; digits=2))\n",
         color=:cyan)

  # ----- Base data (single instance shared by smooth + all VPL starts) -----
  base_gtdata = getGTDRDeedData(customers, generators, periods)
  pjmin = base_gtdata.deedData.pjmin
  pjmax = base_gtdata.deedData.pjmax
  range_j = pjmax .- pjmin

  # ===== Smooth-cost baseline (single solve — convex, deterministic) =====
  logmsg("\n[1/2] Smooth-cost baseline...\n", color=:cyan)
  smooth_model = gtdrdeed(customers, generators, periods; data=base_gtdata)
  smooth_result, t_smooth = solve_with_retry(; max_attempts=3) do
    smooth_model[:ws](w)
  end
  isnothing(smooth_result) && error("Smooth-cost baseline failed to converge.")
  m_smooth = extract_metrics_gtdeed(smooth_result, t_smooth)
  q_smooth = smooth_result.solution.deedSolution.q
  logmsg("  cost=\$$(round(m_smooth["Cost"]; digits=0)),  em=$(round(m_smooth["Emission"]; digits=0)) lb,  util=\$$(round(m_smooth["Utility"]; digits=0)),  time=$(round(m_smooth["Time"]; digits=2))s\n",
         color=:green)

  # ===== VPL multi-start =====
  logmsg("\n[2/2] VPL variant — $(n_starts) random warm starts (Gaussian σ = $(noise_scale)·(pmax-pmin))...\n",
         color=:cyan)

  rows = NamedTuple[]
  push!(rows, (start=0, kind="smooth", status="ok",
                Cost=m_smooth["Cost"], Emission=m_smooth["Emission"],
                Utility=m_smooth["Utility"], Loss=m_smooth["Loss"],
                Time=m_smooth["Time"]))

  best_cost = Inf
  best_q = q_smooth
  best_metrics = nothing

  for k in 1:n_starts
    # Warm start: smooth optimum + Gaussian noise, clipped to capacity bounds
    noise = noise_scale .* range_j .* randn(generators, periods)
    q_init = clamp.(q_smooth .+ noise, pjmin, pjmax)

    vpl_model = gtdrdeed(customers, generators, periods;
                          data=base_gtdata, vpl=true, vpl_α=vpl_α, vpl_β=vpl_β)

    t_run = 0.0
    result = nothing
    try
      timed = @timed vpl_model[:ws](w; q_init=q_init)
      result = timed.value
      t_run = timed.time
    catch e
      logmsg("    start $k: EXCEPTION $(typeof(e))\n", color=:red)
      result = nothing
    end

    if isa(result, SuccessResult)
      m = extract_metrics_gtdeed(result, t_run)
      push!(rows, (start=k, kind="vpl", status="ok",
                    Cost=m["Cost"], Emission=m["Emission"],
                    Utility=m["Utility"], Loss=m["Loss"], Time=m["Time"]))
      logmsg("    start $k:  cost=\$$(round(m["Cost"]; digits=0)),  em=$(round(m["Emission"]; digits=0)),  time=$(round(t_run; digits=1))s\n",
             color=:cyan)
      if m["Cost"] < best_cost
        best_cost = m["Cost"]
        best_q = result.solution.deedSolution.q
        best_metrics = m
      end
    else
      push!(rows, (start=k, kind="vpl", status="fail",
                    Cost=NaN, Emission=NaN, Utility=NaN, Loss=NaN, Time=t_run))
      logmsg("    start $k:  FAILED ($(round(t_run; digits=1))s)\n", color=:yellow)
    end
  end

  runs_df = DataFrame(rows)
  excel_runs = outputfilename("vpl_runs"; dated=false, root=folder)
  XLSX.writetable("$(excel_runs).xlsx", "VPL_RUNS" => runs_df, overwrite=true)
  logmsg("  Saved: $(excel_runs).xlsx\n", color=:green)

  # ===== Summary =====
  vpl_valid = filter(r -> r.kind == "vpl" && r.status == "ok", runs_df)
  if nrow(vpl_valid) == 0
    @warn "All VPL starts failed; summary will only contain the smooth row."
    summary_df = DataFrame(
      Variant=["Smooth quadratic"],
      Cost=[m_smooth["Cost"]],
      Emission=[m_smooth["Emission"]],
      Utility=[m_smooth["Utility"]],
      Loss=[m_smooth["Loss"]],
      Time=[m_smooth["Time"]],
    )
  else
    i_best = argmin(vpl_valid.Cost)
    i_worst = argmax(vpl_valid.Cost)
    summary_df = DataFrame(
      Variant=["Smooth quadratic",
        "VPL (best of $(nrow(vpl_valid)))",
        "VPL (mean)",
        "VPL (worst of $(nrow(vpl_valid)))",
        "VPL (std)"],
      Cost=[m_smooth["Cost"], vpl_valid.Cost[i_best], mean(vpl_valid.Cost),
        vpl_valid.Cost[i_worst], std(vpl_valid.Cost)],
      Emission=[m_smooth["Emission"], vpl_valid.Emission[i_best],
        mean(vpl_valid.Emission), vpl_valid.Emission[i_worst],
        std(vpl_valid.Emission)],
      Utility=[m_smooth["Utility"], vpl_valid.Utility[i_best],
        mean(vpl_valid.Utility), vpl_valid.Utility[i_worst],
        std(vpl_valid.Utility)],
      Loss=[m_smooth["Loss"], vpl_valid.Loss[i_best], mean(vpl_valid.Loss),
        vpl_valid.Loss[i_worst], std(vpl_valid.Loss)],
      Time=[m_smooth["Time"], vpl_valid.Time[i_best], mean(vpl_valid.Time),
        vpl_valid.Time[i_worst], std(vpl_valid.Time)],
    )
  end
  excel_sum = outputfilename("vpl_summary"; dated=false, root=folder)
  XLSX.writetable("$(excel_sum).xlsx", "VPL_SUMMARY" => summary_df, overwrite=true)
  logmsg("  Saved: $(excel_sum).xlsx\n", color=:green)

  # ===== Dispatch comparison plot =====
  hours = 1:periods
  p = plot(
    xlabel="Hour",
    ylabel="Generator dispatch  q_j^t  (MW)",
    title="Dispatch under smooth (solid) vs.\\ VPL best (dashed) — SC1",
    legend=:outertopright,
    foreground_color_legend=nothing,
    background_color_legend=RGBA(1, 1, 1, 0.85),
    size=(720, 460),
  )
  for j in 1:generators
    plot!(p, hours, q_smooth[j, :], label="Gen $j (smooth)",
          linewidth=1.8, linestyle=:solid, color=j)
    plot!(p, hours, best_q[j, :], label="Gen $j (VPL)",
          linewidth=1.8, linestyle=:dash, color=j)
  end
  fig_path = outputfilename("vpl_dispatch"; dated=false, root=folder)
  savefig(p, "$(fig_path).pdf")
  logmsg("  Saved: $(fig_path).pdf\n", color=:green)

  logmsg("\nEXPERIMENT 7 COMPLETE — copy $(folder)/vpl_dispatch.pdf into ../papertext/imgs/ when ready to compile.\n",
         color=:green)

  return (smooth=m_smooth, runs_df=runs_df, summary_df=summary_df,
          q_smooth=q_smooth, q_vpl_best=best_q)
end
