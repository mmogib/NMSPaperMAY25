"""
    analysis.jl — Deep data analysis of all experiment XLSX files

    Reads all 22 XLSX files, prints full tables with std dev,
    computes CV, identifies outliers and non-monotonic patterns,
    and answers key narrative questions for the paper.

    Usage:
        cd("papercode")
        using Pkg; Pkg.activate(".")
        include("analysis.jl")
"""

using DataFrames, Statistics, XLSX, Dates

const RESULTS = "results"

# ─────────────────────────────────────────────────────────────────
# Utility: read a single-sheet XLSX into DataFrame
# ─────────────────────────────────────────────────────────────────

function read_xlsx(path::String, sheet::String)
    xlsx = XLSX.readxlsx(path)
    DataFrame(XLSX.eachtablerow(xlsx[sheet]))
end

function read_xlsx_first_sheet(path::String)
    xlsx = XLSX.readxlsx(path)
    snames = XLSX.sheetnames(xlsx)
    DataFrame(XLSX.eachtablerow(xlsx[snames[1]]))
end

function safe_cv(mean_val, std_val)
    (ismissing(mean_val) || ismissing(std_val) || mean_val == 0 || isnan(mean_val) || isnan(std_val)) ? NaN : abs(std_val / mean_val) * 100
end

function print_separator(title::String)
    println("\n", "=" ^ 80)
    println("  ", title)
    println("=" ^ 80)
end

function print_subsep(title::String)
    println("\n  --- ", title, " ---")
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Scenario Testing (8 files)
# ─────────────────────────────────────────────────────────────────

function analyze_experiment1()
    print_separator("EXPERIMENT 1: Scenario Testing (Weight Variations)")

    scenarios = [("scenario1", "SC1 (5c,6g)"), ("scenario2", "SC2 (7c,10g)")]
    weights = ["BC", "C2", "C3", "C4"]

    for (sc_folder, sc_name) in scenarios
        print_subsep(sc_name)
        for wt in weights
            path = joinpath(RESULTS, sc_folder, wt, "2024_05_15", "load_profile.xlsx")
            if !isfile(path)
                println("    [MISSING] $path")
                continue
            end
            xlsx = XLSX.readxlsx(path)
            snames = XLSX.sheetnames(xlsx)
            println("    Weight=$wt: sheets=$(snames)")

            for sn in snames
                try
                    df = DataFrame(XLSX.eachtablerow(xlsx[sn]))
                    if nrow(df) > 0
                        println("      Sheet '$sn': $(nrow(df)) rows x $(ncol(df)) cols")
                        println("      Columns: $(names(df))")
                        # Print summary for numeric columns
                        for col in names(df)
                            vals = df[!, col]
                            if eltype(vals) <: Union{Number, Missing}
                                clean = collect(skipmissing(filter(x -> !ismissing(x) && !isnan(x), vals)))
                                if !isempty(clean)
                                    println("        $col: min=$(round(minimum(clean), digits=4)), max=$(round(maximum(clean), digits=4)), mean=$(round(mean(clean), digits=4))")
                                end
                            end
                        end
                    end
                catch e
                    println("      Sheet '$sn': Error reading — $e")
                end
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Scalability Study
# ─────────────────────────────────────────────────────────────────

function analyze_experiment2()
    print_separator("EXPERIMENT 2: Scalability Study (50-400 customers x 4-20 generators)")

    path = joinpath(RESULTS, "experiment2", "2024_05_15", "solutions.xlsx")
    if !isfile(path)
        println("  [MISSING] $path")
        return nothing
    end

    df = read_xlsx(path, "SOLUTIONS")
    println("  Raw data: $(nrow(df)) rows x $(ncol(df)) cols")
    println("  Columns: $(names(df))")

    # Filter successful runs
    if :success in Symbol.(names(df))
        success_df = filter(r -> r.success == 1.0, df)
        fail_df = filter(r -> r.success != 1.0, df)
        println("  Successful: $(nrow(success_df)) / $(nrow(df))  ($(nrow(fail_df)) failures)")
        if nrow(fail_df) > 0
            println("  Failed configs:")
            for r in eachrow(fail_df)
                println("    c=$(Int(r.c)), g=$(Int(r.g))")
            end
        end
    else
        success_df = df
    end

    # Group by (c, g) and compute mean + std
    grouped = combine(
        groupby(success_df, [:c, :g]),
        :cost => mean => :cost_mean,
        :cost => std => :cost_std,
        :emission => mean => :emission_mean,
        :emission => std => :emission_std,
        :utility => mean => :utility_mean,
        :utility => std => :utility_std,
        :loss => mean => :loss_mean,
        :loss => std => :loss_std,
        :time => mean => :time_mean,
        :time => std => :time_std,
        nrow => :n_trials,
    )

    # Replace missing std (single trial) with 0
    for col in names(grouped)
        if endswith(String(col), "_std")
            grouped[!, col] = coalesce.(grouped[!, col], 0.0)
        end
    end

    # Compute CV
    grouped.cost_cv = safe_cv.(grouped.cost_mean, grouped.cost_std)
    grouped.emission_cv = safe_cv.(grouped.emission_mean, grouped.emission_std)

    println("\n  Aggregated results (mean ± std, CV%):")
    println("  " * "-" ^ 130)
    println("  c     g    | Cost (mean±std) [CV%]         | Emission (mean±std) [CV%]      | Utility mean      | Loss mean     | Time (s)")
    println("  " * "-" ^ 130)

    for r in eachrow(sort(grouped, [:c, :g]))
        cost_str = "$(round(r.cost_mean, digits=1)) ± $(round(r.cost_std, digits=1)) [$(round(r.cost_cv, digits=1))%]"
        emis_str = "$(round(r.emission_mean, digits=1)) ± $(round(r.emission_std, digits=1)) [$(round(r.emission_cv, digits=1))%]"
        println("  $(lpad(Int(r.c), 4)) $(lpad(Int(r.g), 4))  | $(rpad(cost_str, 30))| $(rpad(emis_str, 31))| $(rpad(round(r.utility_mean, digits=1), 18))| $(rpad(round(r.loss_mean, digits=4), 14))| $(round(r.time_mean, digits=2))")
    end

    # Find optimal config (minimum cost)
    print_subsep("Optimal Configuration (minimum cost)")
    min_idx = argmin(grouped.cost_mean)
    opt = grouped[min_idx, :]
    println("    Best cost: c=$(Int(opt.c)), g=$(Int(opt.g)) → Cost=$(round(opt.cost_mean, digits=2))")

    # Find non-monotonic patterns
    print_subsep("Non-monotonic patterns in cost vs customers (for each generator count)")
    for g_val in sort(unique(grouped.g))
        sub = sort(filter(r -> r.g == g_val, grouped), :c)
        costs = sub.cost_mean
        for i in 2:length(costs)
            if costs[i] < costs[i-1]
                println("    g=$(Int(g_val)): cost DECREASES from c=$(Int(sub.c[i-1])) to c=$(Int(sub.c[i])): $(round(costs[i-1], digits=1)) → $(round(costs[i], digits=1))")
            end
        end
    end

    return grouped
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3: IEEE 30-Bus (2 files)
# ─────────────────────────────────────────────────────────────────

function analyze_experiment3()
    print_separator("EXPERIMENT 3: IEEE 30-Bus DC-OPF Network Effect")

    for (sc, sc_name) in [("SC1", "SC1 (5c,6g)"), ("SC2", "SC2 (7c,10g)")]
        print_subsep("$sc_name")
        path = joinpath(RESULTS, "experiment3_ieee30", sc, "ieee30_comparison.xlsx")
        if !isfile(path)
            println("    [MISSING] $path")
            continue
        end

        df = read_xlsx(path, "SUMMARY")
        println("    Columns: $(names(df))")

        for r in eachrow(df)
            cfg = r.Config
            cost = round(r.Cost, digits=2)
            emis = round(r.Emission, digits=2)
            loss = round(r.Loss, digits=4)
            util = round(r.Utility, digits=2)

            cost_std = hasproperty(r, :Cost_std) ? round(r.Cost_std, digits=2) : NaN
            emis_std = hasproperty(r, :Emission_std) ? round(r.Emission_std, digits=2) : NaN
            loss_std = hasproperty(r, :Loss_std) ? round(r.Loss_std, digits=4) : NaN
            util_std = hasproperty(r, :Utility_std) ? round(r.Utility_std, digits=2) : NaN

            cost_cv = safe_cv(r.Cost, hasproperty(r, :Cost_std) ? r.Cost_std : NaN)
            emis_cv = safe_cv(r.Emission, hasproperty(r, :Emission_std) ? r.Emission_std : NaN)

            println("    $cfg:")
            println("      Cost:     $cost ± $cost_std  (CV=$(round(cost_cv, digits=1))%)")
            println("      Emission: $emis ± $emis_std  (CV=$(round(emis_cv, digits=1))%)")
            println("      Loss:     $loss ± $loss_std")
            println("      Utility:  $util ± $util_std")
        end

        # Compute % change
        if nrow(df) == 2
            no_net = df[1, :]
            dc_opf = df[2, :]
            cost_pct = 100 * (dc_opf.Cost - no_net.Cost) / abs(no_net.Cost)
            emis_pct = 100 * (dc_opf.Emission - no_net.Emission) / abs(no_net.Emission)
            loss_pct = 100 * (dc_opf.Loss - no_net.Loss) / abs(no_net.Loss)
            println("    DC-OPF effect: Cost $(round(cost_pct, digits=2))%, Emission $(round(emis_pct, digits=2))%, Loss $(round(loss_pct, digits=2))%")

            # Check if std dev overlaps
            if hasproperty(no_net, :Cost_std) && hasproperty(dc_opf, :Cost_std)
                gap = abs(dc_opf.Cost - no_net.Cost)
                combined_std = sqrt(no_net.Cost_std^2 + dc_opf.Cost_std^2)
                sig = combined_std > 0 ? gap / combined_std : Inf
                println("    Cost significance: gap=$( round(gap, digits=2)), combined_std=$(round(combined_std, digits=2)), ratio=$(round(sig, digits=2))σ")
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Model Progression (2 files)
# ─────────────────────────────────────────────────────────────────

function analyze_experiment4()
    print_separator("EXPERIMENT 4: Model Progression (DEED → DR-DEED → SGSD-DEED)")

    for (sc, sc_name) in [("SC1", "SC1 (5c,6g)"), ("SC2", "SC2 (7c,10g)")]
        print_subsep("$sc_name")
        path = joinpath(RESULTS, "experiment4_progression", sc, "progression.xlsx")
        if !isfile(path)
            println("    [MISSING] $path")
            continue
        end

        df = read_xlsx(path, "SUMMARY")
        println("    Columns: $(names(df))")

        deed_cost = NaN
        deed_emis = NaN
        deed_loss = NaN

        for r in eachrow(df)
            model = r.Model
            cost = round(r.Cost, digits=2)
            emis = round(r.Emission, digits=2)
            loss = round(r.Loss, digits=4)
            util = round(r.Utility, digits=2)

            cost_std = hasproperty(r, :Cost_std) ? round(r.Cost_std, digits=2) : NaN
            emis_std = hasproperty(r, :Emission_std) ? round(r.Emission_std, digits=2) : NaN

            cost_cv = safe_cv(r.Cost, hasproperty(r, :Cost_std) ? r.Cost_std : NaN)

            if model == "DEED"
                deed_cost = r.Cost
                deed_emis = r.Emission
                deed_loss = r.Loss
            end

            cost_red = model == "DEED" ? "—" : "$(round(100*(deed_cost - r.Cost)/abs(deed_cost), digits=1))%"
            emis_red = model == "DEED" ? "—" : "$(round(100*(deed_emis - r.Emission)/abs(deed_emis), digits=1))%"

            println("    $model:")
            println("      Cost:     $cost ± $cost_std  (CV=$(round(cost_cv, digits=1))%)  Reduction: $cost_red")
            println("      Emission: $emis ± $emis_std  Reduction: $emis_red")
            println("      Loss:     $loss    Utility: $util")
        end

        # Check monotonicity: DEED > DR-DEED > SGSD-DEED for cost
        models = df.Model
        costs = df.Cost
        if length(costs) == 3
            monotone = costs[1] > costs[2] > costs[3]
            println("    Monotonicity (Cost): DEED($(round(costs[1], digits=0))) > DR-DEED($(round(costs[2], digits=0))) > SGSD-DEED($(round(costs[3], digits=0))): $monotone")
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 5A: Weight Sensitivity
# ─────────────────────────────────────────────────────────────────

function analyze_experiment5_weights()
    print_separator("EXPERIMENT 5A: Weight Factor Sensitivity (64 simplex points)")

    path = joinpath(RESULTS, "experiment5_sensitivity", "weights", "weight_sensitivity.xlsx")
    if !isfile(path)
        println("  [MISSING] $path")
        return nothing
    end

    df = read_xlsx(path, "WEIGHTS")
    println("  Data: $(nrow(df)) rows x $(ncol(df)) cols")
    println("  Columns: $(names(df))")

    valid = filter(r -> !isnan(r.Cost), df)
    println("  Valid points: $(nrow(valid)) / $(nrow(df))")

    for metric in [:Cost, :Emission, :Loss, :Utility]
        vals = valid[!, metric]
        println("  $metric: min=$(round(minimum(vals), digits=2)), max=$(round(maximum(vals), digits=2)), range=$(round(maximum(vals) - minimum(vals), digits=2)), CV=$(round(safe_cv(mean(vals), std(vals)), digits=1))%")

        # Find the weight config that minimizes/maximizes each metric
        min_idx = argmin(vals)
        max_idx = argmax(vals)
        println("    Min at w=($(round(valid.w1[min_idx], digits=2)), $(round(valid.w2[min_idx], digits=2)), $(round(valid.w3[min_idx], digits=2))): $(round(vals[min_idx], digits=2))")
        println("    Max at w=($(round(valid.w1[max_idx], digits=2)), $(round(valid.w2[max_idx], digits=2)), $(round(valid.w3[max_idx], digits=2))): $(round(vals[max_idx], digits=2))")
    end

    # Cost-emission correlation
    corr = cor(valid.Cost, valid.Emission)
    println("  Cost-Emission correlation: $(round(corr, digits=4))")

    # Cost-Utility correlation
    corr_cu = cor(valid.Cost, valid.Utility)
    println("  Cost-Utility correlation: $(round(corr_cu, digits=4))")

    return valid
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 5B: Theta (Customer Willingness) Sensitivity
# ─────────────────────────────────────────────────────────────────

function analyze_experiment5_theta()
    print_separator("EXPERIMENT 5B: Customer Willingness (θ) Sensitivity")

    path = joinpath(RESULTS, "experiment5_sensitivity", "theta", "theta_sensitivity.xlsx")
    if !isfile(path)
        println("  [MISSING] $path")
        return nothing
    end

    df = read_xlsx(path, "THETA")
    println("  Data: $(nrow(df)) rows")
    println("  Columns: $(names(df))")

    valid = filter(r -> !isnan(r.Cost), df)
    println("  Valid: $(nrow(valid)) / $(nrow(df))")

    println("\n  θ       | Cost          | Emission      | Loss          | Utility")
    println("  " * "-" ^ 80)
    for r in eachrow(sort(valid, :theta))
        println("  $(rpad(round(r.theta, digits=2), 8))| $(rpad(round(r.Cost, digits=2), 14))| $(rpad(round(r.Emission, digits=2), 14))| $(rpad(round(r.Loss, digits=4), 14))| $(round(r.Utility, digits=2))")
    end

    # Check monotonicity
    sorted = sort(valid, :theta)
    for metric in [:Cost, :Emission, :Loss, :Utility]
        vals = sorted[!, metric]
        increasing = all(diff(vals) .>= 0)
        decreasing = all(diff(vals) .<= 0)
        mono = increasing ? "monotonically increasing" : (decreasing ? "monotonically decreasing" : "NON-MONOTONIC")
        println("  $metric vs θ: $mono")
        if !increasing && !decreasing
            # Find reversals
            diffs = diff(vals)
            for i in 1:length(diffs)
                if i > 1 && sign(diffs[i]) != sign(diffs[i-1]) && diffs[i] != 0 && diffs[i-1] != 0
                    println("    Reversal at θ=$(round(sorted.theta[i+1], digits=2)): $(round(vals[i], digits=2)) → $(round(vals[i+1], digits=2))")
                end
            end
        end
    end

    return valid
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 5C: Storage Capacity Sensitivity
# ─────────────────────────────────────────────────────────────────

function analyze_experiment5_storage()
    print_separator("EXPERIMENT 5C: Storage Capacity Sensitivity")

    path = joinpath(RESULTS, "experiment5_sensitivity", "storage", "storage_sensitivity.xlsx")
    if !isfile(path)
        println("  [MISSING] $path")
        return nothing
    end

    df = read_xlsx(path, "STORAGE")
    println("  Data: $(nrow(df)) rows")
    println("  Columns: $(names(df))")

    valid = filter(r -> !isnan(r.Cost), df)
    println("  Valid: $(nrow(valid)) / $(nrow(df))")

    println("\n  Scale   | Cost          | Emission      | Loss          | Utility")
    println("  " * "-" ^ 80)
    for r in eachrow(sort(valid, :scale))
        println("  $(rpad(r.scale, 8))| $(rpad(round(r.Cost, digits=2), 14))| $(rpad(round(r.Emission, digits=2), 14))| $(rpad(round(r.Loss, digits=4), 14))| $(round(r.Utility, digits=2))")
    end

    # Key question: phase transition at 0.25-0.50x?
    sorted = sort(valid, :scale)
    print_subsep("Storage Phase Transition Analysis")
    for metric in [:Cost, :Emission, :Loss, :Utility]
        vals = sorted[!, metric]
        scales = sorted.scale
        println("  $metric:")
        for i in 2:length(vals)
            pct_change = 100 * (vals[i] - vals[i-1]) / abs(vals[i-1])
            println("    $(scales[i-1])x → $(scales[i])x: Δ=$(round(pct_change, digits=2))%  ($(round(vals[i-1], digits=2)) → $(round(vals[i], digits=2)))")
        end
    end

    # Check for diminishing returns
    print_subsep("Diminishing Returns Analysis (Cost)")
    costs = sorted.Cost
    scales = sorted.scale
    if length(costs) >= 3
        for i in 2:length(costs)-1
            marginal_before = (costs[i] - costs[i-1]) / (scales[i] - scales[i-1])
            marginal_after = (costs[i+1] - costs[i]) / (scales[i+1] - scales[i])
            println("    At $(scales[i])x: marginal before=$(round(marginal_before, digits=2)), marginal after=$(round(marginal_after, digits=2))")
        end
    end

    return valid
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 5D: Customer Count Sensitivity
# ─────────────────────────────────────────────────────────────────

function analyze_experiment5_customers()
    print_separator("EXPERIMENT 5D: Customer Count Sensitivity (5→50)")

    path = joinpath(RESULTS, "experiment5_sensitivity", "customers", "customer_sensitivity.xlsx")
    if !isfile(path)
        println("  [MISSING] $path")
        return nothing
    end

    df = read_xlsx(path, "CUSTOMERS")
    println("  Data: $(nrow(df)) rows ($(length(unique(df.customers))) customer counts × trials)")
    println("  Columns: $(names(df))")

    valid = filter(r -> !isnan(r.Cost), df)

    # Aggregate by customer count
    agg = combine(
        groupby(valid, :customers),
        :Cost => mean => :Cost_mean,
        :Cost => std => :Cost_std,
        :Emission => mean => :Emission_mean,
        :Emission => std => :Emission_std,
        :Loss => mean => :Loss_mean,
        :Loss => std => :Loss_std,
        :Utility => mean => :Utility_mean,
        :Utility => std => :Utility_std,
        :Time => mean => :Time_mean,
        :Time => std => :Time_std,
        nrow => :n_trials,
    )

    for col in names(agg)
        if endswith(String(col), "_std")
            agg[!, col] = coalesce.(agg[!, col], 0.0)
        end
    end

    agg.Cost_cv = safe_cv.(agg.Cost_mean, agg.Cost_std)

    sort!(agg, :customers)

    println("\n  Customers | Cost (mean±std) [CV%]            | Emission (mean±std)       | Utility mean      | Loss mean       | Time (s) | n")
    println("  " * "-" ^ 140)
    for r in eachrow(agg)
        cost_str = "$(round(r.Cost_mean, digits=1)) ± $(round(r.Cost_std, digits=1)) [$(round(r.Cost_cv, digits=1))%]"
        emis_str = "$(round(r.Emission_mean, digits=1)) ± $(round(r.Emission_std, digits=1))"
        println("  $(rpad(Int(r.customers), 10))| $(rpad(cost_str, 33))| $(rpad(emis_str, 26))| $(rpad(round(r.Utility_mean, digits=1), 18))| $(rpad(round(r.Loss_mean, digits=4), 16))| $(rpad(round(r.Time_mean, digits=2), 9))| $(r.n_trials)")
    end

    # Key question: Non-monotonicity at 30/40 customers?
    print_subsep("Non-monotonicity Analysis")
    for metric in [:Cost_mean, :Emission_mean, :Loss_mean, :Utility_mean]
        vals = agg[!, metric]
        customers = agg.customers
        println("  $metric:")
        non_mono = false
        for i in 2:length(vals)
            pct_change = 100 * (vals[i] - vals[i-1]) / abs(vals[i-1])
            direction = vals[i] > vals[i-1] ? "↑" : "↓"
            # Flag if direction reverses
            if i >= 3
                prev_dir = vals[i-1] > vals[i-2]
                curr_dir = vals[i] > vals[i-1]
                if prev_dir != curr_dir
                    println("    *** REVERSAL at $(Int(customers[i])): $(round(vals[i-1], digits=1)) → $(round(vals[i], digits=1)) ($direction $(round(abs(pct_change), digits=1))%)")
                    non_mono = true

                    # Check if reversal is within std dev
                    std_col = Symbol(replace(String(metric), "_mean" => "_std"))
                    if std_col in Symbol.(names(agg))
                        std_val = agg[i, std_col]
                        gap = abs(vals[i] - vals[i-1])
                        println("      Gap=$( round(gap, digits=1)), Std=$(round(std_val, digits=1)), Gap/Std=$(round(gap/std_val, digits=2))σ — $(gap < 2*std_val ? "WITHIN noise" : "SIGNIFICANT")")
                    end
                end
            end
        end
        if !non_mono
            println("    Monotonic ✓")
        end
    end

    # Scaling behavior
    print_subsep("Scaling: Cost per Customer")
    for r in eachrow(agg)
        cost_per_cust = r.Cost_mean / r.customers
        println("    c=$(rpad(Int(r.customers), 4)): Cost/customer = $(round(cost_per_cust, digits=2))")
    end

    return agg
end

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 6: Saudi Eastern Province
# ─────────────────────────────────────────────────────────────────

function analyze_experiment6()
    print_separator("EXPERIMENT 6: Saudi Eastern Province Case Study")

    # 6A: IEEE 30-bus network effect
    print_subsep("6A: Network Effect (Saudi Data)")
    for (sc, sc_name) in [("SC1", "SC1 (5c,6g)"), ("SC2", "SC2 (7c,10g)")]
        println("    $sc_name:")
        path = joinpath(RESULTS, "experiment6_saudi", "ieee30", sc, "saudi_ieee30_comparison.xlsx")
        if !isfile(path)
            println("      [MISSING] $path")
            continue
        end

        df = read_xlsx(path, "SUMMARY")
        for r in eachrow(df)
            cost_cv = safe_cv(r.Cost, hasproperty(r, :Cost_std) ? r.Cost_std : NaN)
            println("      $(r.Config): Cost=$(round(r.Cost, digits=2)) ± $(round(hasproperty(r, :Cost_std) ? r.Cost_std : NaN, digits=2)) [CV=$(round(cost_cv, digits=1))%], Emission=$(round(r.Emission, digits=2)) ± $(round(hasproperty(r, :Emission_std) ? r.Emission_std : NaN, digits=2)), Loss=$(round(r.Loss, digits=4)), Utility=$(round(r.Utility, digits=2))")
        end

        if nrow(df) == 2
            cost_pct = 100 * (df.Cost[2] - df.Cost[1]) / abs(df.Cost[1])
            emis_pct = 100 * (df.Emission[2] - df.Emission[1]) / abs(df.Emission[1])
            loss_pct = 100 * (df.Loss[2] - df.Loss[1]) / abs(df.Loss[1])
            println("      DC-OPF effect: Cost $(round(cost_pct, digits=2))%, Emission $(round(emis_pct, digits=2))%, Loss $(round(loss_pct, digits=2))%")
        end
    end

    # 6B: Model progression
    print_subsep("6B: Model Progression (Saudi Data)")
    for (sc, sc_name) in [("SC1", "SC1 (5c,6g)"), ("SC2", "SC2 (7c,10g)")]
        println("    $sc_name:")
        path = joinpath(RESULTS, "experiment6_saudi", "progression", sc, "saudi_progression.xlsx")
        if !isfile(path)
            println("      [MISSING] $path")
            continue
        end

        df = read_xlsx(path, "SUMMARY")

        deed_cost = NaN
        deed_emis = NaN
        for r in eachrow(df)
            if r.Model == "DEED"
                deed_cost = r.Cost
                deed_emis = r.Emission
            end
            cost_red = r.Model == "DEED" ? "—" : "$(round(100*(deed_cost - r.Cost)/abs(deed_cost), digits=1))%"
            emis_red = r.Model == "DEED" ? "—" : "$(round(100*(deed_emis - r.Emission)/abs(deed_emis), digits=1))%"
            cost_cv = safe_cv(r.Cost, hasproperty(r, :Cost_std) ? r.Cost_std : NaN)
            println("      $(r.Model): Cost=$(round(r.Cost, digits=2)) ± $(round(hasproperty(r, :Cost_std) ? r.Cost_std : NaN, digits=2)) [CV=$(round(cost_cv, digits=1))%], Emission=$(round(r.Emission, digits=2)), Loss=$(round(r.Loss, digits=4)), Utility=$(round(r.Utility, digits=2)), CostRed=$cost_red, EmisRed=$emis_red")
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# CROSS-EXPERIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────

function analyze_cross_experiment()
    print_separator("CROSS-EXPERIMENT ANALYSIS: PJM vs Saudi Consistency")

    # Compare Experiment 3 (PJM) vs Experiment 6A (Saudi) — network effect
    println("\n  Network Effect Comparison (DC-OPF % change vs No Network):")
    println("  " * "-" ^ 80)

    for (sc, sc_name) in [("SC1", "SC1"), ("SC2", "SC2")]
        println("  $sc_name:")

        # PJM (Exp 3)
        pjm_path = joinpath(RESULTS, "experiment3_ieee30", sc, "ieee30_comparison.xlsx")
        if isfile(pjm_path)
            pjm = read_xlsx(pjm_path, "SUMMARY")
            if nrow(pjm) == 2
                pjm_cost_pct = 100 * (pjm.Cost[2] - pjm.Cost[1]) / abs(pjm.Cost[1])
                pjm_emis_pct = 100 * (pjm.Emission[2] - pjm.Emission[1]) / abs(pjm.Emission[1])
                println("    PJM:   Cost Δ=$(round(pjm_cost_pct, digits=2))%, Emission Δ=$(round(pjm_emis_pct, digits=2))%")
            end
        end

        # Saudi (Exp 6A)
        saudi_path = joinpath(RESULTS, "experiment6_saudi", "ieee30", sc, "saudi_ieee30_comparison.xlsx")
        if isfile(saudi_path)
            saudi = read_xlsx(saudi_path, "SUMMARY")
            if nrow(saudi) == 2
                saudi_cost_pct = 100 * (saudi.Cost[2] - saudi.Cost[1]) / abs(saudi.Cost[1])
                saudi_emis_pct = 100 * (saudi.Emission[2] - saudi.Emission[1]) / abs(saudi.Emission[1])
                println("    Saudi: Cost Δ=$(round(saudi_cost_pct, digits=2))%, Emission Δ=$(round(saudi_emis_pct, digits=2))%")
            end
        end
    end

    # Compare Experiment 4 (PJM) vs Experiment 6B (Saudi) — model progression
    println("\n  Model Progression Comparison (SGSD-DEED cost reduction vs DEED):")
    println("  " * "-" ^ 80)

    for (sc, sc_name) in [("SC1", "SC1"), ("SC2", "SC2")]
        println("  $sc_name:")

        pjm_path = joinpath(RESULTS, "experiment4_progression", sc, "progression.xlsx")
        if isfile(pjm_path)
            pjm = read_xlsx(pjm_path, "SUMMARY")
            if nrow(pjm) == 3
                pjm_red = 100 * (pjm.Cost[1] - pjm.Cost[3]) / abs(pjm.Cost[1])
                println("    PJM:   SGSD-DEED cost reduction = $(round(pjm_red, digits=1))%")
            end
        end

        saudi_path = joinpath(RESULTS, "experiment6_saudi", "progression", sc, "saudi_progression.xlsx")
        if isfile(saudi_path)
            saudi = read_xlsx(saudi_path, "SUMMARY")
            if nrow(saudi) == 3
                saudi_red = 100 * (saudi.Cost[1] - saudi.Cost[3]) / abs(saudi.Cost[1])
                println("    Saudi: SGSD-DEED cost reduction = $(round(saudi_red, digits=1))%")
            end
        end
    end

    # Solver variance comparison
    println("\n  Solver Variance Comparison (CV% on Cost):")
    println("  " * "-" ^ 80)

    for (sc, sc_name) in [("SC1", "SC1"), ("SC2", "SC2")]
        println("  $sc_name:")

        # Exp 3 PJM
        pjm_path = joinpath(RESULTS, "experiment3_ieee30", sc, "ieee30_comparison.xlsx")
        if isfile(pjm_path)
            pjm = read_xlsx(pjm_path, "SUMMARY")
            for r in eachrow(pjm)
                cv = safe_cv(r.Cost, hasproperty(r, :Cost_std) ? r.Cost_std : NaN)
                println("    PJM $(r.Config): CV=$(round(cv, digits=2))%")
            end
        end

        # Exp 6A Saudi
        saudi_path = joinpath(RESULTS, "experiment6_saudi", "ieee30", sc, "saudi_ieee30_comparison.xlsx")
        if isfile(saudi_path)
            saudi = read_xlsx(saudi_path, "SUMMARY")
            for r in eachrow(saudi)
                cv = safe_cv(r.Cost, hasproperty(r, :Cost_std) ? r.Cost_std : NaN)
                println("    Saudi $(r.Config): CV=$(round(cv, digits=2))%")
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# MAIN: Run all analyses
# ─────────────────────────────────────────────────────────────────

function run_analysis()
    println("╔══════════════════════════════════════════════════════════════════════════╗")
    println("║  SGSD-DEED Deep Data Analysis — All Experiments                         ║")
    println("║  $(Dates.now())                                                  ║")
    println("╚══════════════════════════════════════════════════════════════════════════╝")

    analyze_experiment1()
    exp2 = analyze_experiment2()
    analyze_experiment3()
    analyze_experiment4()
    wt = analyze_experiment5_weights()
    theta = analyze_experiment5_theta()
    storage = analyze_experiment5_storage()
    cust = analyze_experiment5_customers()
    analyze_experiment6()
    analyze_cross_experiment()

    print_separator("ANALYSIS COMPLETE")
    println("  All experiments analyzed. See above for detailed findings.")

    return (; exp2, weights=wt, theta, storage, customers=cust)
end

# Auto-run when included
run_analysis()
