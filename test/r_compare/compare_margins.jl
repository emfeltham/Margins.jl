# compare_margins.jl
# Compare marginal effects between Julia and R

using DataFrames, CSV
using Printf
using Statistics

println("="^80)
println("Marginal Effects Comparison: Julia vs R")
println("="^80)

# Function to standardize variable/factor names for comparison
function standardize_var_name(name)
    # Convert to String first (handles InlineString types)
    name_str = String(name)
    # Remove spaces after categorical levels
    name_str = replace(name_str, r"\s+" => "")
    return name_str
end

# Function to compare two marginal effects DataFrames
function compare_margins(julia_file::String, r_file::String, me_type::String)
    println("\n" * "="^80)
    println("$me_type Comparison")
    println("="^80)

    # Load data
    julia_df = CSV.read(julia_file, DataFrame)
    r_df = CSV.read(r_file, DataFrame)

    println("\nLoaded $(nrow(julia_df)) Julia estimates, $(nrow(r_df)) R estimates")

    # Standardize variable names for matching (if variables exist)
    has_variables = false
    if "variable" in names(julia_df)
        julia_df.std_var = standardize_var_name.(julia_df.variable)
        has_variables = true
    elseif "term" in names(julia_df)
        julia_df.std_var = standardize_var_name.(julia_df.term)
        has_variables = true
    end

    if "factor" in names(r_df)
        r_df.std_var = standardize_var_name.(r_df.factor)
    elseif "variable" in names(r_df)
        r_df.std_var = standardize_var_name.(r_df.variable)
    elseif "term" in names(r_df)
        r_df.std_var = standardize_var_name.(r_df.term)
    end

    # Sort both by standardized variable name for alignment (if applicable)
    if has_variables && "std_var" in names(julia_df) && "std_var" in names(r_df)
        sort!(julia_df, :std_var)
        sort!(r_df, :std_var)
    # For APM/predictions at profiles, sort by profile columns
    elseif "socio4" in names(julia_df) && "at(socio4)" in names(r_df)
        # Rename R's at() columns for easier sorting
        rename!(r_df, "at(socio4)" => "socio4", "at(are_related_dists_a_inv)" => "are_related_dists_a_inv")
        sort!(julia_df, [:socio4, :are_related_dists_a_inv])
        sort!(r_df, [:socio4, :are_related_dists_a_inv])
    end

    # Check if counts match
    if nrow(julia_df) != nrow(r_df)
        println("⚠️  WARNING: Different number of estimates!")
        println("   Julia: $(nrow(julia_df)), R: $(nrow(r_df))")
        println("   Skipping comparison (different specifications)")
        return (NaN, NaN)
    end

    # Extract estimates and SEs with flexible column names
    # Julia columns: estimate/se or AME/SE
    # R columns: prob/SE or AME/SE or estimate/se
    julia_est_col = "estimate" in names(julia_df) ? :estimate : :AME
    julia_se_col = "se" in names(julia_df) ? :se : :SE

    if "Prediction" in names(r_df)
        r_est_col = :Prediction
    elseif "prob" in names(r_df)
        r_est_col = :prob
    elseif "estimate" in names(r_df)
        r_est_col = :estimate
    elseif "AME" in names(r_df)
        r_est_col = :AME
    else
        error("Cannot find estimate column in R data")
    end

    r_se_col = "SE" in names(r_df) ? :SE : "se" in names(r_df) ? :se : :std_error

    julia_est = julia_df[!, julia_est_col]
    julia_se = julia_df[!, julia_se_col]
    r_est = r_df[!, r_est_col]
    r_se = r_df[!, r_se_col]

    # Compute differences
    est_diff = julia_est .- r_est
    se_diff = julia_se .- r_se

    # Compute relative errors (avoiding division by zero)
    est_rel_error = abs.(est_diff) ./ (abs.(julia_est) .+ 1e-10)
    se_rel_error = abs.(se_diff) ./ (abs.(julia_se) .+ 1e-10)

    # Summary statistics
    println("\nEstimate Differences:")
    println("  Max absolute difference: ", @sprintf("%.2e", maximum(abs.(est_diff))))
    println("  Mean absolute difference: ", @sprintf("%.2e", mean(abs.(est_diff))))
    println("  Max relative error: ", @sprintf("%.4f%%", maximum(est_rel_error) * 100))
    println("  Mean relative error: ", @sprintf("%.4f%%", mean(est_rel_error) * 100))

    println("\nStandard Error Differences:")
    println("  Max absolute difference: ", @sprintf("%.2e", maximum(abs.(se_diff))))
    println("  Mean absolute difference: ", @sprintf("%.2e", mean(abs.(se_diff))))
    println("  Max relative error: ", @sprintf("%.2f%%", maximum(se_rel_error) * 100))
    println("  Mean relative error: ", @sprintf("%.2f%%", mean(se_rel_error) * 100))

    # Validation criteria
    max_est_rel_err = maximum(est_rel_error)
    max_se_rel_err = maximum(se_rel_error)

    if max_est_rel_err < 0.0001 && max_se_rel_err < 0.01
        println("\n✓✓✓ SUCCESS: $me_type results statistically equivalent!")
    elseif max_est_rel_err < 0.001 && max_se_rel_err < 0.05
        println("\n✓ PASS: $me_type results agree within tolerance")
    elseif max_est_rel_err < 0.01
        println("\n⚠️  PARTIAL: $me_type results mostly agree")
    else
        println("\n✗ FAILURE: $me_type results differ significantly")
    end

    # Show worst 5 mismatches if any significant differences
    if max_est_rel_err > 0.001 || max_se_rel_err > 0.05
        println("\nWorst 5 estimate mismatches:")
        worst_idx = sortperm(abs.(est_diff), rev=true)[1:min(5, length(est_diff))]
        for idx in worst_idx
            # Use std_var if available, otherwise use row number
            var_name = "std_var" in names(julia_df) ? julia_df.std_var[idx] : "Row_$idx"
            @printf("  %-30s  Julia: %10.6f  R: %10.6f  Diff: %.2e  RelErr: %.4f%%\n",
                    var_name, julia_est[idx], r_est[idx],
                    est_diff[idx], est_rel_error[idx] * 100)
        end
    end

    return (max_est_rel_err, max_se_rel_err)
end

# Compare all marginal effects types
results = Dict{String, Tuple{Float64, Float64}}()

# AME - Average Marginal Effects
if isfile("julia_ame.csv") && isfile("r_ame.csv")
    results["AME"] = compare_margins("julia_ame.csv", "r_ame.csv", "AME (Average Marginal Effects)")
end

# MEM - Marginal Effects at the Mean
if isfile("julia_mem.csv") && isfile("r_mem.csv")
    results["MEM"] = compare_margins("julia_mem.csv", "r_mem.csv", "MEM (Marginal Effects at Mean)")
end

# AAP - Average Adjusted Predictions
if isfile("julia_aap.csv") && isfile("r_aap.csv")
    results["AAP"] = compare_margins("julia_aap.csv", "r_aap.csv", "AAP (Average Adjusted Predictions)")
end

# APM - Adjusted Predictions at the Mean
if isfile("julia_apm.csv") && isfile("r_apm.csv")
    results["APM"] = compare_margins("julia_apm.csv", "r_apm.csv", "APM (Adjusted Predictions at Mean)")
end

# AME by age (if exists)
if isfile("julia_ame_age.csv") && isfile("r_ame_age.csv")
    results["AME_AGE"] = compare_margins("julia_ame_age.csv", "r_ame_age.csv", "AME by Age Groups")
end

# AME by scenario (if exists)
if isfile("julia_ame_scenario.csv") && isfile("r_ame_scenario.csv")
    results["AME_SCENARIO"] = compare_margins("julia_ame_scenario.csv", "r_ame_scenario.csv", "AME by Scenario")
end

# Overall summary
println("\n" * "="^80)
println("Overall Marginal Effects Validation Summary")
println("="^80)

let all_success = true, n_compared = 0
    for (me_type, (max_est_err, max_se_err)) in results
        if isnan(max_est_err)
            @printf("%-20s SKIPPED (different specifications)\n", me_type)
            continue
        end

        n_compared += 1
        status = if max_est_err < 0.0001 && max_se_err < 0.01
            "✓ SUCCESS"
        elseif max_est_err < 0.001 && max_se_err < 0.05
            "✓ PASS"
        elseif max_est_err < 0.01
            "⚠️  PARTIAL"
        else
            all_success = false
            "✗ FAILURE"
        end
        @printf("%-20s %s  (Est: %.4f%%, SE: %.2f%%)\n",
                me_type, status, max_est_err * 100, max_se_err * 100)
    end

    if n_compared == 0
        println("\n⚠️  No marginal effects could be compared")
    elseif all_success
        println("\n✓✓✓ ALL COMPARED MARGINAL EFFECTS VALIDATED!")
        println("    Julia and R produce statistically equivalent marginal effects.")
        if length(results) > n_compared
            println("    ($(length(results) - n_compared) types skipped due to different specifications)")
        end
    else
        println("\n⚠️  Some marginal effects show differences - review detailed output above.")
    end
end

println("\n" * "="^80)
