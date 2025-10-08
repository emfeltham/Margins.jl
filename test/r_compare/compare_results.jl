# compare_results.jl
# Compare Julia and R results to verify statistical agreement

using DataFrames, CSV
using Printf
using Statistics

println("="^80)
println("R Comparison Study - Results Validation")
println("="^80)

# Load coefficient files
println("\nLoading coefficient estimates...")
julia_coef = CSV.read("julia_coefficients.csv", DataFrame)
r_coef = CSV.read("r_coefficients.csv", DataFrame)

println("Julia coefficients: $(nrow(julia_coef)) terms")
println("R coefficients: $(nrow(r_coef)) terms")

# Check if coefficient counts match
if nrow(julia_coef) != nrow(r_coef)
    println("\n⚠️  WARNING: Different number of coefficients!")
    println("   Julia: $(nrow(julia_coef)), R: $(nrow(r_coef))")
    println("   This suggests the models are not identical.")
else
    println("✓ Coefficient counts match")
end

# Standardize term names to enable comparison
# Julia: '&' for interactions, 'var: level' for categoricals
# R: ':' for interactions, 'varlevel' for categoricals, 'TRUE' suffix for booleans
println("\nStandardizing term names...")

function standardize_term_name(term::String)
    # Start with original term
    std = term

    # Julia → Standard conversions
    # Replace Julia's ' & ' with ':'
    std = replace(std, " & " => ":")
    # Remove spaces after colons in categorical terms (e.g., "relation: work" → "relation:work")
    std = replace(std, r": " => ":")

    # R → Standard conversions
    # Remove 'TRUE' suffix for boolean variables
    std = replace(std, r"TRUE$" => "")
    std = replace(std, r"TRUE:" => ":")
    std = replace(std, r":TRUE" => ":")

    return std
end

# Standardize both sets of names
julia_coef.std_term = standardize_term_name.(julia_coef.term)
r_coef.std_term = standardize_term_name.(r_coef.term)

# Check if standardized names match
println("\nChecking standardized term name matches...")
mismatches = []
for i in 1:min(nrow(julia_coef), nrow(r_coef))
    if julia_coef.std_term[i] != r_coef.std_term[i]
        push!(mismatches, (i, julia_coef.std_term[i], r_coef.std_term[i]))
    end
end

if length(mismatches) > 0
    println("\n⚠️  WARNING: $(length(mismatches)) term names don't match after standardization!")
    println("   First 10 mismatches:")
    for (i, j_term, r_term) in mismatches[1:min(10, length(mismatches))]
        println("   Row $i: Julia='$j_term' vs R='$r_term'")
    end
    println("\n   Proceeding with comparison assuming same model order...")
else
    println("✓ All standardized term names match")
end

# Compare estimates and standard errors
println("\n" * "="^80)
println("Coefficient Comparison")
println("="^80)

# Compute differences
est_diff = julia_coef.estimate .- r_coef.estimate
se_diff = julia_coef.std_error .- r_coef.std_error

# Compute relative errors (avoiding division by zero)
est_rel_error = abs.(est_diff) ./ (abs.(julia_coef.estimate) .+ 1e-10)
se_rel_error = abs.(se_diff) ./ (abs.(julia_coef.std_error) .+ 1e-10)

# Summary statistics
println("\nEstimate Differences:")
println("  Max absolute difference: ", @sprintf("%.2e", maximum(abs.(est_diff))))
println("  Mean absolute difference: ", @sprintf("%.2e", mean(abs.(est_diff))))
println("  Max relative error: ", @sprintf("%.2e", maximum(est_rel_error)))
println("  Mean relative error: ", @sprintf("%.2e", mean(est_rel_error)))

println("\nStandard Error Differences:")
println("  Max absolute difference: ", @sprintf("%.2e", maximum(abs.(se_diff))))
println("  Mean absolute difference: ", @sprintf("%.2e", mean(abs.(se_diff))))
println("  Max relative error: ", @sprintf("%.2e", maximum(se_rel_error)))
println("  Mean relative error: ", @sprintf("%.2e", mean(se_rel_error)))

# Check against tolerance thresholds
# These are realistic thresholds for comparing numerical optimization results
# across different implementations (Julia vs R)
est_tol = 1e-4  # Absolute tolerance for estimates (0.0001)
se_tol = 1e-3   # Absolute tolerance for SEs (0.001)
rel_tol = 1e-3  # Relative tolerance (0.1%)

println("\n" * "="^80)
println("Validation Against Thresholds")
println("="^80)

n_failing_est_abs = sum(abs.(est_diff) .> est_tol)
n_failing_se_abs = sum(abs.(se_diff) .> se_tol)
n_failing_est_rel = sum(est_rel_error .> rel_tol)
n_failing_se_rel = sum(se_rel_error .> rel_tol)

if n_failing_est_abs == 0
    println("✓ All estimates agree within absolute tolerance ($(est_tol))")
else
    println("✗ $(n_failing_est_abs) estimates exceed absolute tolerance")
end

if n_failing_est_rel == 0
    println("✓ All estimates agree within relative tolerance ($(@sprintf("%.2f", rel_tol * 100))%)")
else
    println("✗ $(n_failing_est_rel) estimates exceed relative tolerance")
end

if n_failing_se_abs == 0
    println("✓ All standard errors agree within absolute tolerance ($(se_tol))")
else
    println("✗ $(n_failing_se_abs) standard errors exceed absolute tolerance")
end

if n_failing_se_rel == 0
    println("✓ All standard errors agree within relative tolerance ($(@sprintf("%.2f", rel_tol * 100))%)")
else
    println("✗ $(n_failing_se_rel) standard errors exceed relative tolerance")
end

# Additional quality metrics
println("\n" * "="^80)
println("Statistical Quality Assessment")
println("="^80)
println("\nCoefficient Agreement:")
println("  Max relative error: ", @sprintf("%.4f%%", maximum(est_rel_error) * 100))
println("  Mean relative error: ", @sprintf("%.4f%%", mean(est_rel_error) * 100))
println("  99th percentile relative error: ", @sprintf("%.4f%%", quantile(est_rel_error, 0.99) * 100))

println("\nStandard Error Agreement:")
println("  Max relative error: ", @sprintf("%.2f%%", maximum(se_rel_error) * 100))
println("  Mean relative error: ", @sprintf("%.4f%%", mean(se_rel_error) * 100))
println("  99th percentile relative error: ", @sprintf("%.2f%%", quantile(se_rel_error, 0.99) * 100))

# Show worst mismatches if any
if n_failing_est_abs > 0 || n_failing_se_abs > 0
    println("\n" * "="^80)
    println("Worst Mismatches (top 10)")
    println("="^80)

    # Create comparison dataframe
    comp_df = DataFrame(
        term = julia_coef.term,
        julia_est = julia_coef.estimate,
        r_est = r_coef.estimate,
        est_diff = est_diff,
        est_rel_err = est_rel_error,
        julia_se = julia_coef.std_error,
        r_se = r_coef.std_error,
        se_diff = se_diff,
        se_rel_err = se_rel_error
    )

    # Sort by absolute estimate difference
    sort!(comp_df, :est_diff, by=abs, rev=true)

    println("\nTop 10 estimate differences:")
    for i in 1:min(10, nrow(comp_df))
        row = comp_df[i, :]
        println(@sprintf("  %-30s  Julia: %12.6e  R: %12.6e  Diff: %12.6e  RelErr: %.2e",
                        row.term, row.julia_est, row.r_est, row.est_diff, row.est_rel_err))
    end

    # Sort by absolute SE difference
    sort!(comp_df, :se_diff, by=abs, rev=true)

    println("\nTop 10 standard error differences:")
    for i in 1:min(10, nrow(comp_df))
        row = comp_df[i, :]
        println(@sprintf("  %-30s  Julia: %12.6e  R: %12.6e  Diff: %12.6e  RelErr: %.2e",
                        row.term, row.julia_se, row.r_se, row.se_diff, row.se_rel_err))
    end
end

# Overall verdict
println("\n" * "="^80)
println("Overall Validation Result")
println("="^80)

# Success criteria based on relative errors (robust to scale differences)
max_coef_rel_err = maximum(est_rel_error)
mean_coef_rel_err = mean(est_rel_error)
max_se_rel_err = maximum(se_rel_error)

if max_coef_rel_err < 0.0001 && max_se_rel_err < 0.01
    println("\n✓✓✓ SUCCESS: Julia and R models produce statistically equivalent results!")
    println("    Maximum coefficient relative error: $(@sprintf("%.4f%%", max_coef_rel_err * 100))")
    println("    Maximum SE relative error: $(@sprintf("%.2f%%", max_se_rel_err * 100))")
    println("    Differences are within numerical precision expectations.")
    println("    Statistical validity confirmed.")
elseif max_coef_rel_err < 0.001 && max_se_rel_err < 0.05
    println("\n✓ PASS: Julia and R models agree within acceptable tolerance.")
    println("   Maximum coefficient relative error: $(@sprintf("%.4f%%", max_coef_rel_err * 100))")
    println("   Maximum SE relative error: $(@sprintf("%.2f%%", max_se_rel_err * 100))")
    println("   Results are suitable for practical use.")
elseif max_coef_rel_err < 0.01
    println("\n⚠️  PARTIAL SUCCESS: Models mostly agree with minor numerical differences.")
    println("   Maximum coefficient relative error: $(@sprintf("%.4f%%", max_coef_rel_err * 100))")
    println("   Maximum SE relative error: $(@sprintf("%.2f%%", max_se_rel_err * 100))")
    println("   Review worst mismatches above.")
else
    println("\n✗✗✗ FAILURE: Models produce different results!")
    println("   Maximum coefficient relative error: $(@sprintf("%.4f%%", max_coef_rel_err * 100))")
    println("   Possible causes:")
    println("   - Different model formulas")
    println("   - Different categorical reference levels")
    println("   - Different data processing")
    println("   Review the worst mismatches above and check factor levels.")
end

println("\n" * "="^80)
