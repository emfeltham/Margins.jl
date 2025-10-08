"""
Golden Reference Generation for Margins Migration

This script generates reference outputs from the CURRENT Margins.jl implementation
before migrating to FormulaCompiler primitives. These golden references will be
used to validate correctness after the migration.

Run this BEFORE making any changes to the Margins.jl codebase.

Usage:
    julia --project=/Users/emf/.julia/dev/Margins test/support/generate_golden_references.jl
"""

using Pkg
Pkg.activate("/Users/emf/.julia/dev/Margins")

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins
using JLD2
using Statistics

println("="^80)
println("GOLDEN REFERENCE GENERATION")
println("="^80)
println()
println("This script generates reference outputs from the CURRENT implementation")
println("Run BEFORE migrating to FormulaCompiler primitives")
println()

# Create fixtures directory
fixtures_dir = joinpath(@__DIR__, "fixtures")
mkpath(fixtures_dir)

# Set seed for reproducibility
Random.seed!(42)

#=============================================================================
Test Case 1: Simple Continuous Effects - OLS
=============================================================================#
println("Test 1: OLS - Simple continuous effects")

n = 1000
df1 = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    x3 = randn(n)
)

model1 = lm(@formula(y ~ x1 + x2 + x3), df1)

# Linear scale (identity link)
result1_linear = population_margins(model1, df1; type=:effects, vars=[:x1, :x2], scale=:link)

jldsave(joinpath(fixtures_dir, "golden_ols_simple.jld2");
    estimates = result1_linear.estimates,
    standard_errors = result1_linear.standard_errors,
    vcov = result1_linear.vcov,
    variables = result1_linear.variables
)

println("  ✓ Saved: golden_ols_simple.jld2")

#=============================================================================
Test Case 2: Continuous Effects with Interactions - OLS
=============================================================================#
println("Test 2: OLS - Interactions")

df2 = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    x3 = randn(n)
)

model2 = lm(@formula(y ~ x1 * x2 + x3), df2)

result2_linear = population_margins(model2, df2; type=:effects, vars=[:x1, :x2], scale=:link)

jldsave(joinpath(fixtures_dir, "golden_ols_interactions.jld2");
    estimates = result2_linear.estimates,
    standard_errors = result2_linear.standard_errors,
    vcov = result2_linear.vcov,
    variables = result2_linear.variables
)

println("  ✓ Saved: golden_ols_interactions.jld2")

#=============================================================================
Test Case 3: Logistic Regression - Link and Response Scales
=============================================================================#
println("Test 3: Logit - Link and response scales")

df3 = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n),
    x3 = randn(n)
)

model3 = glm(@formula(y ~ x1 + x2 + x3), df3, Binomial(), LogitLink())

result3_link = population_margins(model3, df3; type=:effects, vars=[:x1, :x2, :x3], scale=:link)
result3_response = population_margins(model3, df3; type=:effects, vars=[:x1, :x2, :x3], scale=:response)

jldsave(joinpath(fixtures_dir, "golden_logit_scales.jld2");
    estimates_link = result3_link.estimates,
    se_link = result3_link.standard_errors,
    vcov_link = result3_link.vcov,
    estimates_response = result3_response.estimates,
    se_response = result3_response.standard_errors,
    vcov_response = result3_response.vcov,
    variables = result3_link.variables
)

println("  ✓ Saved: golden_logit_scales.jld2")

#=============================================================================
Test Case 4: Logistic Regression with Interactions
=============================================================================#
println("Test 4: Logit - Interactions")

df4 = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n),
    z = randn(n)
)

model4 = glm(@formula(y ~ x1 * x2 + z), df4, Binomial(), LogitLink())

result4_link = population_margins(model4, df4; type=:effects, vars=[:x1, :x2], scale=:link)
result4_response = population_margins(model4, df4; type=:effects, vars=[:x1, :x2], scale=:response)

jldsave(joinpath(fixtures_dir, "golden_logit_interactions.jld2");
    estimates_link = result4_link.estimates,
    se_link = result4_link.standard_errors,
    vcov_link = result4_link.vcov,
    estimates_response = result4_response.estimates,
    se_response = result4_response.standard_errors,
    vcov_response = result4_response.vcov,
    variables = result4_link.variables
)

println("  ✓ Saved: golden_logit_interactions.jld2")

#=============================================================================
Test Case 5: Probit Model
=============================================================================#
println("Test 5: Probit - Link and response scales")

df5 = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n)
)

model5 = glm(@formula(y ~ x1 + x2), df5, Binomial(), ProbitLink())

result5_link = population_margins(model5, df5; type=:effects, vars=[:x1, :x2], scale=:link)
result5_response = population_margins(model5, df5; type=:effects, vars=[:x1, :x2], scale=:response)

jldsave(joinpath(fixtures_dir, "golden_probit.jld2");
    estimates_link = result5_link.estimates,
    se_link = result5_link.standard_errors,
    vcov_link = result5_link.vcov,
    estimates_response = result5_response.estimates,
    se_response = result5_response.standard_errors,
    vcov_response = result5_response.vcov,
    variables = result5_link.variables
)

println("  ✓ Saved: golden_probit.jld2")

#=============================================================================
Test Case 6: Weighted Regression
=============================================================================#
println("Test 6: OLS - Weighted regression")

df6 = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    wts = abs.(randn(n)) .+ 0.1
)

model6 = lm(@formula(y ~ x1 + x2), df6, wts=df6.wts)

result6 = population_margins(model6, df6; type=:effects, vars=[:x1, :x2], scale=:link)

jldsave(joinpath(fixtures_dir, "golden_weighted.jld2");
    estimates = result6.estimates,
    standard_errors = result6.standard_errors,
    vcov = result6.vcov,
    variables = result6.variables
)

println("  ✓ Saved: golden_weighted.jld2")

#=============================================================================
Test Case 7: Single Variable (edge case)
=============================================================================#
println("Test 7: Single variable edge case")

df7 = DataFrame(
    y = randn(n),
    x = randn(n),
    z = randn(n)
)

model7 = lm(@formula(y ~ x + z), df7)

result7 = population_margins(model7, df7; type=:effects, vars=[:x], scale=:link)

jldsave(joinpath(fixtures_dir, "golden_single_var.jld2");
    estimates = result7.estimates,
    standard_errors = result7.standard_errors,
    vcov = result7.vcov,
    variables = result7.variables
)

println("  ✓ Saved: golden_single_var.jld2")

#=============================================================================
Test Case 8: Complex Formula with Transformations
=============================================================================#
println("Test 8: Complex formula with transformations")

df8 = DataFrame(
    y = randn(n),
    x1 = abs.(randn(n)) .+ 0.1,  # Positive for log
    x2 = randn(n),
    x3 = randn(n)
)

model8 = lm(@formula(y ~ log(x1) + x2 * x3), df8)

result8 = population_margins(model8, df8; type=:effects, vars=[:x1, :x2], scale=:link)

jldsave(joinpath(fixtures_dir, "golden_transformations.jld2");
    estimates = result8.estimates,
    standard_errors = result8.standard_errors,
    vcov = result8.vcov,
    variables = result8.variables
)

println("  ✓ Saved: golden_transformations.jld2")

#=============================================================================
Test Case 9: Large Dataset (performance validation)
=============================================================================#
println("Test 9: Large dataset (10K rows)")

n_large = 10000
df9 = DataFrame(
    y = rand(Bool, n_large),
    x1 = randn(n_large),
    x2 = randn(n_large),
    x3 = randn(n_large)
)

model9 = glm(@formula(y ~ x1 * x2 + x3), df9, Binomial(), LogitLink())

result9_link = population_margins(model9, df9; type=:effects, vars=[:x1, :x2, :x3], scale=:link)
result9_response = population_margins(model9, df9; type=:effects, vars=[:x1, :x2, :x3], scale=:response)

jldsave(joinpath(fixtures_dir, "golden_large_dataset.jld2");
    estimates_link = result9_link.estimates,
    se_link = result9_link.standard_errors,
    vcov_link = result9_link.vcov,
    estimates_response = result9_response.estimates,
    se_response = result9_response.standard_errors,
    vcov_response = result9_response.vcov,
    variables = result9_link.variables
)

println("  ✓ Saved: golden_large_dataset.jld2")

#=============================================================================
Summary
=============================================================================#
println()
println("="^80)
println("GOLDEN REFERENCE GENERATION COMPLETE")
println("="^80)
println()
println("Generated 9 test cases covering:")
println("  ✓ OLS (identity link)")
println("  ✓ Logit (logistic link)")
println("  ✓ Probit (probit link)")
println("  ✓ Interactions")
println("  ✓ Transformations")
println("  ✓ Weighted regression")
println("  ✓ Single variable edge case")
println("  ✓ Large datasets (10K rows)")
println("  ✓ Both link and response scales")
println()
println("Files saved to: $fixtures_dir")
println()
println("Next steps:")
println("  1. Implement FormulaCompiler migration")
println("  2. Run validation script: test/validate_golden_references.jl")
println("  3. Verify rtol=1e-12 agreement")
println()