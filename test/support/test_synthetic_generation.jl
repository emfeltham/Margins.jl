#!/usr/bin/env julia

# Test script to verify synthetic data generation
using DataFrames
using GLM
using StatsModels
import CSV

# Include the generator
include("generate_large_synthetic_data.jl")

println("Testing synthetic data generation...")

# Generate a small test dataset first (1000 rows)
println("Generating small test dataset (1000 rows)...")
test_df = generate_synthetic_dataset(1000)

println("Generated dataset dimensions: $(size(test_df))")
println("Column names match original: ", all(in(names(test_df)), ["response", "socio4", "dists_p_inv", "are_related_dists_a_inv", "dists_a_inv", "num_common_nbs"]))

# Test that the model formula works with synthetic data
println("\nTesting model fitting with synthetic data...")

try
    fx = @formula(response ~
        socio4 +
        (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
        !socio4 & dists_a_inv +
        (num_common_nbs & (dists_a_inv <= inv(2))) & (1 + are_related_dists_a_inv + dists_p_inv) +
        # individual variables
        (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
        same_building + population +
        hhi_religion + hhi_indigenous +
        coffee_cultivation + market +
        relation) & (1 + socio4 + are_related_dists_a_inv) +
        # tie variables
        (
            degree_a_mean + degree_h +
            age_a_mean + age_h * age_h_nb_1_socio +
            schoolyears_a_mean + schoolyears_h * schoolyears_h_nb_1_socio +
            man_x * man_x_mixed_nb_1 +
            wealth_d1_4_a_mean + wealth_d1_4_h * wealth_d1_4_h_nb_1_socio +
            isindigenous_x * isindigenous_homop_nb_1 +
            religion_c_x * religion_homop_nb_1
        ) & (1 + socio4 + are_related_dists_a_inv) +
        religion_c_x & hhi_religion +
        isindigenous_x & hhi_indigenous
    )

    # Try to fit the model - this will test if all required columns exist and are properly formatted
    model = fit(GeneralizedLinearModel, fx, test_df, Bernoulli(), LogitLink())

    println("✓ Model fitting successful!")
    println("Model summary:")
    println("  - Number of coefficients: $(length(coef(model)))")
    println("  - Deviance: $(round(deviance(model), digits=2))")

catch e
    println("✗ Model fitting failed: $e")

    # Diagnose the issue
    println("\nDiagnosing column types:")
    for col in ["response", "socio4", "man_p", "religion_c_p", "relation", "man_x", "religion_c_x", "isindigenous_x"]
        if col in names(test_df)
            println("  $col: $(eltype(test_df[!, col])) - Sample: $(first(test_df[!, col], 3))")
        else
            println("  $col: MISSING")
        end
    end
end

println("\n" * "="^60)
println("To generate the full 620K dataset, run:")
println("julia -e 'include(\"test/support/generate_large_synthetic_data.jl\"); save_synthetic_dataset()'")