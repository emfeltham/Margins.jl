#!/usr/bin/env julia

# Test to verify interaction terms work correctly with the categorical solution

using DataFrames, GLM, CategoricalArrays, Tables
push!(LOAD_PATH, "src")
using Margins

println("=== Verifying Interaction Terms Work Correctly ===")

# Create test data with strong interaction
df = DataFrame(
    y = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
    x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    group = categorical(["A", "B", "A", "B", "A", "B", "A", "B"])
)

# Fit model with strong interaction: y = β₀ + β₁x + β₂group_B + β₃(x*group_B)
model = lm(@formula(y ~ x * group), df)
println("✅ Model fitted")
println("  Coefficients: $(round.(coef(model), digits=4))")

# Manual verification of interaction effect
β = coef(model)
println("  Expected marginal effect at group=A: $(round(β[2], digits=4)) (β_x)")
println("  Expected marginal effect at group=B: $(round(β[2] + β[4], digits=4)) (β_x + β_interaction)")

# Test with explicit profile specifications
println("\nTesting profile marginal effects...")

try
    # Effect at group=A
    result_A = profile_margins(model, df; 
                              at=Dict(:group => "A"),
                              type=:effects, 
                              vars=[:x])
    
    # Effect at group=B  
    result_B = profile_margins(model, df;
                              at=Dict(:group => "B"),
                              type=:effects,
                              vars=[:x])
    
    df_A = DataFrame(result_A)
    df_B = DataFrame(result_B)
    
    effect_A = df_A.estimate[1]
    effect_B = df_B.estimate[1]
    
    println("✅ Profile margins computed")
    println("  Computed effect at group=A: $(round(effect_A, digits=4))")
    println("  Computed effect at group=B: $(round(effect_B, digits=4))")
    println("  Difference: $(round(effect_B - effect_A, digits=4))")
    
    # Verify against expected values
    expected_A = β[2]
    expected_B = β[2] + β[4]
    
    println("\nVerification:")
    println("  Expected A: $(round(expected_A, digits=4)), Got: $(round(effect_A, digits=4))")
    println("  Expected B: $(round(expected_B, digits=4)), Got: $(round(effect_B, digits=4))")
    
    if abs(effect_A - expected_A) < 0.001 && abs(effect_B - expected_B) < 0.001
        println("✅ Interaction effects are statistically correct!")
    else
        println("❌ Interaction effects don't match expected values")
    end
    
catch e
    println("❌ Error: $e")
end

println("\n=== Test Complete ===")