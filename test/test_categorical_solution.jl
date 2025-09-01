#!/usr/bin/env julia

# Test the categorical solution in Margins.jl

using DataFrames, GLM, CategoricalArrays, Tables
push!(LOAD_PATH, "src")
using Margins

println("=== Testing Categorical Solution in Margins.jl ===")

# Create test data with interaction between continuous and categorical variables
df = DataFrame(
    y = [1.2, 2.1, 1.8, 3.2, 2.5, 1.9, 3.8, 2.8, 2.1, 3.5],
    x = [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 3.5, 2.8, 2.0, 3.2],
    group = categorical(["A", "B", "A", "A", "B", "A", "B", "B", "A", "B"])
)

# Fit model with interaction term (this creates the categorical dependency)
model = lm(@formula(y ~ x * group), df)
println("✅ Model fitted with interaction term")
println("  Coefficients: $(coef(model))")

# Test 1: Profile marginal effects with categorical override
println("\nTest 1: Profile marginal effects with categorical override...")
try
    # This should work with the new solution
    result = profile_margins(model, df; 
                           at=Dict(:group => "A", :x => 2.0),
                           type=:effects, 
                           vars=[:x])
    
    println("✅ Profile margins computed successfully")
    df_result = DataFrame(result)
    println("  Results:")
    for row in eachrow(df_result)
        println("    $(row.term): $(round(row.estimate, digits=4)) ± $(round(row.se, digits=4))")
    end
catch e
    println("❌ Error: $e")
    println("  This should NOT happen with the new solution")
end

# Test 2: Profile marginal effects with different categorical levels
println("\nTest 2: Comparing effects at different categorical levels...")
try
    # Effect at group=A
    result_A = profile_margins(model, df; 
                              at=Dict(:group => "A", :x => 2.0),
                              type=:effects, 
                              vars=[:x])
    
    # Effect at group=B  
    result_B = profile_margins(model, df;
                              at=Dict(:group => "B", :x => 2.0),
                              type=:effects,
                              vars=[:x])
    
    println("✅ Profile margins computed for both groups")
    
    df_A = DataFrame(result_A)
    df_B = DataFrame(result_B)
    
    effect_A = df_A.estimate[1]
    effect_B = df_B.estimate[1]
    
    println("  Marginal effect of x at group=A: $(round(effect_A, digits=4))")
    println("  Marginal effect of x at group=B: $(round(effect_B, digits=4))")
    println("  Difference: $(round(effect_B - effect_A, digits=4))")
    
    if abs(effect_A - effect_B) > 0.001
        println("✅ Effects differ by categorical context (expected with interaction)")
    else
        println("⚠️  Effects are similar (unexpected with interaction)")
    end
    
catch e
    println("❌ Error: $e")
end

println("\n=== Test Complete ===")