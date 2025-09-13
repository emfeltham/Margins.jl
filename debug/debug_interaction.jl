#!/usr/bin/env julia

# Test with interaction model where marginal effects should differ between groups

using Margins, GLM, DataFrames, CategoricalArrays

# Create test data
n = 20
data = DataFrame(
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,  # HS: 1-10
         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],  # College: 11-20
    x1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # HS: all x1=1
          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],   # College: all x1=2
    education = CategoricalArray([repeat(["HS"], 10); repeat(["College"], 10)])
)

# Test with INTERACTION model
model_interaction = lm(@formula(y ~ x1 * education), data)
println("Interaction Model coefficients: ", coef(model_interaction))
println("Model formula: y ~ x1 * education")

# Test marginal effects
println("\n" * repeat("=", 40))
println("INTERACTION MODEL MARGINAL EFFECTS")
println(repeat("=", 40))

# For interaction model, marginal effect of x1 should be:
# ∂y/∂x1 = β₁ + β₃ * education_dummy
# Where β₃ is the interaction coefficient

base_result = population_margins(model_interaction, data; type=:effects, vars=[:x1])
println("Base (overall) result:")
println(DataFrame(base_result))

grouped_result = population_margins(model_interaction, data; type=:effects, vars=[:x1], groups=:education)
println("Grouped result:")
grouped_df = DataFrame(grouped_result)
println(grouped_df)

# Manual calculation for verification
β = coef(model_interaction)
println("\nManual calculation:")
println("β₀ (intercept) = ", β[1])
println("β₁ (x1) = ", β[2]) 
println("β₂ (educationHS) = ", β[3])
println("β₃ (x1 & educationHS) = ", β[4])

println("\nExpected marginal effects:")
println("For College: ∂y/∂x1 = β₁ = ", β[2])
println("For HS: ∂y/∂x1 = β₁ + β₃ = ", β[2] + β[4])

# Check if our grouped results match these expectations
if nrow(grouped_df) >= 2
    college_result = grouped_df[grouped_df.at_education .== "College", :estimate][1]
    hs_result = grouped_df[grouped_df.at_education .== "HS", :estimate][1]
    
    println("\nActual results from grouping:")
    println("College: ", college_result)
    println("HS: ", hs_result)
    
    println("\nDifference check:")
    println("Expected College: ", β[2], " vs Actual: ", college_result, " → ", abs(β[2] - college_result) < 1e-10 ? " MATCH" : " DIFFERENT")
    println("Expected HS: ", β[2] + β[4], " vs Actual: ", hs_result, " → ", abs(β[2] + β[4] - hs_result) < 1e-10 ? " MATCH" : " DIFFERENT")
end