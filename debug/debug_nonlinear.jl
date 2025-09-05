#!/usr/bin/env julia

# Test with logistic model where marginal effects vary by group

using Margins, GLM, DataFrames, CategoricalArrays, Statistics

# Create binary outcome data where marginal effects should differ
n = 100
data = DataFrame(
    # Create data where different groups have different baseline probabilities
    x1 = [repeat([0.0], 25); repeat([1.0], 25); repeat([2.0], 25); repeat([3.0], 25)],
    education = CategoricalArray([repeat(["HS"], 50); repeat(["College"], 50)]),
    # Binary outcome with different patterns by education
    y = [[rand() < 0.2 ? 1 : 0 for _ in 1:25];      # HS, x1=0: low prob
         [rand() < 0.8 ? 1 : 0 for _ in 1:25];      # HS, x1=1: high prob  
         [rand() < 0.1 ? 1 : 0 for _ in 1:25];      # College, x1=2: very low prob
         [rand() < 0.9 ? 1 : 0 for _ in 1:25]]       # College, x1=3: very high prob
)

println("Data summary:")
println("Overall: ", nrow(data), " observations")
for edu in ["HS", "College"]
    for x1_val in [0.0, 1.0, 2.0, 3.0]
        subset = data[(data.education .== edu) .& (data.x1 .== x1_val), :]
        if nrow(subset) > 0
            prob = mean(subset.y)
            println("  $edu, x1=$x1_val: ", nrow(subset), " obs, P(y=1) = ", round(prob, digits=3))
        end
    end
end

# Fit logistic model
model_logit = glm(@formula(y ~ x1 + education), data, Binomial(), LogitLink())
println("\nLogistic model coefficients: ", coef(model_logit))

# Test marginal effects (should differ between groups due to nonlinear link function)
println("\n" * repeat("=", 40))
println("LOGISTIC MODEL MARGINAL EFFECTS")
println(repeat("=", 40))

base_result = population_margins(model_logit, data; type=:effects, vars=[:x1], target=:mu)
println("Base (overall) result:")
println(DataFrame(base_result))

grouped_result = population_margins(model_logit, data; type=:effects, vars=[:x1], groups=:education, target=:mu)
println("Grouped result:")
grouped_df = DataFrame(grouped_result)
println(grouped_df)

# Check if results are different
if nrow(grouped_df) >= 2
    college_result = grouped_df[grouped_df.at_education .== "College", :estimate][1]
    hs_result = grouped_df[grouped_df.at_education .== "HS", :estimate][1]
    
    println("\nComparison:")
    println("College marginal effect: ", college_result)
    println("HS marginal effect: ", hs_result)
    println("Difference: ", abs(college_result - hs_result))
    
    if abs(college_result - hs_result) > 1e-6
        println(" SUCCESS: Different marginal effects between groups!")
        println("This confirms that subgroup computation is working correctly.")
    else
        println("  WARNING: Marginal effects are still identical")
        println("This suggests there may still be an issue with subgroup computation")
    end
end