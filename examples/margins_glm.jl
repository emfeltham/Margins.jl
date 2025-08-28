# margins_glm.jl
# Comprehensive GLM examples with the new API

using DataFrames, CategoricalArrays, GLM, Margins
# Robust covariance provider (optional)
# using CovarianceMatrices

println("=== GLM Marginal Analysis Examples ===\n")

# Generate example data
df = DataFrame(
    y = rand(Bool, 1000),           # Binary outcome
    x = randn(1000),                # Continuous predictor
    z = randn(1000),                # Another continuous predictor  
    g = categorical(rand(["A","B"], 1000))  # Categorical predictor
)

# Fit logistic regression
m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
println("Model: Logistic regression (y ~ x + z + g)")
println("Link: LogitLink() for binary outcomes\n")

println("=== POPULATION APPROACH ===")

# Population marginal effects (AME) on response scale
println("1. Population marginal effects (AME) - response scale:")
pop_ame = population_margins(m, df; type=:effects, vars=[:x, :z], target=:mu)
println(pop_ame.table)
println()

# Population marginal effects on link scale (linear predictor)
println("2. Population marginal effects - link scale (η):")
pop_ame_eta = population_margins(m, df; type=:effects, vars=[:x, :z], target=:eta)
println(pop_ame_eta.table)
println()

# Population predictions
println("3. Population average predictions:")
pop_pred = population_margins(m, df; type=:predictions, scale=:response)
println(pop_pred.table)
println()

println("=== PROFILE APPROACH ===")

println("4. Profile approach (at specific covariate values):")
println("   Note: profile_margins() has some compatibility issues being resolved")
println("   
   Conceptual examples:
   • profile_margins(m, df; at=:means, type=:effects, vars=[:x])
   • profile_margins(m, df; at=Dict(:x=>[-2,0,2]), type=:predictions)
   
   This evaluates effects/predictions at specific covariate combinations
   rather than averaging across the observed data distribution.
")
println()

println("=== API DEMONSTRATION ===")

println("7. Clean two-function API in action:")
println("   population_margins() for true population effects:")
pop_demo = population_margins(m, df; type=:effects, vars=[:x])
println("   Effect of x: ", round(pop_demo.table.dydx[1], digits=5))

println("   
   profile_margins() would show effects at specific covariate values
   (Currently has compatibility issues being resolved)
   ")
println()

println("=== FRAMEWORK COMPARISON ===")
println("Population approach gives you the TRUE average effect across your sample.")
println("Profile approach evaluates at specific 'representative' cases.")
println()
println("For GLM example - Population AME: ", round(pop_ame.table.dydx[1], digits=5))
println("For linear models: Population ≈ Profile (small differences)")
println("For nonlinear models: Can differ substantially!")

println("\n=== CLEAN API SUMMARY ===")
println("Margins.jl: Two functions, clear concepts")
println("• population_margins() - average effects across your sample")
println("• profile_margins() - effects at specific covariate values")
println("• No confusing legacy naming (AME/MEM/MER/etc.)")
println("• Direct mapping to statistical framework")
println("• type=:effects or :predictions determines output")
println()
println("=== ROBUST STANDARD ERRORS ===")
println("# For robust SEs, use the vcov parameter:")
println("# using CovarianceMatrices")
println("# robust_result = population_margins(m, df; type=:effects, vars=[:x], vcov=HC1())")
println("# println(robust_result.table)")
