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
pop_ame = population_margins(m, df; type=:effects, vars=[:x, :z], scale=:response)
println(DataFrame(pop_ame))
println()

# Population marginal effects on link scale (linear predictor)
println("2. Population marginal effects - link scale (η):")
pop_ame_eta = population_margins(m, df; type=:effects, vars=[:x, :z], scale=:link)
println(DataFrame(pop_ame_eta))
println()

# Population predictions
println("3. Population average predictions:")
pop_pred = population_margins(m, df; type=:predictions, scale=:response)
println(DataFrame(pop_pred))
println()

println("=== PROFILE APPROACH ===")

# Profile marginal effects at sample means
println("4. Profile marginal effects at sample means (MEM):")
prof_ame = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x, :z], scale=:response)
println(DataFrame(prof_ame))
println()

# Profile predictions at specific scenarios  
println("5. Profile predictions at specific x values:")
prof_pred = profile_margins(m, df, cartesian_grid(x=[-2, 0, 2]); type=:predictions, scale=:response)
println(DataFrame(prof_pred))
println()

# Profile effects comparison across scenarios
println("6. Profile effects at different x and z combinations:")
prof_effects = profile_margins(m, df, cartesian_grid(x=[-1, 0, 1], z=[-1, 0, 1]); type=:effects, vars=[:x], scale=:response)
prof_df = DataFrame(prof_effects)
# Show first few rows with key columns (adjust column names based on actual output)
println("Effect of x at different x,z scenarios:")
println(first(prof_df, 6))
println()

println("=== API DEMONSTRATION ===")

println("7. Clean two-function API demonstration:")
println("Population vs Profile comparison for variable x:")

# Population approach
pop_demo = population_margins(m, df; type=:effects, vars=[:x], scale=:response)
pop_effect = DataFrame(pop_demo).estimate[1]

# Profile approach (at means)
prof_demo = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x], scale=:response)
prof_effect = DataFrame(prof_demo).estimate[1]

println("Population AME: ", round(pop_effect, digits=5))
println("Profile MEM:    ", round(prof_effect, digits=5))
println("Difference:     ", round(abs(pop_effect - prof_effect), digits=6))
println()

println("=== FRAMEWORK COMPARISON ===")
println("Population approach gives you the TRUE average effect across your sample.")
println("Profile approach evaluates at specific 'representative' cases.")
println()
println("For GLM example - Population AME for x: ", round(DataFrame(pop_ame).estimate[1], digits=5))
println("For linear models: Population ≈ Profile (small differences)")
println("For nonlinear models: Can differ substantially!")

println("\n=== CLEAN API SUMMARY ===")
println("Margins.jl: Two functions, clear concepts")
println("• population_margins() - average effects across your sample")
println("• profile_margins() - effects at specific covariate values")
println("• Direct mapping to statistical framework")
println("• type=:effects or :predictions determines output")
println()
println("=== ROBUST STANDARD ERRORS ===")
println("For robust/clustered standard errors, use CovarianceMatrices.jl:")
println()
println("# Install: using Pkg; Pkg.add(\"CovarianceMatrices\")")
println("# using CovarianceMatrices")
println()
println("# Robust standard errors:")
println("# robust_vcov = HC1(m)")
println("# robust_result = population_margins(m, df; type=:effects, vars=[:x], vcov=robust_vcov)")
println()
println("# Clustered standard errors:")
println("# clustered_vcov = CRVE1(m, df.cluster_var)")
println("# clustered_result = population_margins(m, df; type=:effects, vcov=clustered_vcov)")
println()
println("Note: Margins.jl uses delta-method for all marginal effect standard errors,")
println("properly accounting for the nonlinearity in GLM transformations.")
