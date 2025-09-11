# basic_usage.jl
# Examples showing the new population_margins() and profile_margins() API

using Margins
using DataFrames, GLM, StatsModels, CategoricalArrays
using Random

# Create example data
Random.seed!(06515)

df = DataFrame(
    wage = exp.(randn(1000) .+ 5),        # Log-normal wages
    education = rand(8:16, 1000) * 1.0,   # Years of education (as Float64)
    experience = rand(0:40, 1000) * 1.0,  # Years of experience (as Float64) 
    gender = categorical(rand(["Male", "Female"], 1000)),
    urban = Float64.(rand([0, 1], 1000))   # Urban indicator (as Float64)
)

# Note: Converting data to Float64 for FormulaCompiler compatibility

println("=== Margins.jl: Population vs Profile Framework ===\n")

# Fit a wage model
wage_model = lm(@formula(log(wage) ~ education + experience + experience^2 + gender + urban), df)

println("Model fitted: log(wage) ~ education + experience + experience² + gender + urban\n")

println("=== POPULATION APPROACH (Average over observed data) ===")

# Example 1: Population marginal effects (AME equivalent)
println("1. Population marginal effects for education:")
pop_education = population_margins(wage_model, df; type=:effects, vars=[:education])
println(DataFrame(pop_education))
println()

# Example 2: Population effects for multiple variables
println("2. Population marginal effects for multiple variables:")
pop_multiple = population_margins(wage_model, df; type=:effects, vars=[:education, :experience])
println(DataFrame(pop_multiple))
println()

# Example 3: Population predictions (APE equivalent)
println("3. Population average predictions:")
pop_pred = population_margins(wage_model, df; type=:predictions, scale=:response)
println(DataFrame(pop_pred))
println()

println("=== PROFILE APPROACH (At specific covariate values) ===")

# Example 4: Profile marginal effects at sample means (MEM)
println("4. Profile marginal effects at sample means:")
prof_education = profile_margins(wage_model, df, means_grid(df); type=:effects, vars=[:education])
println(DataFrame(prof_education))
println()

# Example 5: Profile predictions at specific scenarios
println("5. Profile predictions at specific education levels:")
prof_pred = profile_margins(wage_model, df, cartesian_grid(education=[12.0, 16.0, 20.0]); type=:predictions)
println(DataFrame(prof_pred))
println()

println("=== COMPARISON: Population vs Profile Framework ===")

# Example 6: Compare the two approaches conceptually
println("6. Framework comparison:")
println("   Population approach: Averages across your actual sample distribution")
println("   Profile approach: Evaluates at specific representative cases")
println("   
   For linear models: Results are very similar
   For nonlinear models: Can differ substantially!
   
   Choose based on your research question:
   • Population → 'What is the average effect in my sample?'
   • Profile → 'What is the effect for a typical case?'
   ")
println()

# Example 7: GLM example with population approach
println("7. GLM example - Urban residence probability (population effects):")
urban_model = glm(@formula(urban ~ education + experience + wage), df, Binomial(), LogitLink())
urban_effects = population_margins(urban_model, df; type=:effects, vars=[:education, :wage], scale=:response)
println(DataFrame(urban_effects))

println("\n=== API SUMMARY ===")
println("Margins.jl provides two primary functions:")
println("• population_margins(model, data; type, vars, ...) - population-averaged analysis")
println("• profile_margins(model, data; at, type, vars, ...) - profile-based analysis")
println()
println("Parameters:")
println("• type = :effects (derivatives) or :predictions (levels)")
println("• vars = variables for effects (e.g., [:x1, :x2] or :continuous)")
println("• scale = :response or :link (for both effects and predictions)")
println("• Profile grids: means_grid(), cartesian_grid(), balanced_grid() functions")
println("• Population scenarios: scenarios=Dict(...) for counterfactuals")
println()
println("Clean, conceptually-grounded API for modern marginal analysis!")
