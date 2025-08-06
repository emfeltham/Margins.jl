# basic_usage.jl - Examples showing new margins() function

using Margins
using DataFrames, GLM, StatsModels
using Random

# Create example data
Random.seed!(123)
df = DataFrame(
    wage = exp.(randn(1000) .+ 5),  # Log-normal wages
    education = rand(8:16, 1000),   # Years of education
    experience = rand(0:40, 1000),  # Years of experience
    gender = categorical(rand(["Male", "Female"], 1000)),
    urban = rand([false, true], 1000)
)

println("=== Basic margins() Usage Examples ===\n")

# Fit a wage model
wage_model = lm(@formula(log(wage) ~ education + experience + I(experience^2) + gender + urban), df)

println("Model fitted: log(wage) ~ education + experience + experience² + gender + urban\n")

# Example 1: Basic marginal effects
println("1. Basic marginal effects for education:")
me_education = margins(wage_model, df, :education)
println(me_education)
println()

# Example 2: Multiple variables
println("2. Marginal effects for multiple variables:")
me_multiple = margins(wage_model, df, [:education, :experience])
println(me_multiple)
println()

# Example 3: Categorical variable with different contrasts
println("3a. Gender effects with pairwise contrasts:")
me_gender_pairwise = margins(wage_model, df, :gender; contrasts = :pairwise)
println(me_gender_pairwise)
println()

println("3b. Gender effects with baseline contrasts:")
me_gender_baseline = margins(wage_model, df, :gender; contrasts = :baseline)
println(me_gender_baseline)
println()

# Example 4: Representative values
println("4. Education effects at representative experience levels:")
me_repr = margins(wage_model, df, :education; 
                 representative_values = Dict(:experience => [5, 15, 25]))
println(me_repr)
println()

# Example 5: Predictions at specific values
println("5. Predicted wages at different education levels:")
pred = margins(wage_model, df, :education;
              type = :prediction,
              representative_values = Dict(
                  :education => [12, 16, 20],
                  :experience => [10],
                  :gender => ["Male"],
                  :urban => [true]
              ))
println(pred)
println()

# Example 6: DataFrame export
println("6. Export to DataFrame:")
df_results = DataFrame(me_multiple)
println(first(df_results, 5))
println()

# Example 7: GLM example
println("7. GLM example - Urban residence probability:")
urban_model = glm(@formula(urban ~ education + experience + wage), df, Binomial(), LogitLink())
me_urban = margins(urban_model, df, [:education, :wage])
println(me_urban)
