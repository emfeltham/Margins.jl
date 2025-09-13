#!/usr/bin/env julia

# Debug script to understand result structure

using Margins, GLM, DataFrames, CategoricalArrays

# Create test data 
n = 20  # Smaller for easier debugging
data = DataFrame(
    y = randn(n),
    x1 = randn(n),
    education = CategoricalArray(repeat(["HS", "College"], n÷2))
)

println("Data preview:")
println(data)

# Fit model
model = lm(@formula(y ~ x1 + education), data)
println("\nModel fitted")

# Test grouping
println("\n" * repeat("=", 50))
println("TESTING GROUPING RESULTS")
println(repeat("=", 50))

println("\nBase (non-grouped):")
result_base = population_margins(model, data; type=:effects, vars=[:x1])
df_base = DataFrame(result_base)
println("Columns: ", names(df_base))
println(df_base)

println("\nGrouped by education:")
result_grouped = population_margins(model, data; type=:effects, vars=[:x1], groups=:education)
df_grouped = DataFrame(result_grouped)
println("Columns: ", names(df_grouped))
println("Size: ", size(df_grouped))
println(df_grouped)

# Let's also check if the raw MarginsResult has the right information
println("\nMarginsResult internals:")
println("result_grouped.estimates: ", result_grouped.estimates)
println("result_grouped.profile_values: ", result_grouped.profile_values)
if haskey(result_grouped.metadata, :analysis_type)
    println("result_grouped.metadata[:analysis_type]: ", result_grouped.metadata[:analysis_type])
end