#!/usr/bin/env julia

# Test script for Phase 2 grouping functionality

using Margins, GLM, DataFrames, CategoricalArrays

println("Testing Phase 2 Grouping Functionality")
println(repeat("=", 50))

# Create test data with categorical grouping variable
n = 100
data = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    education = CategoricalArray(rand(["HS", "College"], n)),
    gender = CategoricalArray(rand(["Male", "Female"], n))
)

println("Data summary:")
println("N = ", nrow(data))
println("Education levels: ", unique(data.education))
println("Gender levels: ", unique(data.gender))

# Fit a simple linear model
model = lm(@formula(y ~ x1 + x2 + education + gender), data)
println("\nModel fitted successfully")

println("\n" * repeat("=", 50))
println("PHASE 2 TESTS")
println(repeat("=", 50))

# Test 1: Simple categorical grouping (single variable)
println("\nTest 1: Simple categorical grouping - groups=:education")
try
    result1 = population_margins(model, data; type=:effects, vars=[:x1], groups=:education)
    println("✅ SUCCESS: Simple grouping works")
    df1 = DataFrame(result1)
    println("Result shape: ", size(df1))
    println("Columns: ", names(df1))
    println("First few rows:")
    println(first(df1, 3))
catch e
    println("❌ ERROR: Simple grouping failed")
    println("Error: ", e)
end

println("\n" * repeat("-", 30))

# Test 2: Cross-tabulation (multiple variables)
println("Test 2: Cross-tabulation - groups=[:education, :gender]")
try
    result2 = population_margins(model, data; type=:effects, vars=[:x1], groups=[:education, :gender])
    println("✅ SUCCESS: Cross-tabulation works")
    df2 = DataFrame(result2)
    println("Result shape: ", size(df2))
    println("Columns: ", names(df2))
    println("First few rows:")
    println(first(df2, 4))
catch e
    println("❌ ERROR: Cross-tabulation failed")
    println("Error: ", e)
end

println("\n" * repeat("-", 30))

# Test 3: Compare with non-grouped computation
println("Test 3: Sanity check - non-grouped computation")
try
    result_base = population_margins(model, data; type=:effects, vars=[:x1])
    println("✅ SUCCESS: Base computation works")
    df_base = DataFrame(result_base)
    println("Base result: ")
    println(df_base)
catch e
    println("❌ ERROR: Base computation failed")
    println("Error: ", e)
end

println("\n" * repeat("=", 50))
println("Phase 2 testing complete")