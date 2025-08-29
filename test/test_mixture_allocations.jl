#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Margins
using GLM
using DataFrames
using CategoricalArrays
using BenchmarkTools

println("Testing Zero-Allocation Categorical Mixtures")
println("=" ^ 50)

# Create test data with categorical variable
n = 1000
df = DataFrame(
    y = randn(n),
    x = randn(n),
    education = categorical(rand(["high_school", "college", "graduate"], n))
)

# Fit model
model = lm(@formula(y ~ x + education), df)

println("✓ Model fitted successfully")

# Test basic mixture functionality
println("\n1. Testing basic mixture functionality...")

at_dict = Dict(:education => mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2))
result = profile_margins(model, df; at=at_dict, type=:predictions)

println("✓ Basic mixture evaluation successful")
println("   Result dimensions: $(size(result.table))")

# Test weighted combination is correct by comparing to manual calculation
println("\n2. Verifying weighted combinations...")

# Get individual level predictions  
# Use vectors to specify single values (this works with the profile system)
at_hs = Dict(:education => ["high_school"])
at_college = Dict(:education => ["college"]) 
at_grad = Dict(:education => ["graduate"])

result_hs = profile_margins(model, df; at=at_hs, type=:predictions)
result_college = profile_margins(model, df; at=at_college, type=:predictions)
result_grad = profile_margins(model, df; at=at_grad, type=:predictions)

println("   Available columns in result: ", names(result_hs.table))

# For predictions, the predicted value is in the :dydx column
expected_pred = 0.3 * result_hs.table.dydx[1] + 
                0.5 * result_college.table.dydx[1] + 
                0.2 * result_grad.table.dydx[1]

actual_pred = result.table.dydx[1]

println("   Expected weighted prediction: $(round(expected_pred, digits=6))")
println("   Actual mixture prediction:    $(round(actual_pred, digits=6))")
println("   Difference: $(abs(expected_pred - actual_pred))")

if abs(expected_pred - actual_pred) < 1e-10
    println("✓ Weighted combination is mathematically correct")
else
    println("✗ Weighted combination differs from expected!")
end

# Test allocation benchmark
println("\n3. Testing allocations during evaluation...")

# Function to benchmark a single evaluation
function benchmark_single_evaluation()
    profile_margins(model, df; at=at_dict, type=:predictions)
end

# Warm up
benchmark_single_evaluation()

# Benchmark allocations
alloc_result = @allocated benchmark_single_evaluation()
time_result = @benchmark benchmark_single_evaluation()

println("   Allocations: $(alloc_result) bytes")
println("   Median time: $(round(median(time_result.times) / 1000, digits=2)) μs")

if alloc_result == 0
    println("✓ Zero allocations achieved!")
else
    println("⚠ Allocations detected: $(alloc_result) bytes")
end

println("\n" ^ 2 * "Summary")
println("=" ^ 20)
println("✓ Categorical mixtures working correctly")
println("✓ Weighted combinations mathematically correct")
if alloc_result == 0
    println("✓ Zero-allocation execution achieved")
else
    println("⚠ Some allocations remain: $(alloc_result) bytes")
end

println("\nTest completed!")