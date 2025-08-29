#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Margins
using GLM
using DataFrames
using CategoricalArrays
using BenchmarkTools
import FormulaCompiler

println("Testing Vanilla ModelRow Allocations (No Scenarios)")
println("=" ^ 55)

# Create test data
n = 1000
df = DataFrame(
    y = randn(n),
    x = randn(n),
    education = categorical(rand(["high_school", "college", "graduate"], n))
)

# Fit model
model = lm(@formula(y ~ x + education), df)
println("✓ Model fitted successfully")

# Test 1: Pure original data evaluation (no scenarios at all)
println("\n1. Testing pure original data evaluation...")

data_nt = Tables.columntable(df)
compiled = FormulaCompiler.compile_formula(model, data_nt)

println("✓ Compiled evaluator built")

output = zeros(Float64, 4)  # Pre-allocate output buffer
row_idx = 5  # Test with row 5

# Function to test pure evaluation on original data
function test_vanilla_evaluation()
    compiled(output, data_nt, row_idx)
    return output[1]
end

# Warm up
test_vanilla_evaluation()

# Benchmark pure evaluation
alloc_vanilla = @allocated test_vanilla_evaluation()
time_vanilla = @benchmark test_vanilla_evaluation()

println("   Vanilla modelrow allocations: $(alloc_vanilla) bytes")
println("   Vanilla modelrow median time: $(round(median(time_vanilla.times) / 1000, digits=2)) μs")

# Test 2: Empty scenario (should be equivalent to vanilla)
println("\n2. Testing empty scenario...")

empty_scenario = FormulaCompiler.create_scenario("empty", data_nt, Dict{Symbol,Any}())

println("✓ Empty scenario built")

function test_empty_scenario_evaluation()
    compiled(output, empty_scenario.data, row_idx)
    return output[1]
end

# Warm up
test_empty_scenario_evaluation()

# Benchmark empty scenario
alloc_empty = @allocated test_empty_scenario_evaluation()
time_empty = @benchmark test_empty_scenario_evaluation()

println("   Empty scenario allocations: $(alloc_empty) bytes") 
println("   Empty scenario median time: $(round(median(time_empty.times) / 1000, digits=2)) μs")

println("\n" ^ 2 * "Allocation Analysis")
println("=" ^ 25)
println("Vanilla (no scenario):  $(alloc_vanilla) bytes")
println("Empty scenario:         $(alloc_empty) bytes")

if alloc_vanilla == 0
    println("✓ Vanilla evaluation is zero-allocation as expected")
else
    println("⚠ REGRESSION: Vanilla evaluation has $(alloc_vanilla) bytes")
end

if alloc_empty == 0
    println("✓ Empty scenario is zero-allocation")  
else
    println("⚠ Empty scenario has $(alloc_empty) bytes")
end

println("\nVanilla Test completed!")