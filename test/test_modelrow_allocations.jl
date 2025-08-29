#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Margins
using GLM
using DataFrames
using CategoricalArrays
using BenchmarkTools
import FormulaCompiler

println("Testing Zero-Allocation ModelRow Evaluation")
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

# Build the compiled evaluator and scenario data (this should allocate)
println("\n1. Building evaluator and scenario...")
at_dict = Dict(:education => mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2))

# Use the internal API to get the compiled evaluator
data_nt = Tables.columntable(df)
compiled = FormulaCompiler.compile_formula(model, data_nt)
profiles = Margins._build_profiles(at_dict, data_nt)
processed_profiles = [Margins._process_profile_for_scenario(prof, data_nt) for prof in profiles]

# Create scenario (this should allocate, but only once)  
scenario = FormulaCompiler.create_scenario("mixture_test", data_nt, processed_profiles[1])

println("✓ Evaluator and scenario built")
println("   Scenario education type: $(typeof(scenario.data.education))")

# Test individual modelrow evaluation (this should be zero-allocation)
println("\n2. Testing individual modelrow evaluation...")

output = zeros(Float64, 4)  # Pre-allocate output buffer
row_idx = 1

# Function to test single evaluation
function test_single_evaluation()
    compiled(output, scenario.data, row_idx)
    return output[1]  # Return first component
end

# Warm up
test_single_evaluation()

# Benchmark single modelrow evaluation
alloc_result = @allocated test_single_evaluation()
time_result = @benchmark test_single_evaluation()

println("   Single modelrow allocations: $(alloc_result) bytes")
println("   Single modelrow median time: $(round(median(time_result.times) / 1000, digits=2)) μs")

if alloc_result == 0
    println("✓ Zero-allocation modelrow evaluation achieved!")
else
    println("⚠ Modelrow evaluation has allocations: $(alloc_result) bytes")
end

# Test multiple modelrow evaluations to ensure consistency
println("\n3. Testing multiple modelrow evaluations...")

function test_multiple_evaluations(n_evals=100)
    total = 0.0
    for i in 1:n_evals
        compiled(output, scenario.data, 1)  # Always use row 1 for consistency
        total += output[1]
    end
    return total / n_evals
end

# Warm up
test_multiple_evaluations(10)

# Benchmark multiple evaluations
alloc_multi = @allocated test_multiple_evaluations(100)
time_multi = @benchmark test_multiple_evaluations(100)

println("   100 evaluations allocations: $(alloc_multi) bytes")
println("   100 evaluations median time: $(round(median(time_multi.times) / 1000, digits=2)) μs")
println("   Per-evaluation: $(round(median(time_multi.times) / 100 / 1000, digits=3)) μs")

println("\n" ^ 2 * "Summary")
println("=" ^ 20)
println("✓ Categorical mixture scenario creation working")
if alloc_result == 0
    println("✓ Zero-allocation modelrow evaluation")
else
    println("⚠ Modelrow has $(alloc_result) bytes allocations")
end

if alloc_multi == 0
    println("✓ Zero-allocation multiple evaluations")
else
    println("⚠ Multiple evaluations have $(alloc_multi) bytes total allocations")
end

println("\nModelRow Test completed!")