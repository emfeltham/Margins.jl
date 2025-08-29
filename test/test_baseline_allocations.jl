#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Margins
using GLM
using DataFrames
using CategoricalArrays
using BenchmarkTools
import FormulaCompiler

println("Testing Baseline ModelRow Allocations (No Mixtures)")
println("=" ^ 55)

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

# Test 1: Regular string categorical override (no mixture)
println("\n1. Testing regular categorical override...")
at_dict = Dict(:education => ["college"])  # Single string value in vector

data_nt = Tables.columntable(df)
compiled = FormulaCompiler.compile_formula(model, data_nt)
profiles = Margins._build_profiles(at_dict, data_nt)
processed_profiles = [Margins._process_profile_for_scenario(prof, data_nt) for prof in profiles]
scenario_regular = FormulaCompiler.create_scenario("regular_test", data_nt, processed_profiles[1])

println("✓ Regular scenario built")
println("   Scenario education type: $(typeof(scenario_regular.data.education))")

output1 = zeros(Float64, 4)
function test_regular_evaluation()
    compiled(output1, scenario_regular.data, 1)
    return output1[1]
end

# Warm up
test_regular_evaluation()

# Benchmark
alloc_regular = @allocated test_regular_evaluation()
time_regular = @benchmark test_regular_evaluation()

println("   Regular modelrow allocations: $(alloc_regular) bytes")
println("   Regular modelrow median time: $(round(median(time_regular.times) / 1000, digits=2)) μs")

# Test 2: Mixture categorical override 
println("\n2. Testing mixture categorical override...")
at_dict_mix = Dict(:education => mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2))

profiles_mix = Margins._build_profiles(at_dict_mix, data_nt)
processed_profiles_mix = [Margins._process_profile_for_scenario(prof, data_nt) for prof in profiles_mix]
scenario_mix = FormulaCompiler.create_scenario("mixture_test", data_nt, processed_profiles_mix[1])

println("✓ Mixture scenario built")
println("   Scenario education type: $(typeof(scenario_mix.data.education))")

output2 = zeros(Float64, 4)
function test_mixture_evaluation()
    compiled(output2, scenario_mix.data, 1)
    return output2[1]
end

# Warm up
test_mixture_evaluation()

# Benchmark
alloc_mixture = @allocated test_mixture_evaluation()
time_mixture = @benchmark test_mixture_evaluation()

println("   Mixture modelrow allocations: $(alloc_mixture) bytes")
println("   Mixture modelrow median time: $(round(median(time_mixture.times) / 1000, digits=2)) μs")

println("\n" ^ 2 * "Allocation Comparison")
println("=" ^ 25)
println("Regular categorical:  $(alloc_regular) bytes")
println("Mixture categorical:  $(alloc_mixture) bytes")
println("Difference:          $(alloc_mixture - alloc_regular) bytes")

if alloc_mixture == alloc_regular
    println("✓ Mixture implementation has same allocations as regular categorical")
elseif alloc_mixture < alloc_regular
    println("✓ Mixture implementation has fewer allocations!")
else
    println("⚠ Mixture implementation has $(alloc_mixture - alloc_regular) additional bytes")
end

println("\nBaseline Test completed!")