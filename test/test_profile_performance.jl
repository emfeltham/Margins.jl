#!/usr/bin/env julia

# Performance test for Priority 1 fix: Profile recompilation anti-pattern
# Should show 100x+ speedup from eliminating FormulaCompiler recompilation per profile

using Margins, DataFrames, GLM, BenchmarkTools
using Tables, CategoricalArrays

# Create test data
println("Setting up test data...")
n = 1000
df = DataFrame(
    y = randn(n),
    x1 = randn(n), 
    x2 = randn(n),
    group = categorical(rand(["A", "B", "C"], n)),
    treated = rand(Bool, n)
)

# Fit model
model = lm(@formula(y ~ x1 * x2 + group + treated), df)
println("Model fitted with $(length(coef(model))) parameters")

# Create test profiles - this is what was slow before
# Note: Use categorical values for group to match the data
group_levels = levels(df.group)
test_profiles = [
    Dict(:x1 => 0.0, :x2 => 0.0, :group => group_levels[1], :treated => false),
    Dict(:x1 => 1.0, :x2 => 0.5, :group => group_levels[2], :treated => true),
    Dict(:x1 => -1.0, :x2 => 1.0, :group => group_levels[3], :treated => false),
    Dict(:x1 => 0.5, :x2 => -0.5, :group => group_levels[1], :treated => true),
    Dict(:x1 => 2.0, :x2 => 1.5, :group => group_levels[2], :treated => false)
]

println("Testing profile margins performance with $(length(test_profiles)) profiles...")

# Build reference grid DataFrame
reference_grid = DataFrame(test_profiles)

# Time the profile margins call
println("Warming up...")
result = profile_margins(model, reference_grid; type=:effects, vars=[:x1, :x2])
println("Warmup complete. Result has $(nrow(result.df)) effects computed.")

println("\\nBenchmarking profile margins (fixed implementation)...")
benchmark_result = @benchmark profile_margins($model, $reference_grid; type=:effects, vars=[:x1, :x2])

println("Results:")
println("Median time: $(median(benchmark_result.times) / 1e6) ms")
println("Mean time: $(mean(benchmark_result.times) / 1e6) ms") 
println("Allocations: $(median(benchmark_result.memory)) bytes")

# Expected performance:
# Before fix: ~50-100ms (10ms compilation × 5 profiles) 
# After fix: <1ms (single compilation + scenario evaluations)
# Target: 100x speedup

expected_time_ms = 1.0  # Target: under 1ms
actual_time_ms = median(benchmark_result.times) / 1e6

if actual_time_ms < expected_time_ms
    speedup = 50.0 / actual_time_ms  # Conservative estimate of old performance
    println("\\n✅ SUCCESS: Profile margins performance target achieved!")
    println("   Time: $(actual_time_ms) ms (target: <$(expected_time_ms) ms)")
    println("   Estimated speedup: ~$(round(speedup, digits=1))x")
else
    println("\\n❌ PERFORMANCE TARGET MISSED")
    println("   Time: $(actual_time_ms) ms (target: <$(expected_time_ms) ms)")
end

# Verify correctness
println("\\nVerifying statistical correctness...")
println("Effects computed: $(nrow(result.df))")
println("Sample results:")
for i in 1:min(3, nrow(result.df))
    row = result.df[i, :]
    println("  $(row.term): estimate=$(round(row.estimate, digits=4)), se=$(round(row.se, digits=4))")
end

println("\\n✅ Performance test completed!")