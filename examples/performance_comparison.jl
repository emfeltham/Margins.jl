# # Performance Comparison and Optimization
#
# **Benchmarking and optimization strategies for Margins.jl**
#
# This example demonstrates the performance characteristics of Margins.jl,
# comparing Population vs Profile approaches across different dataset sizes
# and showcasing optimization strategies for production workflows.

using Margins, DataFrames, GLM, Random
using BenchmarkTools, Statistics
using Printf

Random.seed!(06515)

println("=== Margins.jl Performance Analysis ===")
println("Demonstrating O(1) profile vs O(n) population scaling")

# ## 1. Performance Testing Framework

function generate_test_data(n)
    """Generate consistent test datasets of varying sizes"""
    DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n), 
        x3 = randn(n),
        group = rand(["A", "B", "C"], n),
        region = rand(["North", "South", "East", "West"], n),
        treatment = rand([0, 1], n)
    )
end

function fit_test_model(data)
    """Fit consistent model for performance testing"""
    lm(@formula(y ~ x1 + x2 + x3 + group + region + treatment), data)
end

# Generate datasets of different sizes
sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
datasets = Dict(n => generate_test_data(n) for n in sizes)
models = Dict(n => fit_test_model(data) for (n, data) in datasets)

println("Test datasets created: $(join(sizes, ", ")) observations")

# ## 2. Profile Margins: O(1) Constant Time Performance

println("\n=== Profile Margins: O(1) Performance ===")
println("Profile margins should show constant time regardless of dataset size")

profile_times = Dict{Int, Float64}()
profile_allocations = Dict{Int, Int}()

for n in sizes
    data = datasets[n]
    model = models[n]
    
    # Benchmark profile margins at sample means
    result = @benchmark profile_margins($model, $data; at=:means, type=:effects) samples=10 seconds=30
    profile_times[n] = median(result.times) / 1000  # Convert to microseconds
    profile_allocations[n] = median(result.allocs)
    
    @printf("n=%6d: %8.0f μs (%d allocations)\n", n, profile_times[n], profile_allocations[n])
end

# Verify O(1) scaling
println("\nProfile margins scaling analysis:")
base_size = sizes[1]
base_time = profile_times[base_size]
for n in sizes[2:end]
    scaling_factor = n / base_size
    time_ratio = profile_times[n] / base_time
    @printf("n=%6d: %dx data size, %.2fx time (should be ~1.0 for O(1))\n", 
            n, round(Int, scaling_factor), time_ratio)
end

# ## 3. Population Margins: Optimized O(n) Scaling

println("\n=== Population Margins: O(n) Performance ===")
println("Population margins should show linear scaling with optimized per-row cost")

population_times = Dict{Int, Float64}()
population_allocations = Dict{Int, Int}()
per_row_times = Dict{Int, Float64}()

for n in sizes
    data = datasets[n]
    model = models[n]
    
    # Benchmark population margins
    result = @benchmark population_margins($model, $data; type=:effects) samples=5 seconds=30
    population_times[n] = median(result.times) / 1000  # Convert to microseconds  
    population_allocations[n] = median(result.allocs)
    per_row_times[n] = population_times[n] / n  # Microseconds per row
    
    @printf("n=%6d: %8.0f μs (%d allocations) = %.0f ns/row\n", 
            n, population_times[n], population_allocations[n], per_row_times[n] * 1000)
end

# Verify O(n) scaling with consistent per-row performance
println("\nPopulation margins scaling analysis:")
for n in sizes
    @printf("n=%6d: %.0f ns per row\n", n, per_row_times[n] * 1000)
end

target_per_row = 150  # Target: ~150ns per row
println("\nPer-row performance vs target (150ns/row):")
for n in sizes
    actual = per_row_times[n] * 1000
    ratio = actual / target_per_row
    status = ratio < 1.5 ? "✓ GOOD" : "⚠ REVIEW"
    @printf("n=%6d: %.0f ns/row (%.1fx target) %s\n", n, actual, ratio, status)
end

# ## 4. Complex Scenarios: Profile Performance Independence

println("\n=== Complex Profile Scenarios ===")
println("Profile performance should remain O(1) even with many scenarios")

# Test complex scenario specification
complex_scenarios = Dict(
    :x1 => [-2, -1, 0, 1, 2],         # 5 values
    :x2 => [-1, 0, 1],                # 3 values  
    :group => ["A", "B", "C"],        # 3 values
    :treatment => [0, 1]              # 2 values
)
total_scenarios = 5 * 3 * 3 * 2  # = 90 scenarios

println("Testing $(total_scenarios) scenarios across different dataset sizes:")

complex_times = Dict{Int, Float64}()
for n in [1_000, 50_000, 100_000]  # Test on different sizes
    data = datasets[n]
    model = models[n]
    
    result = @benchmark profile_margins($model, $data; at=$complex_scenarios, type=:effects) samples=5
    complex_times[n] = median(result.times) / 1000
    
    @printf("n=%6d: %8.0f μs for %d scenarios\n", n, complex_times[n], total_scenarios)
end

# Verify scenario performance is independent of dataset size
println("\nScenario complexity scaling:")
base_complex_time = complex_times[1_000]
for n in [50_000, 100_000]
    ratio = complex_times[n] / base_complex_time
    @printf("n=%6d: %.2fx time vs n=1000 (should be ~1.0)\n", n, ratio)
end

# ## 5. Memory Allocation Analysis

println("\n=== Memory Allocation Analysis ===") 
println("Showing constant allocation patterns")

println("Profile margins allocations:")
for n in sizes
    allocs = profile_allocations[n]
    bytes = allocs * 64  # Rough estimate: 64 bytes per allocation
    @printf("n=%6d: %4d allocations (~%.1f KB)\n", n, allocs, bytes/1024)
end

println("\nPopulation margins allocations:")
base_pop_allocs = population_allocations[sizes[1]]
for n in sizes
    allocs = population_allocations[n]
    ratio = allocs / base_pop_allocs
    bytes = allocs * 64
    @printf("n=%6d: %4d allocations (~%.1f KB) %.1fx base\n", n, allocs, bytes/1024, ratio)
end

# ## 6. Backend Performance Comparison

println("\n=== Backend Performance Comparison ===")
println("Finite differences (:fd) vs Automatic differentiation (:ad)")

test_size = 10_000
test_data = datasets[test_size]
test_model = models[test_size]

# Test different backends for population margins
backends = [:fd, :ad]
backend_results = Dict()

for backend in backends
    println("\nTesting backend: $(backend)")
    
    # Population margins
    pop_result = @benchmark population_margins($test_model, $test_data; backend=$backend, type=:effects) samples=3
    pop_time = median(pop_result.times) / 1000
    pop_allocs = median(pop_result.allocs)
    
    # Profile margins  
    prof_result = @benchmark profile_margins($test_model, $test_data; at=:means, backend=$backend, type=:effects) samples=10
    prof_time = median(prof_result.times) / 1000
    prof_allocs = median(prof_result.allocs)
    
    backend_results[backend] = (
        pop_time=pop_time, pop_allocs=pop_allocs,
        prof_time=prof_time, prof_allocs=prof_allocs
    )
    
    @printf("  Population: %8.0f μs (%d allocs)\n", pop_time, pop_allocs)
    @printf("  Profile:    %8.0f μs (%d allocs)\n", prof_time, prof_allocs)
end

# Compare backends
fd_results = backend_results[:fd]
ad_results = backend_results[:ad]

println("\nBackend comparison (FD vs AD):")
@printf("Population - Time ratio: %.2fx, Allocation ratio: %.2fx\n",
        ad_results.pop_time / fd_results.pop_time,
        ad_results.pop_allocs / fd_results.pop_allocs)
@printf("Profile    - Time ratio: %.2fx, Allocation ratio: %.2fx\n", 
        ad_results.prof_time / fd_results.prof_time,
        ad_results.prof_allocs / fd_results.prof_allocs)

# ## 7. Production Optimization Strategies

println("\n=== Production Optimization Strategies ===")

# Strategy 1: Use profile margins for exploration, population for final analysis
println("\n1. Exploration vs Final Analysis Strategy:")

exploration_data = datasets[100_000]  # Large dataset
exploration_model = models[100_000]

# Fast exploration with profiles
exploration_time = @elapsed begin
    scenarios = Dict(:x1 => [-1, 0, 1], :treatment => [0, 1])
    exploration_result = profile_margins(exploration_model, exploration_data; 
                                       at=scenarios, type=:effects)
end

# Final analysis with population (subset of variables)  
final_time = @elapsed begin
    final_result = population_margins(exploration_model, exploration_data;
                                    type=:effects, vars=[:treatment, :x1])
end

@printf("Exploration (6 scenarios): %.0f ms\n", exploration_time * 1000)
@printf("Final analysis (population): %.0f ms\n", final_time * 1000)

# Strategy 2: Optimal backend selection
println("\n2. Backend Selection Strategy:")
println("   - :fd for production (zero allocation after warmup)")
println("   - :ad for development/verification (higher accuracy)")

# Strategy 3: Memory-efficient large dataset handling
println("\n3. Large Dataset Strategy:")
large_data = datasets[100_000]
large_model = models[100_000]

# Demonstrate memory-efficient population analysis
memory_time = @elapsed begin
    # Focus on key variables only
    key_effects = population_margins(large_model, large_data;
                                   type=:effects, 
                                   vars=[:treatment, :x1],
                                   backend=:fd)
end

@printf("Memory-efficient analysis (100k rows, key variables): %.0f ms\n", memory_time * 1000)

# ## 8. Performance Summary and Recommendations

println("\n=== Performance Summary ===")

# Calculate overall statistics
median_profile_time = median(collect(values(profile_times)))
median_per_row_time = median(collect(values(per_row_times))) * 1000

println("\nKey Performance Metrics:")
@printf("Profile margins:    ~%.0f μs (constant time, O(1))\n", median_profile_time)
@printf("Population margins: ~%.0f ns per row (linear scaling, O(n))\n", median_per_row_time) 

println("\nDataset Size Recommendations:")
println("• < 10k observations:   Use either approach freely")
println("• 10k-100k observations: Profile preferred for exploration")  
println("• > 100k observations:   Profile for scenarios, population selectively")

println("\nBackend Recommendations:")
println("• Production workflows:  :fd (zero allocation)")
println("• Development/testing:   :ad (higher accuracy)")
println("• Large datasets:        :fd (memory efficiency)")

println("\nOptimization Strategies:")
println("• Use profile margins for scenario analysis (O(1) scaling)")
println("• Use population margins for final parameter estimates") 
println("• Specify vars parameter to focus on key variables")
println("• Consider subgroup analysis with 'over' parameter")
println("• Leverage constant-time profile performance for complex scenarios")

println("\n=== Performance Analysis Complete ===")
println("Margins.jl delivers production-grade performance with statistical rigor!")