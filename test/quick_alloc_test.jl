# Quick test to verify the fix using exact same data as allocation test
using Test, BenchmarkTools, Margins, DataFrames, GLM, StatsModels
include("test_utilities.jl")  # This provides make_test_data

println("=== Quick Allocation Test (Exact Same as Original) ===\n")

# Use exact same sizes as original test
dataset_sizes = [100, 1000, 10000]

results = []
for n_rows in dataset_sizes
    println("Testing n_rows = $n_rows")
    
    # Use exact same data generation as original test
    data = make_test_data(n=n_rows)
    model = fit(LinearModel, @formula(continuous_response ~ x + y), data)
    
    # Warmup
    result_warmup = population_margins(model, data; type=:effects, vars=[:x, :y])
    
    # Benchmark with exact same parameters
    bench = @benchmark population_margins($model, $data; type=:effects, vars=[:x, :y]) samples=10 evals=1
    
    min_allocs = minimum(bench).allocs
    allocs_per_row = min_allocs / n_rows
    time_ms = minimum(bench).time / 1e6
    
    push!(results, (n_rows, min_allocs, allocs_per_row, time_ms))
    println("  Allocations: $min_allocs bytes ($allocs_per_row per row)")
    println("  Time: $(round(time_ms, digits=2)) ms")
end

# Calculate scaling like original test
if length(results) >= 2
    println("\n=== SCALING ANALYSIS ===")
    first_result = results[1]
    last_result = results[end]
    
    first_n, first_allocs = first_result[1], first_result[2]
    last_n, last_allocs = last_result[1], last_result[2]
    
    size_ratio = last_n / first_n
    alloc_ratio = last_allocs / first_allocs
    
    println("Dataset size increase: $(size_ratio)x")
    println("Allocation increase: $(alloc_ratio)x")
    
    # This is the exact test condition from the original
    test_condition = alloc_ratio < 100
    println("Test condition (alloc_ratio < 100): $test_condition")
    
    if test_condition
        println("✅ FIX SUCCESSFUL: Allocation scaling is under control!")
    else
        println("❌ FIX FAILED: Allocation scaling still problematic")
        
        # Debug info
        println("\nDEBUG INFO:")
        for (i, (n, allocs, per_row, time)) in enumerate(results)
            println("  Result $i: n=$n, allocs=$allocs, per_row=$per_row")
        end
    end
end