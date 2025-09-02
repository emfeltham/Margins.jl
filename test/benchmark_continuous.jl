#!/usr/bin/env julia 
# benchmark_continuous.jl - Simple continuous benchmarking setup

using Pkg; Pkg.activate(".")
using Margins
using GLM, DataFrames
using BenchmarkTools
using Random

function run_continuous_benchmark()
    println("Continuous Benchmarking for Margins.jl")
    println("="^40)
    
    # Standard benchmark dataset
    Random.seed!(123)
    n = 5000
    data = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        y = randn(n)
    )
    data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
    model = lm(@formula(y ~ x1 + x2), data)
    
    println("Standard benchmark: n=$n, Linear model")
    
    # Population margins benchmark
    println("\n1. Population Margins (AME):")
    bench = @benchmark population_margins($model, $data; type=:effects, vars=[:x1, :x2], backend=:fd) samples=10 evals=3
    time_per_row = minimum(bench.times) / n
    memory_kb = minimum(bench.memory) / 1024
    
    println("   Time per row: $(round(time_per_row, digits=1))ns")
    println("   Memory: $(round(memory_kb, digits=1))KB") 
    println("   Status: $(time_per_row < 150 ? "✅ PASS" : "❌ FAIL") (target: <150ns/row)")
    
    # Profile margins benchmark
    println("\n2. Profile Margins (MEM):")
    bench = @benchmark profile_margins($model, $data; at=Dict(:x1 => 0.0, :x2 => 1.0), type=:effects, vars=[:x1, :x2], backend=:fd) samples=10 evals=3
    time_us = minimum(bench.times) / 1000
    memory_kb = minimum(bench.memory) / 1024
    
    println("   Time: $(round(time_us, digits=1))μs")
    println("   Memory: $(round(memory_kb, digits=1))KB")
    println("   Status: $(time_us < 20 ? "✅ PASS" : "❌ FAIL") (target: <20μs)")
    
    # Population predictions benchmark
    println("\n3. Population Predictions (AAP):")
    bench = @benchmark population_margins($model, $data; type=:predictions, backend=:fd) samples=10 evals=3
    time_per_row = minimum(bench.times) / n
    memory_kb = minimum(bench.memory) / 1024
    
    println("   Time per row: $(round(time_per_row, digits=1))ns")
    println("   Memory: $(round(memory_kb, digits=1))KB")
    println("   Status: $(time_per_row < 20 ? "✅ PASS" : "❌ FAIL") (target: <20ns/row)")
    
    println("\n" * "="^40)
    println("Continuous benchmark completed!")
    println("Run this script regularly to monitor performance")
    
    return Dict(
        :population_effects_ns_per_row => time_per_row,
        :population_predictions_ns_per_row => minimum(bench.times) / n,
        :profile_effects_us => time_us,
        :timestamp => now()
    )
end

# Run benchmark
results = run_continuous_benchmark()

# Save results (in a real setup, this would go to a database or log file)
println("\nResults saved for performance tracking:")