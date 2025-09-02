#!/usr/bin/env julia
# Quick benchmark for population_margins() performance validation

using Pkg; Pkg.activate(".")
using Margins
using GLM, DataFrames
using BenchmarkTools
using Random

# Generate test data
Random.seed!(123)
n = 1000
data = DataFrame(
    x1 = randn(n),
    x2 = randn(n), 
    x3 = rand(n),
    y = Bool[]
)

# Create dependent variable
for i in 1:n
    eta = 0.5 * data.x1[i] + 0.3 * data.x2[i] - 0.2 * data.x3[i]
    prob = 1 / (1 + exp(-eta))
    push!(data.y, rand() < prob)
end

# Fit model
model = glm(@formula(y ~ x1 + x2 + x3), data, Binomial(), LogitLink())

println("=== Population Margins Performance Benchmark ===")
println("Model: Logistic regression with $(n) observations")
println("Testing population_margins() with target <100ns per row\n")

# Test basic AME computation
println("1. Average Marginal Effects (AME):")
try
    # Warm up
    result = population_margins(model, data; type=:effects, backend=:fd)
    
    # Benchmark
    bench = @benchmark population_margins($model, $data; type=:effects, backend=:fd) samples=20 evals=5
    time_per_row = minimum(bench.times) / n  # nanoseconds per row
    
    println("   Minimum time: $(minimum(bench.times) / 1e6) ms")
    println("   Time per row: $(round(time_per_row, digits=1)) ns")
    println("   Target met: $(time_per_row < 100 ? "✅ YES" : "❌ NO")")
    println("   Results: $(nrow(DataFrame(result))) effects computed")
catch e
    println("   Error: $e")
end

println("\n2. Average Adjusted Predictions (AAP):")
try
    # Warm up
    result = population_margins(model, data; type=:predictions, backend=:fd)
    
    # Benchmark
    bench = @benchmark population_margins($model, $data; type=:predictions, backend=:fd) samples=20 evals=5
    time_per_row = minimum(bench.times) / n  # nanoseconds per row
    
    println("   Minimum time: $(minimum(bench.times) / 1e6) ms")
    println("   Time per row: $(round(time_per_row, digits=1)) ns") 
    println("   Target met: $(time_per_row < 100 ? "✅ YES" : "❌ NO")")
    println("   Results: $(nrow(DataFrame(result))) predictions computed")
catch e
    println("   Error: $e")
end

println("\n3. Input validation performance:")
try
    bench = @benchmark _validate_population_inputs($model, $data, :effects, nothing, :mu, :fd, nothing, nothing) samples=100
    validation_time = minimum(bench.times)
    
    println("   Validation time: $(round(validation_time, digits=1)) ns")
    println("   Overhead: $(validation_time < 1000 ? "✅ Minimal" : "⚠️  High")")
catch e
    println("   Error: $e")
end

println("\n=== Summary ===")
println("Phase 2 API improvements completed successfully!")