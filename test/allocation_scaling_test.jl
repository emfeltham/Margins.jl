#!/usr/bin/env julia
# allocation_scaling_test.jl - Systematic allocation scaling analysis

using BenchmarkTools
using GLM
using DataFrames  
using Margins
using Printf

function test_allocation_scaling()
    println("=== ALLOCATION SCALING ANALYSIS ===")
    println()
    
    # Test different dataset sizes
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    
    println("Testing population_margins allocation scaling:")
    println("Dataset Size | Allocations | Allocs/Row | Type")
    println("-------------|-------------|------------|-----")
    
    for n in sizes
        # Create test data
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n),
            x3 = rand([0, 1], n)
        )
        data.y = 0.5 * data.x1 + 0.3 * data.x2 + 0.2 * data.x3 + randn(n) * 0.1
        
        # Fit model
        model = lm(@formula(y ~ x1 + x2 + x3), data)
        
        # Warmup
        result_warmup = population_margins(model, data; backend=:fd, vars=[:x1, :x2])
        
        # Benchmark
        bench = @benchmark population_margins($model, $data; backend=:fd, vars=[:x1, :x2]) samples=10 evals=1
        
        allocs = minimum(bench).allocs
        allocs_per_row = allocs / n
        
        # Classify allocation type
        classification = if allocs_per_row < 0.01
            "Fixed Cost"
        elseif allocs_per_row < 1.0  
            "Sub-linear"
        elseif allocs_per_row < 2.0
            "Linear (Good)"
        else
            "Super-linear"
        end
        
        println(Printf.@sprintf("%12d | %11d | %10.4f | %s", n, allocs, allocs_per_row, classification))
    end
    
    println()
    println("Testing profile_margins allocation scaling:")
    println("Dataset Size | Allocations | Allocs/Row | Type") 
    println("-------------|-------------|------------|-----")
    
    for n in sizes
        # Create test data
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n),
            x3 = rand([0, 1], n)
        )
        data.y = 0.5 * data.x1 + 0.3 * data.x2 + 0.2 * data.x3 + randn(n) * 0.1
        
        # Fit model
        model = lm(@formula(y ~ x1 + x2 + x3), data)
        
        # Warmup
        result_warmup = profile_margins(model, data, means_grid(data); backend=:fd, vars=[:x1, :x2])
        
        # Benchmark
        bench = @benchmark profile_margins($model, $data, means_grid($data); backend=:fd, vars=[:x1, :x2]) samples=10 evals=1
        
        allocs = minimum(bench).allocs
        allocs_per_row = allocs / n
        
        # Classify allocation type
        classification = if allocs_per_row < 0.01
            "Fixed Cost"
        elseif allocs_per_row < 1.0
            "Sub-linear"
        elseif allocs_per_row < 2.0
            "Linear (Good)"  
        else
            "Super-linear"
        end
        
        println(Printf.@sprintf("%12d | %11d | %10.4f | %s", n, allocs, allocs_per_row, classification))
    end
    
    println()
    println("=== INTERPRETATION ===")
    println("Fixed Cost: Allocations don't scale with data size (< 0.01 per row)")
    println("Sub-linear: Allocations grow slower than data size (< 1 per row)")  
    println("Linear (Good): ~1-2 allocations per row (expected for data processing)")
    println("Super-linear: > 2 allocations per row (concerning scaling)")
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_allocation_scaling()
end