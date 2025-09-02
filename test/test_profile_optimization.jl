#!/usr/bin/env julia
# Test profile :means optimization - Before/after comparison

using Pkg; Pkg.activate(".")
using Margins
using GLM, DataFrames
using BenchmarkTools
using Random

function test_profile_means_performance()
    println("Testing profile :means optimization...")
    
    # Test multiple dataset sizes to verify O(1) scaling
    test_sizes = [1000, 5000, 10000, 25000]
    
    println("\nDataset Size | Time per Profile | Memory Usage | Scaling Assessment")
    println("-" * "-"^60)
    
    results = []
    
    for n in test_sizes
        # Create test data
        Random.seed!(123)
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n),
            y = randn(n)
        )
        data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
        model = lm(@formula(y ~ x1 + x2), data)
        
        # Warmup
        try
            profile_margins(model, data; at=:means, type=:effects, vars=[:x1, :x2], backend=:fd)
            
            # Benchmark
            bench = @benchmark profile_margins($model, $data; at=:means, type=:effects, vars=[:x1, :x2], backend=:fd) samples=10 evals=2
            
            min_time_us = minimum(bench.times) / 1000  # Convert to microseconds
            memory_kb = minimum(bench.memory) / 1024
            
            push!(results, (n=n, time_us=min_time_us, memory_kb=memory_kb))
            
            target_met = min_time_us < 1000  # Target: <1000μs (1ms)
            status = target_met ? "✅" : "❌"
            
            println("$(lpad(n, 11)) | $(lpad(round(min_time_us, digits=1), 15))μs | $(lpad(round(memory_kb, digits=1), 11))KB | $status")
            
        catch e
            println("$(lpad(n, 11)) | ERROR: $e")
            break
        end
    end
    
    # Analyze scaling behavior
    if length(results) >= 2
        println("\nScaling Analysis:")
        first_result = results[1]
        last_result = results[end]
        
        time_ratio = last_result.time_us / first_result.time_us
        memory_ratio = last_result.memory_kb / first_result.memory_kb
        size_ratio = last_result.n / first_result.n
        
        println("Dataset size increase: $(size_ratio)x")
        println("Time scaling factor: $(round(time_ratio, digits=2))x")
        println("Memory scaling factor: $(round(memory_ratio, digits=2))x")
        
        if time_ratio < 2.0
            println("✅ Time scaling: Excellent (O(1) or sub-linear)")
        elseif time_ratio < size_ratio * 0.5
            println("⚠️  Time scaling: Good (sub-linear)")
        else
            println("❌ Time scaling: Poor (linear or super-linear)")
        end
        
        if memory_ratio < 2.0
            println("✅ Memory scaling: Excellent")
        else
            println("❌ Memory scaling: Poor")
        end
    end
    
    # Compare to target
    println("\nPhase 4 Target Assessment:")
    println("Target: <1000μs per profile, <1KB memory")
    
    if !isempty(results)
        latest = results[end]
        time_met = latest.time_us < 1000
        memory_met = latest.memory_kb < 1
        
        println("Latest result (n=$(latest.n)):")
        println("  Time: $(round(latest.time_us, digits=1))μs ($(time_met ? "✅ MET" : "❌ MISSED"))")
        println("  Memory: $(round(latest.memory_kb, digits=1))KB ($(memory_met ? "✅ MET" : "❌ MISSED"))")
        
        if time_met && memory_met
            println("\n🎉 Phase 4 profile targets ACHIEVED!")
        elseif time_met
            println("\n⚠️  Time target met, memory optimization still needed")
        else
            println("\n❌ Further optimization required")
        end
    end
end

test_profile_means_performance()