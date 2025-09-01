# test_allocation_scaling.jl - Investigate allocation scaling with data size
#
# CRITICAL QUESTION: Do allocations scale with data size (O(n)) or are they fixed costs (O(1))?
# For production econometric analysis, this determines whether the package is suitable
# for large datasets (millions of observations).

using BenchmarkTools
using GLM
using DataFrames
using Margins

function test_scaling_behavior()
    println("=" ^ 80)
    println("ALLOCATION SCALING ANALYSIS") 
    println("=" ^ 80)
    
    # Test different data sizes
    sizes = [100, 500, 1000, 2500, 5000, 10000, 25000]
    
    println("\nTesting population_margins() scaling (FD backend)...")
    println("Size\t| Allocations\t| Time (ms)\t| Alloc/Row\t| Pattern")
    println(repeat("-", 70))
    
    population_results = []
    
    for n in sizes
        # Create test data
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n), 
            y = randn(n)
        )
        model = lm(@formula(y ~ x1 + x2), data)
        
        # Warmup
        population_margins(model, data; backend=:fd, vars=[:x1])
        
        # Measure allocations and time
        bench = @benchmark population_margins($model, $data; backend=:fd, vars=[:x1]) samples=20 evals=1
        
        allocs = minimum(bench).allocs
        time_ms = minimum(bench).time / 1e6  # Convert to milliseconds
        allocs_per_row = allocs / n
        
        # Determine scaling pattern
        if length(population_results) > 0
            prev_allocs = population_results[end][2]
            prev_n = population_results[end][1]
            scaling_ratio = allocs / prev_allocs
            size_ratio = n / prev_n
            
            if abs(scaling_ratio - size_ratio) < 0.3  # Linear scaling
                pattern = "O(n)"
            elseif scaling_ratio < 1.5  # Nearly constant
                pattern = "O(1)"
            else
                pattern = "O(?)"
            end
        else
            pattern = "baseline"
        end
        
        push!(population_results, (n, allocs, time_ms, allocs_per_row, pattern))
        
        println("$(lpad(n, 4))\t| $(lpad(allocs, 10))\t| $(lpad(round(time_ms, digits=2), 8))\t| $(lpad(round(allocs_per_row, digits=2), 8))\t| $(pattern)")
    end
    
    println("\n" * repeat("=", 80))
    println("\nTesting profile_margins() scaling (FD backend, at=:means)...")
    println("Size\t| Allocations\t| Time (ms)\t| Alloc/Row\t| Pattern")
    println(repeat("-", 70))
    
    profile_results = []
    
    for n in sizes[1:5]  # Limit profile testing due to overhead
        # Create test data
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n),
            y = randn(n)
        )
        model = lm(@formula(y ~ x1 + x2), data)
        
        # Warmup
        profile_margins(model, data; at=:means, backend=:fd, vars=[:x1])
        
        # Measure allocations and time
        bench = @benchmark profile_margins($model, $data; at=:means, backend=:fd, vars=[:x1]) samples=10 evals=1
        
        allocs = minimum(bench).allocs
        time_ms = minimum(bench).time / 1e6
        allocs_per_row = allocs / n
        
        # Determine scaling pattern
        if length(profile_results) > 0
            prev_allocs = profile_results[end][2]
            prev_n = profile_results[end][1]
            scaling_ratio = allocs / prev_allocs
            size_ratio = n / prev_n
            
            if abs(scaling_ratio - size_ratio) < 0.3
                pattern = "O(n)"
            elseif scaling_ratio < 1.5
                pattern = "O(1)"
            else
                pattern = "O(?)"
            end
        else
            pattern = "baseline"
        end
        
        push!(profile_results, (n, allocs, time_ms, allocs_per_row, pattern))
        
        println("$(lpad(n, 4))\t| $(lpad(allocs, 10))\t| $(lpad(round(time_ms, digits=2), 8))\t| $(lpad(round(allocs_per_row, digits=2), 8))\t| $(pattern)")
    end
    
    println("\n" * repeat("=", 80))
    println("ANALYSIS SUMMARY")
    println(repeat("=", 80))
    
    # Analyze population scaling
    println("\n📊 POPULATION MARGINS SCALING:")
    first_n, first_allocs = population_results[1][1], population_results[1][2]
    last_n, last_allocs = population_results[end][1], population_results[end][2]
    
    actual_scaling = last_allocs / first_allocs
    size_scaling = last_n / first_n
    
    println("   Data size increased: $(first_n) → $(last_n) ($(round(size_scaling, digits=1))x)")
    println("   Allocations changed: $(first_allocs) → $(last_allocs) ($(round(actual_scaling, digits=1))x)")
    
    if actual_scaling < 1.5
        println("   🟢 SCALING: Nearly O(1) - EXCELLENT for large datasets")
        println("      → Allocations are mostly fixed costs (infrastructure overhead)")
    elseif actual_scaling < size_scaling * 0.5
        println("   🟡 SCALING: Sub-linear - GOOD for large datasets") 
        println("      → Some scaling but better than O(n)")
    elseif abs(actual_scaling - size_scaling) < size_scaling * 0.3
        println("   🔴 SCALING: Nearly O(n) - CONCERNING for large datasets")
        println("      → Allocations scale linearly with data size")
    else
        println("   ⚠️  SCALING: Complex pattern - needs investigation")
    end
    
    # Analyze allocation per row trend
    allocs_per_row_trend = [result[4] for result in population_results]
    if maximum(allocs_per_row_trend) - minimum(allocs_per_row_trend) < minimum(allocs_per_row_trend) * 0.3
        println("   📈 ALLOCATION EFFICIENCY: Stable allocations per row")
    else
        println("   📈 ALLOCATION EFFICIENCY: Variable allocations per row - investigate")
    end
    
    # Analyze profile scaling
    if length(profile_results) > 1
        println("\n📊 PROFILE MARGINS SCALING:")
        first_n, first_allocs = profile_results[1][1], profile_results[1][2] 
        last_n, last_allocs = profile_results[end][1], profile_results[end][2]
        
        actual_scaling = last_allocs / first_allocs
        size_scaling = last_n / first_n
        
        println("   Data size increased: $(first_n) → $(last_n) ($(round(size_scaling, digits=1))x)")
        println("   Allocations changed: $(first_allocs) → $(last_allocs) ($(round(actual_scaling, digits=1))x)")
        
        if actual_scaling < 1.5
            println("   🟢 SCALING: Nearly O(1) - Profile computation is data-size independent")
        else
            println("   🔴 SCALING: Depends on data size - investigate profile grid overhead")
        end
    end
    
    println("\n" * repeat("=", 80))
    println("PRODUCTION READINESS ASSESSMENT")
    println(repeat("=", 80))
    
    # Final assessment
    if actual_scaling < 1.5
        println("✅ READY FOR LARGE DATASETS")
        println("   • Allocation overhead is mostly fixed (infrastructure)")
        println("   • Suitable for datasets with millions of observations")
        println("   • Population-based methods scale efficiently")
    elseif actual_scaling < size_scaling * 0.5
        println("⚠️  USABLE FOR LARGE DATASETS WITH CAUTION")
        println("   • Some allocation scaling but sub-linear")
        println("   • Monitor memory usage for very large datasets")
        println("   • Consider data chunking for extreme sizes")
    else
        println("❌ NOT READY FOR LARGE DATASETS")
        println("   • Allocations scale too closely with data size")
        println("   • Memory usage will be problematic for large datasets") 
        println("   • Further optimization needed")
    end
    
    return population_results, profile_results
end

# Run the analysis
population_results, profile_results = test_scaling_behavior()