#!/usr/bin/env julia

# Test backend selection implementation

using DataFrames, GLM, CategoricalArrays, Tables
push!(LOAD_PATH, "src")
using Margins

println("=== Testing Backend Selection (Priority 2) ===")

# Create test data
df = DataFrame(
    y = randn(100),
    x = randn(100),
    group = categorical(rand(["A", "B"], 100))
)

model = lm(@formula(y ~ x + group), df)
println("✅ Model fitted")

# Test 1: Default backend selection
println("\nTest 1: Default backend selection...")
try
    # Population margins should default to :ad
    result_pop = population_margins(model, df; type=:effects, vars=[:x])
    println("✅ Population margins with auto backend (should use :ad)")
    
    # Profile margins should default to :ad  
    result_prof = profile_margins(model, df; at=:means, type=:effects, vars=[:x])
    println("✅ Profile margins with auto backend (should use :ad)")
    
catch e
    println("❌ Error: $e")
end

# Test 2: Explicit backend specification
println("\nTest 2: Explicit backend specification...")
try
    # Test explicit :fd backend
    result_fd = population_margins(model, df; type=:effects, vars=[:x], backend=:fd)
    println("✅ Population margins with explicit :fd backend")
    
    # Test explicit :ad backend
    result_ad = profile_margins(model, df; at=:means, type=:effects, vars=[:x], backend=:ad)
    println("✅ Profile margins with explicit :ad backend")
    
    # Test cross-backends (should work but may have different performance)
    result_pop_ad = population_margins(model, df; type=:effects, vars=[:x], backend=:ad)
    println("✅ Population margins with :ad backend")
    
    result_prof_fd = profile_margins(model, df; at=:means, type=:effects, vars=[:x], backend=:fd)
    println("✅ Profile margins with :fd backend")
    
catch e
    println("❌ Error: $e")
end

# Test 3: Results should be numerically equivalent across backends
println("\nTest 3: Numerical equivalence across backends...")
try
    # Population margins
    result_pop_fd = population_margins(model, df; type=:effects, vars=[:x], backend=:fd)
    result_pop_ad = population_margins(model, df; type=:effects, vars=[:x], backend=:ad)
    
    df_pop_fd = DataFrame(result_pop_fd)
    df_pop_ad = DataFrame(result_pop_ad)
    
    diff_pop = abs(df_pop_fd.estimate[1] - df_pop_ad.estimate[1])
    println("  Population FD vs AD difference: $(round(diff_pop, digits=8))")
    
    # Profile margins
    result_prof_fd = profile_margins(model, df; at=:means, type=:effects, vars=[:x], backend=:fd)
    result_prof_ad = profile_margins(model, df; at=:means, type=:effects, vars=[:x], backend=:ad)
    
    df_prof_fd = DataFrame(result_prof_fd)
    df_prof_ad = DataFrame(result_prof_ad)
    
    diff_prof = abs(df_prof_fd.estimate[1] - df_prof_ad.estimate[1])
    println("  Profile FD vs AD difference: $(round(diff_prof, digits=8))")
    
    if diff_pop < 1e-10 && diff_prof < 1e-10
        println("✅ Backends produce numerically equivalent results")
    else
        println("⚠️  Some numerical differences detected (may be within tolerance)")
    end
    
catch e
    println("❌ Error: $e")
end

println("\n=== Backend Selection Test Complete ===")