using Pkg
Pkg.activate(".")

using Margins, GLM, DataFrames, StatsModels, BenchmarkTools, CategoricalArrays
using Random

# Create test datasets of different sizes
function create_test_data(n::Int)
    Random.seed!(123)
    DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n)
    )
end

# Test profile margins performance at different dataset sizes
println("=== Profile Margins Performance Test ===")
println("Testing O(n) scaling fix...")

for n in [1000, 5000, 10000]
    println("\n--- Dataset size: $n ---")
    
    data = create_test_data(n)
    model = lm(@formula(y ~ x1 + x2), data)
    
    # Test profile margins (should be constant time regardless of n)
    print("Profile margins: ")
    @time result_profile = profile_margins(model, data; at=:means, type=:effects, vars=[:x1])
    
    # Test population margins for comparison (should scale with n)  
    print("Population margins: ")
    @time result_pop = population_margins(model, data; type=:effects, vars=[:x1])
    
    # Verify results
    df_profile = DataFrame(result_profile)
    df_pop = DataFrame(result_pop)
    
    println("Profile results: $(nrow(df_profile)) rows")
    println("Population results: $(nrow(df_pop)) rows")
    
    if nrow(df_profile) != 1
        println("❌ FAILED: Profile margins returned $(nrow(df_profile)) results instead of 1")
    else
        println("✅ Profile margins working correctly")
    end
end

println("\n=== Summary ===")
println("If the fix worked correctly:")
println("- Profile margins time should be constant regardless of dataset size")
println("- Population margins time should increase with dataset size")
println("- Profile margins should always return exactly 1 result")