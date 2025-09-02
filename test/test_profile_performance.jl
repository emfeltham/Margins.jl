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
        x2 = randn(n),
        region = categorical(rand(["North", "South"], n))
    )
end

# Test profile margins performance at different dataset sizes
for n in [1000, 5000, 10000]
    println("\n=== Testing with n = $n ===")
    
    data = create_test_data(n)
    model = lm(@formula(y ~ x1 + x2 + region), data)
    
    # Test profile margins at means
    println("Profile margins at means:")
    @time result = profile_margins(model, data; at=:means, type=:effects, vars=[:x1])
    
    # Also test population margins for comparison
    println("Population margins (for comparison):")
    @time result_pop = population_margins(model, data; type=:effects, vars=[:x1])
    
    df = DataFrame(result)
    df_pop = DataFrame(result_pop)
    println("Profile result estimates: ", df.estimate)
    println("Population result estimates: ", df_pop.estimate)
end