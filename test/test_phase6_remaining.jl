# Test remaining Phase 6 items
# This will help identify what actually needs to be finished

using Margins
using GLM, DataFrames, CategoricalArrays, Random

println("=== Testing Remaining Phase 6 Items ===\n")

# Generate test data
Random.seed!(123)
n = 1000
data = DataFrame(
    y = randn(n) .* 2 .+ 5,
    education = categorical(rand(["HS", "College", "Grad"], n)),
    income = rand(25000:100000, n),  
    treatment = rand([0, 1], n),
    x1 = randn(n),
    x2 = randn(n)
)

model = lm(@formula(y ~ education + income + treatment + x1 + x2), data)

println("1. Testing Nested Continuous Binning")
println("   Syntax: :education => (:income, 4) for education-specific income quartiles")

try
    result = population_margins(model, data; 
        vars=[:treatment], 
        groups=:education => (:income, 4)
    )
    println("   ✅ Works! Got $(nrow(DataFrame(result))) results")
    df = DataFrame(result)
    println("   Sample results:")
    println("   ", first(df, 3))
catch e
    println("   ❌ Failed: ", e)
end

println("\n2. Testing Empty Subgroups Handling")
println("   Creating data with potential empty subgroups...")

# Create data that will likely have empty subgroups when binned
sparse_data = DataFrame(
    y = randn(50),
    education = categorical(vcat(fill("HS", 45), fill("College", 5))), # Very uneven
    income = vcat(rand(20000:30000, 45), rand(80000:90000, 5)),  # Two distinct ranges
    x = randn(50)
)

sparse_model = lm(@formula(y ~ education + income + x), sparse_data)

try
    result = population_margins(sparse_model, sparse_data; 
        vars=[:x], 
        groups=[:education, (:income, 5)]  # 5 bins from sparse data
    )
    df = DataFrame(result)
    println("   Results from sparse data (", nrow(df), " rows):")
    println("   ", df)
    
    # Check for NaN values
    nan_count = sum(isnan.(df.estimate))
    if nan_count > 0
        println("   ⚠️  Found $nan_count NaN estimates - empty subgroup issue exists")
    else
        println("   ✅ No NaN values found")
    end
catch e
    println("   ❌ Failed: ", e)
end

println("\n3. Testing Combination Explosion Warnings")
println("   Testing large combination counts...")

try
    # This should create many combinations - test if there are warnings
    large_data = DataFrame(
        y = randn(100),
        a = categorical(rand(["A1", "A2", "A3", "A4"], 100)),
        b = categorical(rand(["B1", "B2", "B3", "B4"], 100)),
        c = categorical(rand(["C1", "C2", "C3", "C4"], 100)),
        x = randn(100)
    )
    
    large_model = lm(@formula(y ~ a + b + c + x), large_data)
    
    println("   Testing :a => [:b, :c] (should create ~16 combinations)")
    result = population_margins(large_model, large_data; 
        vars=[:x], 
        groups=:a => [:b, :c]
    )
    println("   ✅ Completed - got $(nrow(DataFrame(result))) results")
    
    # Test scenarios that could create explosion
    println("   Testing scenarios with multiple variables...")
    result2 = population_margins(large_model, large_data; 
        vars=[:x], 
        scenarios=Dict(:a => ["A1", "A2", "A3"], :b => ["B1", "B2"])  # 6 combinations
    )
    println("   ✅ Multi-variable scenarios work - got $(nrow(DataFrame(result2))) results")
    
catch e
    println("   ❌ Failed: ", e)
end

println("\n=== Summary ===")
println("This test helps identify which Phase 6 items remain unimplemented.")