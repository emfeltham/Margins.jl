# Test potential edge cases that might need enhanced validation

using Margins
using GLM, DataFrames, CategoricalArrays, Random

println("=== Testing Potential Edge Cases ===\n")

# Generate test data
Random.seed!(123)
n = 100
data = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    cat1 = categorical(rand(["A", "B", "C"], n)),
    cat2 = categorical(rand(["X", "Y"], n)),
    income = rand(20000:100000, n)
)

model = lm(@formula(y ~ x1 + x2 + cat1 + cat2 + income), data)

println("1. Testing very small datasets")
small_data = first(data, 5)
small_model = lm(@formula(y ~ x1 + x2), small_data)
try
    result = population_margins(small_model, small_data; vars=[:x1])
    println("   ✅ Small dataset (n=5) works")
catch e
    println("   ⚠️  Small dataset issue: ", e)
end

println("\n2. Testing single observation")
try
    tiny_data = first(data, 1)
    tiny_model = lm(@formula(y ~ x1), tiny_data)
    result = population_margins(tiny_model, tiny_data; vars=[:x1])
    println("   ✅ Single observation works")
catch e
    println("   ⚠️  Single observation issue: ", e)
end

println("\n3. Testing empty vars specification")
try
    result = population_margins(model, data; vars=Symbol[])
    println("   ✅ Empty vars works")
catch e
    println("   ⚠️  Empty vars issue: ", e)
end

println("\n4. Testing non-existent variable")
try
    result = population_margins(model, data; vars=[:nonexistent])
    println("   ⚠️  Should have errored for non-existent variable")
catch e
    println("   ✅ Correctly catches non-existent variable: ", typeof(e))
end

println("\n5. Testing incompatible model types")
try
    # Test with data that doesn't match model
    bad_data = DataFrame(y = randn(50), z = randn(50))  # Different column names
    result = population_margins(model, bad_data; vars=[:x1])
    println("   ⚠️  Should have errored for incompatible data")
catch e
    println("   ✅ Correctly catches incompatible data: ", typeof(e))
end

println("\n6. Testing extreme values")
try
    extreme_data = DataFrame(
        y = [1e10, -1e10, 1e-10, -1e-10, 0],
        x = [1e5, -1e5, 1e-5, -1e-5, 0]
    )
    extreme_model = lm(@formula(y ~ x), extreme_data)
    result = population_margins(extreme_model, extreme_data; vars=[:x])
    println("   ✅ Extreme values handled")
catch e
    println("   ⚠️  Extreme values issue: ", e)
end

println("\n7. Testing missing data handling")
try
    missing_data = DataFrame(
        y = [1, 2, missing, 4, 5],
        x = [1, missing, 3, 4, 5]
    )
    # This should fail during model fitting, not in margins
    missing_model = lm(@formula(y ~ x), missing_data)
    result = population_margins(missing_model, missing_data; vars=[:x])
    println("   ⚠️  Should have handled missing data differently")
catch e
    println("   ✅ Missing data appropriately handled: ", typeof(e))
end

println("\n8. Testing duplicate column names (if possible)")
# DataFrames prevents duplicate column names, so this test is not applicable
println("   ✅ DataFrame prevents duplicate names by design")

println("\n9. Testing very large variable counts")
try
    # Create data with many variables
    large_data = data
    for i in 1:50
        large_data[!, Symbol("var$i")] = randn(nrow(data))
    end
    
    # Model with many variables
    formula_str = "y ~ " * join([String(name) for name in names(large_data) if name != :y], " + ")
    large_formula = eval(Meta.parse("@formula($formula_str)"))
    large_model = lm(large_formula, large_data)
    
    # Request effects for many variables at once
    many_vars = [Symbol("var$i") for i in 1:20]
    result = population_margins(large_model, large_data; vars=many_vars)
    println("   ✅ Many variables (20) handled successfully")
catch e
    println("   ⚠️  Many variables issue: ", e)
end

println("\n10. Testing current N reporting")
try
    result = population_margins(model, data; vars=[:x1])
    df = DataFrame(result)
    if "n" in names(df) || "N" in names(df)
        println("   ✅ N already reported in results")
    else
        println("   ❌ N not reported - this needs to be implemented")
        println("   Current columns: ", names(df))
    end
catch e
    println("   Error testing N reporting: ", e)
end

println("\n=== Edge Case Summary ===")
println("This identifies which edge cases need enhanced validation.")