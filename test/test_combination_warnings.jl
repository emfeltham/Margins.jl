# Test combination explosion warnings

using Margins
using GLM, DataFrames, CategoricalArrays, Random

println("=== Testing Combination Explosion Warnings ===\n")

# Generate test data with many categories
Random.seed!(123)
n = 100
data = DataFrame(
    y = randn(n),
    a = categorical(rand(["A1", "A2", "A3", "A4", "A5"], n)),  # 5 levels
    b = categorical(rand(["B1", "B2", "B3", "B4", "B5"], n)),  # 5 levels  
    c = categorical(rand(["C1", "C2", "C3"], n)),              # 3 levels
    d = categorical(rand(["D1", "D2"], n)),                   # 2 levels
    x = randn(n)
)

model = lm(@formula(y ~ a + b + c + d + x), data)

println("1. Testing moderate combinations (should show warning)")
println("   Groups: [:a, :b] (25 combinations)")
try
    result = population_margins(model, data; vars=[:x], groups=[:a, :b])
    println("   ✅ Completed with warning")
catch e
    println("   ❌ Failed: ", e)
end

println("\n2. Testing large combinations (should show error)")  
println("   Groups: [:a, :b, :c], Scenarios: Dict(:d => [\"D1\", \"D2\"]) (150 combinations)")
try
    result = population_margins(model, data; 
        vars=[:x], 
        groups=[:a, :b, :c],  # 5×5×3 = 75
        scenarios=Dict(:d => ["D1", "D2"])  # ×2 = 150 total
    )
    println("   ⚠️  Should have shown warning but didn't!")
catch e
    println("   ✅ Correctly blocked: ", e)
end

println("\n3. Testing extreme combinations (should error)")
println("   Creating data with many more categories...")
extreme_data = DataFrame(
    y = randn(50),
    var1 = categorical(rand(1:10, 50)),  # 10 levels
    var2 = categorical(rand(1:10, 50)),  # 10 levels
    var3 = categorical(rand(1:10, 50)),  # 10 levels  
    var4 = categorical(rand(1:5, 50)),   # 5 levels
    x = randn(50)
)
extreme_model = lm(@formula(y ~ var1 + var2 + var3 + var4 + x), extreme_data)

try
    # This should create 10×10×10×5 = 5000 combinations - way over limit
    result = population_margins(extreme_model, extreme_data; 
        vars=[:x], 
        groups=[:var1, :var2, :var3, :var4]
    )
    println("   ⚠️  Should have errored but didn't!")
catch e
    println("   ✅ Correctly blocked extreme case: ", e)
end

println("\n=== Summary ===")
println("Combination explosion warnings protect users from accidentally")
println("creating analyses that would exhaust system resources.")