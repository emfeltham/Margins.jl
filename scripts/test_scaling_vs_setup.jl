#!/usr/bin/env julia
# Test to distinguish scaling allocations from setup allocations

using Margins, GLM, DataFrames, CategoricalArrays, BenchmarkTools

println("=== Testing Scaling vs Setup Allocations ===\n")

# Test 1: Simple continuous model (should have 0 scaling)
println("1. Simple continuous model (y ~ x1 + x2)")
for n in [100, 500, 1000, 5000]
    data = DataFrame(x1=randn(n), x2=randn(n), y=randn(n))
    data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
    model = lm(@formula(y ~ x1 + x2), data)

    # Warmup
    population_margins(model, data; backend=:fd, vars=[:x1, :x2])

    # Measure
    bench = @benchmark population_margins($model, $data; backend=:fd, vars=[:x1, :x2]) samples=20 evals=1
    allocs = minimum(bench).allocs

    println("  n=$n: $allocs allocs")
end

# Calculate scaling rate from simple model
println("\n2. Complex model with categorical (y ~ x1 + x2 + cat_var)")
for n in [100, 500, 1000, 5000]
    data = DataFrame(
        x1=randn(n),
        x2=randn(n),
        cat_var=categorical(rand(["A", "B", "C"], n)),
        y=randn(n)
    )
    data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
    model = lm(@formula(y ~ x1 + x2 + cat_var), data)

    # Warmup - auto-detect all vars (includes categorical)
    population_margins(model, data; backend=:fd)

    # Measure
    bench = @benchmark population_margins($model, $data; backend=:fd) samples=20 evals=1
    allocs = minimum(bench).allocs

    println("  n=$n: $allocs allocs")
end

println("\n3. Analyzing scaling pattern")
println("If allocations scale linearly with n → per-row scaling (BAD)")
println("If allocations are roughly constant → setup only (GOOD)")
println("\nFor continuous-only: allocs should be ~161 regardless of n")
println("For with-categorical: need to check if growth is O(1) or O(n)")
