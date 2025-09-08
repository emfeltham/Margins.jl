# Debug script to confirm weights vector allocation is the bottleneck
using Margins, GLM, DataFrames, BenchmarkTools, Tables, FormulaCompiler

println("=== Weights Vector Allocation Analysis ===\n")

function test_weights_allocation(n_rows)
    println("Testing with n_rows = $n_rows")
    
    # Create test data
    df = DataFrame(
        y = randn(n_rows),
        x = randn(n_rows)
    )
    data_nt = Tables.columntable(df)
    model = lm(@formula(y ~ x), df)
    
    # Build engine components
    engine = Margins.get_or_build_engine(model, data_nt, [:x], GLM.vcov)
    rows = 1:n_rows
    
    # Test 1: Allocation from ones(Float64, length(rows))
    println("  Testing ones(Float64, length(rows)) allocation:")
    alloc_ones = @allocated ones(Float64, length(rows))
    println("    ones(Float64, $n_rows): $alloc_ones bytes")
    
    # Test 2: Using pre-allocated weights vs allocating weights
    println("  Testing _accumulate_weighted_ame_gradient! with different weight strategies:")
    
    gβ_sum = Vector{Float64}(undef, length(engine.β))
    
    # Pre-allocated weights (reusable)
    preallocated_weights = ones(Float64, n_rows)
    
    # Warmup
    Margins._accumulate_weighted_ame_gradient!(gβ_sum, engine.de, engine.β, 1:min(10, n_rows), :x, preallocated_weights; 
                                               link=engine.link, backend=:fd)
    
    # Test with pre-allocated weights
    alloc_prealloc = @allocated Margins._accumulate_weighted_ame_gradient!(gβ_sum, engine.de, engine.β, rows, :x, preallocated_weights; 
                                                                          link=engine.link, backend=:fd)
    println("    With pre-allocated weights: $alloc_prealloc bytes")
    
    # Test with freshly allocated weights (like _compute_continuous_ame does)
    function test_with_fresh_weights(gβ_sum, de, β, rows, var, link, backend)
        fresh_weights = ones(Float64, length(rows))  # This is what _compute_continuous_ame does!
        Margins._accumulate_weighted_ame_gradient!(gβ_sum, de, β, rows, var, fresh_weights; 
                                                  link=link, backend=backend)
    end
    
    # Warmup
    test_with_fresh_weights(gβ_sum, engine.de, engine.β, 1:min(10, n_rows), :x, engine.link, :fd)
    
    # Test with fresh allocation (current _compute_continuous_ame pattern)
    alloc_fresh = @allocated test_with_fresh_weights(gβ_sum, engine.de, engine.β, rows, :x, engine.link, :fd)
    println("    With fresh weights (current pattern): $alloc_fresh bytes")
    println("    Allocation overhead from weights: $(alloc_fresh - alloc_prealloc) bytes")
    
    # Test 3: What if we avoid weights entirely for unweighted case?
    println("  Testing unweighted gradient accumulation (theoretical fix):")
    
    # This would be an optimized version that doesn't need weights for the unweighted case
    function test_unweighted_gradient_accumulation(gβ_sum, de, β, rows, var, link, backend)
        # Simulated unweighted version - just the core computation
        fill!(gβ_sum, 0.0)
        temp_grad = Vector{Float64}(undef, length(β))  # Small temp buffer
        
        for row in rows
            if link isa GLM.IdentityLink
                FormulaCompiler.fd_jacobian_column!(temp_grad, de, row, var)
            else
                FormulaCompiler.me_mu_grad_beta!(temp_grad, de, β, row, var; link=link)
            end
            gβ_sum .+= temp_grad
        end
        
        gβ_sum ./= length(rows)
        return gβ_sum
    end
    
    temp_grad = Vector{Float64}(undef, length(engine.β))
    
    # Warmup
    test_unweighted_gradient_accumulation(gβ_sum, engine.de, engine.β, 1:min(10, n_rows), :x, engine.link, :fd)
    
    # Test unweighted approach
    alloc_unweighted = @allocated test_unweighted_gradient_accumulation(gβ_sum, engine.de, engine.β, rows, :x, engine.link, :fd)
    println("    Unweighted approach: $alloc_unweighted bytes")
    println("    Savings vs current: $(alloc_fresh - alloc_unweighted) bytes")
    
    println("  ---")
    return (alloc_ones, alloc_prealloc, alloc_fresh, alloc_unweighted)
end

# Test different dataset sizes
sizes = [100, 500, 1000, 2000, 5000]
results = []

for size in sizes
    result = test_weights_allocation(size)
    push!(results, (size, result...))
    println()
end

println("=== WEIGHTS ALLOCATION ANALYSIS SUMMARY ===")
println("Size | ones() Alloc | Pre-allocated | Fresh Weights | Unweighted | Overhead")
println("-----|--------------|---------------|---------------|------------|----------")
for (size, ones_alloc, prealloc, fresh, unweighted) in results
    overhead = fresh - prealloc
    println("$size | $ones_alloc bytes | $prealloc bytes | $fresh bytes | $unweighted bytes | $overhead bytes")
end

# Calculate scaling factors
if length(results) >= 2
    println("\n=== SCALING FACTORS ===")
    first_size, first_results = results[1][1], results[1][2:end]
    last_size, last_results = results[end][1], results[end][2:end]
    
    size_ratio = last_size / first_size
    println("Dataset size increase: $(size_ratio)x")
    
    names = ["ones() allocation", "Pre-allocated weights", "Fresh weights", "Unweighted approach"]
    for (i, name) in enumerate(names)
        if first_results[i] > 0 && last_results[i] > 0
            alloc_ratio = last_results[i] / first_results[i]
            println("$name scaling: $(alloc_ratio)x")
        end
    end
    
    # Most importantly: does the overhead scale?
    first_overhead = first_results[3] - first_results[2]  # fresh - prealloc
    last_overhead = last_results[3] - last_results[2]
    if first_overhead > 0
        overhead_scaling = last_overhead / first_overhead
        println("Weights overhead scaling: $(overhead_scaling)x")
    end
end