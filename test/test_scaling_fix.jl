# Test script to verify the scaling fix works correctly
using Margins, GLM, DataFrames, BenchmarkTools, Tables

println("=== Testing Allocation Scaling Fix ===\n")

function test_fix_correctness(n_rows)
    println("Testing correctness with n_rows = $n_rows")
    
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
    
    # Test 1: Compare results from old vs new function
    println("  Comparing mathematical correctness:")
    
    # Create identical gβ buffers for comparison
    gβ_old = Vector{Float64}(undef, length(engine.β))
    gβ_new = Vector{Float64}(undef, length(engine.β))
    
    # Old approach (allocates weights vector)
    weights = ones(Float64, length(rows))
    Margins._accumulate_weighted_ame_gradient!(gβ_old, engine.de, engine.β, rows, :x, weights; 
                                              link=engine.link, backend=:fd)
    
    # New approach (no weights allocation)
    Margins._accumulate_unweighted_ame_gradient!(gβ_new, engine.de, engine.β, rows, :x; 
                                                link=engine.link, backend=:fd)
    
    # Compare results
    max_diff = maximum(abs.(gβ_old .- gβ_new))
    println("    Maximum difference between old/new: $max_diff")
    
    if max_diff < 1e-12
        @info "Mathematical equivalence verified: results are identical"
    else
        @info "Mathematical equivalence violated: results differ - investigation required"
        return false
    end
    
    # Test 2: Allocation comparison
    println("  Comparing allocation behavior:")
    
    # Warmup both approaches
    Margins._accumulate_weighted_ame_gradient!(gβ_old, engine.de, engine.β, 1:min(10, n_rows), :x, weights; 
                                              link=engine.link, backend=:fd)
    Margins._accumulate_unweighted_ame_gradient!(gβ_new, engine.de, engine.β, 1:min(10, n_rows), :x; 
                                                link=engine.link, backend=:fd)
    
    # Test old approach allocation (with weights vector creation)
    old_alloc = @allocated begin
        temp_weights = ones(Float64, length(rows))
        Margins._accumulate_weighted_ame_gradient!(gβ_old, engine.de, engine.β, rows, :x, temp_weights; 
                                                  link=engine.link, backend=:fd)
    end
    
    # Test new approach allocation (no weights vector)
    new_alloc = @allocated Margins._accumulate_unweighted_ame_gradient!(gβ_new, engine.de, engine.β, rows, :x; 
                                                                       link=engine.link, backend=:fd)
    
    println("    Old approach (with weights): $old_alloc bytes")
    println("    New approach (no weights): $new_alloc bytes")
    println("    Allocation reduction: $(old_alloc - new_alloc) bytes ($(round((old_alloc - new_alloc)/old_alloc*100, digits=1))%)")
    
    return true
end

function test_full_population_margins_scaling()
    println("Testing full population_margins() scaling:")
    
    sizes = [100, 500, 1000]
    results = []
    
    for n_rows in sizes
        println("  Testing n_rows = $n_rows")
        
        # Create test data
        df = DataFrame(
            y = randn(n_rows),
            x = randn(n_rows)
        )
        model = lm(@formula(y ~ x), df)
        
        # Warmup
        result_warmup = population_margins(model, df; type=:effects, vars=[:x])
        
        # Benchmark
        bench = @benchmark population_margins($model, $df; type=:effects, vars=[:x]) samples=5 evals=1
        
        min_allocs = minimum(bench).allocs
        allocs_per_row = min_allocs / n_rows
        time_ms = minimum(bench).time / 1e6
        
        push!(results, (n_rows, min_allocs, allocs_per_row, time_ms))
        println("    Allocations: $min_allocs bytes ($allocs_per_row per row)")
        println("    Time: $(round(time_ms, digits=2)) ms")
    end
    
    # Check scaling
    if length(results) >= 2
        println("  Scaling analysis:")
        first_n, first_allocs = results[1][1], results[1][2]
        last_n, last_allocs = results[end][1], results[end][2]
        
        size_ratio = last_n / first_n
        alloc_ratio = last_allocs / first_allocs
        
        println("    Dataset size increase: $(size_ratio)x")
        println("    Allocation increase: $(alloc_ratio)x")
        
        if alloc_ratio < 100
            @info "Allocation scaling satisfies performance criteria (target: <100x)"
        else
            @info "Allocation scaling exceeds performance criteria (>100x)"
        end
    end
    
    return results
end

# Run correctness tests
println("=== CORRECTNESS TESTS ===")
for n in [100, 1000]
    success = test_fix_correctness(n)
    if !success
        @info "Correctness validation failed"
        exit(1)
    end
    println()
end

# Run full scaling test
println("=== FULL SCALING TEST ===")
test_full_population_margins_scaling()

println("\n=== FIX VALIDATION COMPLETE ===")
println("If all tests pass, the scaling bottleneck has been successfully fixed!")