# debug_clean.jl - Find the remaining O(n) allocation sources

using BenchmarkTools
using GLM
using DataFrames
using Margins
using Tables
import FormulaCompiler

function debug_remaining_allocations()
    println(" DEBUGGING REMAINING O(n) ALLOCATIONS")
    println("=" ^ 60)
    
    # Create test data
    n_small = 1000
    n_large = 5000
    
    data_small = DataFrame(x1 = randn(n_small), y = randn(n_small))
    data_large = DataFrame(x1 = randn(n_large), y = randn(n_large))
    
    model_small = lm(@formula(y ~ x1), data_small)
    model_large = lm(@formula(y ~ x1), data_large)
    
    println("Testing step by step...")
    println()
    
    # Step 1: Tables.columntable conversion
    println("Step 1: Tables.columntable conversion")
    bench_small = @benchmark Tables.columntable($data_small) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Tables.columntable($data_large) samples=10 evals=1 
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    println("  Small: $allocs_small allocs, Large: $allocs_large allocs, Ratio: $(round(scaling_ratio, digits=1))x")
    
    # Convert to column tables for further testing
    data_nt_small = Tables.columntable(data_small)
    data_nt_large = Tables.columntable(data_large)
    
    # Step 2: Engine building
    println("Step 2: Engine building")
    bench_small = @benchmark Margins.build_engine($model_small, $data_nt_small, [:x1]) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Margins.build_engine($model_large, $data_nt_large, [:x1]) samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    println("  Small: $allocs_small allocs, Large: $allocs_large allocs, Ratio: $(round(scaling_ratio, digits=1))x")
    
    # Build engines for further testing
    engine_small = Margins.build_engine(model_small, data_nt_small, [:x1])
    engine_large = Margins.build_engine(model_large, data_nt_large, [:x1])
    
    # Step 3: Core AME computation
    println("Step 3: Core AME computation")
    bench_small = @benchmark Margins._ame_continuous_and_categorical($engine_small, $data_nt_small; target=:mu, backend=:fd) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Margins._ame_continuous_and_categorical($engine_large, $data_nt_large; target=:mu, backend=:fd) samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    println("  Small: $allocs_small allocs, Large: $allocs_large allocs, Ratio: $(round(scaling_ratio, digits=1))x")
    
    if scaling_ratio > 2.0
        println("   FOUND THE CULPRIT: _ame_continuous_and_categorical is allocating O(n)")
    else
        println("   This step is not the problem")
    end
end

debug_remaining_allocations()