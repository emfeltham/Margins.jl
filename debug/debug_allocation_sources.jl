# debug_allocation_sources.jl - Identify sources of O(n) scaling

using BenchmarkTools
using GLM
using DataFrames
using Margins

function debug_allocation_sources()
    println("🔍 DEBUGGING O(n) ALLOCATION SOURCES")
    println("=" ^ 60)
    
    # Create two test sizes to compare
    n_small = 1000
    n_large = 5000
    
    # Small dataset
    data_small = DataFrame(x1 = randn(n_small), x2 = randn(n_small), y = randn(n_small))
    model_small = lm(@formula(y ~ x1 + x2), data_small)
    
    # Large dataset  
    data_large = DataFrame(x1 = randn(n_large), x2 = randn(n_large), y = randn(n_large))
    model_large = lm(@formula(y ~ x1 + x2), data_large)
    
    println("Testing component allocations...")
    println("\nComponent\t\t\t| Small (n=$n_small)\t| Large (n=$n_large)\t| Scaling")
    println("-" ^ 80)
    
    # Test individual components to isolate allocation sources
    
    # 1. Engine building (should be O(1))
    print("Engine building\t\t\t| ")
    data_nt_small = Tables.columntable(data_small)
    data_nt_large = Tables.columntable(data_large)
    
    bench_small = @benchmark Margins.build_engine($model_small, $data_nt_small, [:x1]) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Margins.build_engine($model_large, $data_nt_large, [:x1]) samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    size_ratio = n_large / n_small
    
    print("$allocs_small\t\t| $allocs_large\t\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - size_ratio) < 0.3
        println("O(n) ❌") 
    else
        println("$(round(scaling_ratio, digits=1))x")
    end
    
    # 2. FormulaCompiler operations (should be O(1) after engine build)
    engine_small = Margins.build_engine(model_small, data_nt_small, [:x1])
    engine_large = Margins.build_engine(model_large, data_nt_large, [:x1])
    
    print("AME computation (FC)\t\t| ")
    # Test the core AME computation from FormulaCompiler
    bench_small = @benchmark Margins._ame_continuous_and_categorical($engine_small, $data_nt_small; backend=:fd) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Margins._ame_continuous_and_categorical($engine_large, $data_nt_large; backend=:fd) samples=10 evals=1  
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    
    print("$allocs_small\t\t| $allocs_large\t\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - size_ratio) < 0.3
        println("O(n) ❌")
    else
        println("$(round(scaling_ratio, digits=1))x")
    end
    
    # 3. DataFrame creation and result formatting
    print("Result DataFrame creation\t| ")
    # Mock some results to test DataFrame overhead
    results_small = DataFrame(term=["x1"], estimate=[1.0], se=[0.1])
    G_small = randn(1, 2)
    
    results_large = DataFrame(term=["x1"], estimate=[1.0], se=[0.1])  
    G_large = randn(1, 2)
    
    bench_small = @benchmark MarginsResult($results_small, $G_small, Dict{Symbol,Any}()) samples=20 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark MarginsResult($results_large, $G_large, Dict{Symbol,Any}()) samples=20 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    
    print("$allocs_small\t\t| $allocs_large\t\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - size_ratio) < 0.3
        println("O(n) ❌")
    else
        println("$(round(scaling_ratio, digits=1))x")
    end
    
    # 4. Test the full pipeline to see where scaling comes from
    print("Full population_margins\t\t| ")
    
    # Warmup
    population_margins(model_small, data_small; backend=:fd, vars=[:x1])
    population_margins(model_large, data_large; backend=:fd, vars=[:x1])
    
    bench_small = @benchmark population_margins($model_small, $data_small; backend=:fd, vars=[:x1]) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark population_margins($model_large, $data_large; backend=:fd, vars=[:x1]) samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    
    print("$allocs_small\t\t| $allocs_large\t\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - size_ratio) < 0.3
        println("O(n) ❌")
    else
        println("$(round(scaling_ratio, digits=1))x")
    end
    
    println("\n🔍 DETAILED ANALYSIS:")
    println("=" ^ 60)
    
    # The O(n) scaling is most likely coming from:
    println("🎯 LIKELY O(n) SOURCES:")
    println("1. FormulaCompiler modelrow() calls in loops")
    println("   → Each row needs design matrix computation")
    println("   → X_row = FormulaCompiler.modelrow(compiled, data_nt, i)")
    println("   → May allocate per row despite buffer attempts")
    
    println("\n2. GLM.jl link function calls")
    println("   → GLM.linkinv() and GLM.mueta() per observation") 
    println("   → May have hidden allocations")
    
    println("\n3. Dot product and mathematical operations")
    println("   → dot(X_row, engine.β) per observation")
    println("   → Accumulation operations: G .+= ...")
    
    println("\n4. Buffer boundary issues")
    println("   → Our buffer reuse may not be working as expected")
    println("   → views() might still allocate in some contexts")
    
    println("\n💡 INVESTIGATION NEEDED:")
    println("• Profile FormulaCompiler.modelrow() allocation behavior")
    println("• Check if GLM functions allocate per call") 
    println("• Verify buffer reuse is actually working")
    println("• Consider batch operations instead of row-by-row processing")
end

debug_allocation_sources()