# debug_formulacompiler.jl - Investigate FormulaCompiler allocation sources

using BenchmarkTools
using GLM
using DataFrames
using Margins
using Tables
using LinearAlgebra
import FormulaCompiler

function debug_formulacompiler_allocations()
    println(" DEBUGGING FORMULACOMPILER ALLOCATION SOURCES")
    println("=" ^ 60)
    
    # Create test data
    n_small = 1000
    n_large = 5000
    
    data_small = DataFrame(x1 = randn(n_small), y = randn(n_small))
    data_large = DataFrame(x1 = randn(n_large), y = randn(n_large))
    
    model_small = lm(@formula(y ~ x1), data_small)
    model_large = lm(@formula(y ~ x1), data_large)
    
    data_nt_small = Tables.columntable(data_small)
    data_nt_large = Tables.columntable(data_large)
    
    engine_small = Margins.build_engine(model_small, data_nt_small, [:x1])
    engine_large = Margins.build_engine(model_large, data_nt_large, [:x1])
    
    rows_small = 1:n_small
    rows_large = 1:n_large
    
    println("Testing individual FormulaCompiler function calls:")
    println()
    
    # Test 1: FormulaCompiler.accumulate_ame_gradient!
    println("1. FormulaCompiler.accumulate_ame_gradient!")
    fill!(engine_small.gβ_accumulator, 0.0)
    fill!(engine_large.gβ_accumulator, 0.0)
    
    bench_small = @benchmark FormulaCompiler.accumulate_ame_gradient!(
        ($engine_small).gβ_accumulator, ($engine_small).de, ($engine_small).β, $rows_small, :x1;
        link=GLM.IdentityLink(), backend=:fd
    ) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark FormulaCompiler.accumulate_ame_gradient!(
        ($engine_large).gβ_accumulator, ($engine_large).de, ($engine_large).β, $rows_large, :x1;
        link=GLM.IdentityLink(), backend=:fd
    ) samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    print_result("accumulate_ame_gradient!", allocs_small, allocs_large, n_small, n_large)
    
    # Test 2: FormulaCompiler.marginal_effects_eta! in loop
    println("2. FormulaCompiler.marginal_effects_eta! loop")
    
    bench_small = @benchmark begin
        for row in $rows_small
            FormulaCompiler.marginal_effects_eta!(($engine_small).g_buf, ($engine_small).de, ($engine_small).β, row; backend=:fd)
        end
    end samples=5 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        for row in $rows_large
            FormulaCompiler.marginal_effects_eta!(($engine_large).g_buf, ($engine_large).de, ($engine_large).β, row; backend=:fd)
        end
    end samples=5 evals=1
    allocs_large = minimum(bench_large).allocs
    
    print_result("marginal_effects_eta! loop", allocs_small, allocs_large, n_small, n_large)
    
    # Test 3: FormulaCompiler.modelrow! in loop  
    println("3. FormulaCompiler.modelrow! loop")
    
    bench_small = @benchmark begin
        for row in $rows_small
            FormulaCompiler.modelrow!(($engine_small).row_buf, ($engine_small).compiled, $data_nt_small, row)
        end
    end samples=5 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        for row in $rows_large
            FormulaCompiler.modelrow!(($engine_large).row_buf, ($engine_large).compiled, $data_nt_large, row)
        end
    end samples=5 evals=1
    allocs_large = minimum(bench_large).allocs
    
    print_result("modelrow! loop", allocs_small, allocs_large, n_small, n_large)
    
    # Test 4: Just the dot product operations in loop
    println("4. Dot product operations in loop")
    
    bench_small = @benchmark begin
        acc = 0.0
        for row in $rows_small
            FormulaCompiler.modelrow!(($engine_small).row_buf, ($engine_small).compiled, $data_nt_small, row)
            acc += dot(($engine_small).β, ($engine_small).row_buf)
        end
    end samples=5 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        acc = 0.0
        for row in $rows_large
            FormulaCompiler.modelrow!(($engine_large).row_buf, ($engine_large).compiled, $data_nt_large, row)
            acc += dot(($engine_large).β, ($engine_large).row_buf)
        end
    end samples=5 evals=1
    allocs_large = minimum(bench_large).allocs
    
    print_result("dot product loop", allocs_small, allocs_large, n_small, n_large)
    
    # Test 5: FormulaCompiler.delta_method_se calls
    println("5. FormulaCompiler.delta_method_se")
    
    # Fill with dummy gradient
    fill!(engine_small.gβ_accumulator, 1.0)
    fill!(engine_large.gβ_accumulator, 1.0)
    
    bench_small = @benchmark FormulaCompiler.delta_method_se(($engine_small).gβ_accumulator, ($engine_small).Σ) samples=20 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark FormulaCompiler.delta_method_se(($engine_large).gβ_accumulator, ($engine_large).Σ) samples=20 evals=1
    allocs_large = minimum(bench_large).allocs
    
    print_result("delta_method_se", allocs_small, allocs_large, n_small, n_large)
    
    println("\n ANALYSIS:")
    println("If any of the above show O(n) scaling, that's the allocation source.")
    println("Expected: accumulate_ame_gradient! and delta_method_se should be O(1)")
    println("Suspect: marginal_effects_eta! loop or modelrow! loop may be O(n)")
end

function print_result(name, allocs_small, allocs_large, n_small, n_large)
    scaling_ratio = allocs_large / allocs_small
    size_ratio = n_large / n_small
    
    println("  Small (n=$n_small): $allocs_small allocs")
    println("  Large (n=$n_large): $allocs_large allocs") 
    println("  Scaling: $(round(scaling_ratio, digits=1))x (expected $(round(size_ratio, digits=1))x for O(n))")
    
    if allocs_small == 0 && allocs_large == 0
        println("   Zero allocations - perfect!")
    elseif scaling_ratio < 1.5
        println("   O(1) scaling - not the problem")
    elseif abs(scaling_ratio - size_ratio) < 0.5
        println("   O(n) scaling - THIS IS THE CULPRIT!")
    else
        println("    Unclear scaling pattern")
    end
    println()
end

debug_formulacompiler_allocations()