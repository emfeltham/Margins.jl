# debug_remaining_allocations.jl - Find the remaining O(n) allocation sources

using BenchmarkTools
using GLM
using DataFrames
using Margins
using Tables
import FormulaCompiler

function debug_remaining_allocations()
    println("🔍 DEBUGGING REMAINING O(n) ALLOCATIONS")
    println("=" ^ 60)
    
    # Create test data
    n_small = 1000
    n_large = 5000
    
    data_small = DataFrame(x1 = randn(n_small), y = randn(n_small))
    data_large = DataFrame(x1 = randn(n_large), y = randn(n_large))
    
    model_small = lm(@formula(y ~ x1), data_small)
    model_large = lm(@formula(y ~ x1), data_large)
    
    # Convert to column tables
    data_nt_small = Tables.columntable(data_small)
    data_nt_large = Tables.columntable(data_large)
    
    # Build engines
    engine_small = Margins.build_engine(model_small, data_nt_small, [:x1])
    engine_large = Margins.build_engine(model_large, data_nt_large, [:x1])
    
    println("Isolating allocation sources in population_margins pipeline...")
    println("\\nStep\\t\\t\\t\\t| Small (n=$n_small)\\t| Large (n=$n_large)\\t| Scaling")
    println("-" ^ 80)
    
    # Step 1: Tables.columntable conversion
    print("Tables.columntable\\t\\t| ")
    bench_small = @benchmark Tables.columntable(\$data_small) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Tables.columntable(\$data_large) samples=10 evals=1 
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    print("\$allocs_small\\t\\t| \$allocs_large\\t\\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - 5.0) < 1.0
        println("O(n) ❌")
    else
        println("\$(round(scaling_ratio, digits=1))x")
    end
    
    # Step 2: Engine building (should be O(1))
    print("Engine building\\t\\t\\t| ")
    bench_small = @benchmark Margins.build_engine(\$model_small, \$data_nt_small, [:x1]) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Margins.build_engine(\$model_large, \$data_nt_large, [:x1]) samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    print("\$allocs_small\\t\\t| \$allocs_large\\t\\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - 5.0) < 1.0
        println("O(n) ❌")
    else
        println("\$(round(scaling_ratio, digits=1))x")
    end
    
    # Step 3: Core AME computation
    print("_ame_continuous_and_categorical\\t| ")
    bench_small = @benchmark Margins._ame_continuous_and_categorical(\$engine_small, \$data_nt_small; target=:mu, backend=:fd) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark Margins._ame_continuous_and_categorical(\$engine_large, \$data_nt_large; target=:mu, backend=:fd) samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    print("\$allocs_small\\t\\t| \$allocs_large\\t\\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - 5.0) < 1.0
        println("O(n) ❌")
    else
        println("\$(round(scaling_ratio, digits=1))x")
    end
    
    # Step 4: Just the marginal effects loop (isolate the AME computation)
    rows_small = 1:n_small
    rows_large = 1:n_large
    
    print("Manual ME loop\\t\\t\\t| ")
    bench_small = @benchmark begin
        ame_val = 0.0
        for row in \$rows_small
            FormulaCompiler.marginal_effects_eta!((\$engine_small).g_buf, (\$engine_small).de, (\$engine_small).β, row; backend=:fd)
            ame_val += (\$engine_small).g_buf[1]
        end
    end samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        ame_val = 0.0
        for row in \$rows_large
            FormulaCompiler.marginal_effects_eta!((\$engine_large).g_buf, (\$engine_large).de, (\$engine_large).β, row; backend=:fd)
            ame_val += (\$engine_large).g_buf[1]
        end
    end samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    print("\$allocs_small\\t\\t| \$allocs_large\\t\\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - 5.0) < 1.0
        println("O(n) ❌")
    else
        println("\$(round(scaling_ratio, digits=1))x")
    end
    
    # Step 5: Just the elasticity computation part
    print("Elasticity computation\\t\\t| ")
    bench_small = @benchmark begin
        x_acc = 0.0
        y_acc = 0.0
        xcol = (\$data_nt_small).x1
        for row in \$rows_small
            x_acc += float(xcol[row])
            FormulaCompiler.modelrow!((\$engine_small).row_buf, (\$engine_small).compiled, \$data_nt_small, row)
            η = dot((\$engine_small).β, (\$engine_small).row_buf)
            y_acc += η
        end
    end samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        x_acc = 0.0
        y_acc = 0.0
        xcol = (\$data_nt_large).x1
        for row in \$rows_large
            x_acc += float(xcol[row])
            FormulaCompiler.modelrow!((\$engine_large).row_buf, (\$engine_large).compiled, \$data_nt_large, row)
            η = dot((\$engine_large).β, (\$engine_large).row_buf)
            y_acc += η
        end
    end samples=10 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    print("\$allocs_small\\t\\t| \$allocs_large\\t\\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - 5.0) < 1.0
        println("O(n) ❌")
    else
        println("\$(round(scaling_ratio, digits=1))x")
    end
    
    # Step 6: Result formatting
    print("DataFrame + MarginsResult\\t| ")
    results_df = DataFrame(term=["x1"], estimate=[1.0], se=[0.1])
    G = randn(1, 2)
    metadata = Dict{Symbol,Any}()
    
    bench_small = @benchmark MarginsResult(\$results_df, \$G, \$metadata) samples=20 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark MarginsResult(\$results_df, \$G, \$metadata) samples=20 evals=1
    allocs_large = minimum(bench_large).allocs
    
    scaling_ratio = allocs_large / allocs_small
    print("\$allocs_small\\t\\t| \$allocs_large\\t\\t| ")
    if scaling_ratio < 1.5
        println("O(1) ✅")
    elseif abs(scaling_ratio - 5.0) < 1.0
        println("O(n) ❌")
    else
        println("\$(round(scaling_ratio, digits=1))x")
    end
    
    println("\\n🎯 CONCLUSION:")
    println("The O(n) source should be identified above.")
    println("If Tables.columntable is O(n), the issue is data conversion.")
    println("If _ame_continuous_and_categorical is O(n), the issue is in our implementation.")
    println("If Manual ME loop is O(n), FormulaCompiler has O(n) allocations despite claims.")
    println("If Elasticity computation is O(n), our modelrow! usage is the problem.")
end

debug_remaining_allocations()