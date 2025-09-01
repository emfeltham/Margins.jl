# debug_ame_allocations.jl - Deep dive into AME allocation sources

using BenchmarkTools
using GLM
using DataFrames
using Margins
import FormulaCompiler
using LinearAlgebra

function test_ame_components()
    println("🔍 DEBUGGING AME ALLOCATION SOURCES")
    println("=" ^ 60)
    
    # Create test data
    n_small = 1000
    n_large = 5000
    
    data_small = DataFrame(x1 = randn(n_small), y = randn(n_small))
    data_large = DataFrame(x1 = randn(n_large), y = randn(n_large))
    
    model_small = lm(@formula(y ~ x1), data_small)
    model_large = lm(@formula(y ~ x1), data_large)
    
    # Build engines
    data_nt_small = Tables.columntable(data_small)
    data_nt_large = Tables.columntable(data_large)
    
    engine_small = Margins.build_engine(model_small, data_nt_small, [:x1])
    engine_large = Margins.build_engine(model_large, data_nt_large, [:x1])
    
    println("Component analysis...")
    println("\nComponent\t\t\t| Small (n=$n_small)\t| Large (n=$n_large)\t| Scaling")
    println("-" ^ 80)
    
    # Test the pure FormulaCompiler AME function directly
    print("FC accumulate_ame_gradient!\t| ")
    
    rows_small = 1:n_small
    rows_large = 1:n_large
    
    # Pre-fill accumulator buffers
    fill!(engine_small.gβ_accumulator, 0.0)
    fill!(engine_large.gβ_accumulator, 0.0)
    
    bench_small = @benchmark FormulaCompiler.accumulate_ame_gradient!(
        $(engine_small.gβ_accumulator), $(engine_small.de), $(engine_small.β), $rows_small, :x1;
        link=GLM.IdentityLink(), backend=:fd
    ) samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark FormulaCompiler.accumulate_ame_gradient!(
        $(engine_large.gβ_accumulator), $(engine_large.de), $(engine_large.β), $rows_large, :x1;
        link=GLM.IdentityLink(), backend=:fd
    ) samples=10 evals=1
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
    
    # Test the marginal effects computation loop
    print("Marginal effects loop\t\t| ")
    
    bench_small = @benchmark begin
        ame_val = 0.0
        for row in $rows_small
            FormulaCompiler.marginal_effects_eta!($(engine_small.g_buf), $(engine_small.de), $(engine_small.β), row; backend=:fd)
            ame_val += $(engine_small.g_buf)[1]
        end
    end samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        ame_val = 0.0  
        for row in $rows_large
            FormulaCompiler.marginal_effects_eta!($(engine_large.g_buf), $(engine_large.de), $(engine_large.β), row; backend=:fd)
            ame_val += $(engine_large.g_buf)[1]
        end
    end samples=10 evals=1
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
    
    # Test just the DataFrame construction and result processing
    print("DataFrame operations\t\t| ")
    
    bench_small = @benchmark begin
        results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
        push!(results, (term="x1", estimate=1.0, se=0.1))
        G = Matrix{Float64}(undef, 1, length($(engine_small.β)))
        G[1, :] .= 1.0
        MarginsResult(results, G, Dict{Symbol,Any}())
    end samples=20 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
        push!(results, (term="x1", estimate=1.0, se=0.1))
        G = Matrix{Float64}(undef, 1, length($(engine_large.β)))
        G[1, :] .= 1.0
        MarginsResult(results, G, Dict{Symbol,Any}())
    end samples=20 evals=1
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
    
    # Test just the elasticity computation part which has loops over rows
    print("Elasticity computation\t\t| ")
    
    bench_small = @benchmark begin
        x_acc = 0.0
        y_acc = 0.0
        xcol = $(data_nt_small.x1)
        for row in $rows_small
            x_acc += float(xcol[row])
            FormulaCompiler.modelrow!($(engine_small.row_buf), $(engine_small.compiled), $data_nt_small, row)
            η = dot($(engine_small.β), $(engine_small.row_buf))
            y_acc += η
        end
    end samples=10 evals=1
    allocs_small = minimum(bench_small).allocs
    
    bench_large = @benchmark begin
        x_acc = 0.0
        y_acc = 0.0
        xcol = $(data_nt_large.x1)
        for row in $rows_large
            x_acc += float(xcol[row])
            FormulaCompiler.modelrow!($(engine_large.row_buf), $(engine_large.compiled), $data_nt_large, row)
            η = dot($(engine_large.β), $(engine_large.row_buf))
            y_acc += η
        end
    end samples=10 evals=1
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
    
    println("\n🎯 CONCLUSION:")
    println("If FC accumulate_ame_gradient! is O(n), the allocations are coming from FormulaCompiler")
    println("If elasticity computation is O(n), the allocations are from our modelrow! loop")
    println("If both are O(1), then the allocations are elsewhere in the pipeline")
end

test_ame_components()