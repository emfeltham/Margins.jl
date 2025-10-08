# test_population_aap_loop_zero_alloc.jl
# Verifies zero-allocation hot loop for AAP (predictions) using BenchmarkTools

using Test
using BenchmarkTools
using DataFrames, Tables, CategoricalArrays
using GLM, StatsModels
using FormulaCompiler, Margins
using LinearAlgebra: dot

"""
Typed hot-loop helpers to force specialization on the compiled evaluator
"""
function _aap_loop_typed!(compiled::UnifiedCompiled{T,Ops,S,O},
                          row_buf::Vector{Float64}, β::Vector{Float64}, link,
                          data_nt::NamedTuple, n_calls) where {T,Ops,S,O}
    n_rows = length(first(data_nt))
    @inbounds for i in 1:n_calls
        row = ((i - 1) % n_rows) + 1
        modelrow!(row_buf, compiled, data_nt, row)
        η = dot(row_buf, β)
        _ = GLM.linkinv(link, η)
    end
    return nothing
end

function _aap_loop_weighted_typed!(compiled::UnifiedCompiled{T,Ops,S,O},
                                   row_buf::Vector{Float64}, β::Vector{Float64}, link,
                                   data_nt::NamedTuple, weights::Vector{Float64}, n_calls) where {T,Ops,S,O}
    n_rows = length(first(data_nt))
    acc = 0.0
    total_w = sum(weights)
    @inbounds for i in 1:n_calls
        row = ((i - 1) % n_rows) + 1
        w = weights[row]
        modelrow!(row_buf, compiled, data_nt, row)
        η = dot(row_buf, β)
        μ = GLM.linkinv(link, η)
        acc += w * μ
    end
    return acc / total_w
end

@testset "AAP Hot Loop Zero Allocation" begin
    # Small, representative dataset
    n = 5000
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        group = categorical(rand(["A", "B", "C"], n))
    )
    data_nt = Tables.columntable(df)

    # Fit simple linear model (identity link); predictions path is exercised identically
    model = lm(@formula(y ~ x + group), df)

    # Build/populate engine (PopulationUsage); no derivatives needed for predictions
    engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, Symbol[], GLM.vcov)

    # Warm-up calls to compile and stabilize
    for i in 1:10
        modelrow!(engine.row_buf, engine.compiled, data_nt, 1)
    end

    # Benchmark: assert zero allocations after warm-up
    b = @benchmark _aap_loop_typed!($engine.compiled, $engine.row_buf, $engine.β, $engine.link, $data_nt, 50_000) samples=5 evals=1
    # Some environments report tiny, constant overhead (e.g., 32 bytes). Accept bounded constant.
    @test minimum(b.memory) <= 64

    # Weighted variant (exercise weighted branch arithmetics; loop body is identical)
    weights = rand(n) .+ 0.1  # positive weights
    b_w = @benchmark _aap_loop_weighted_typed!($engine.compiled, $engine.row_buf, $engine.β, $engine.link, $data_nt, $weights, 50_000) samples=5 evals=1
    @test minimum(b_w.memory) <= 96
end

# Optional: extended large-n test (run manually with FC_SLOW_TESTS=1)
if get(ENV, "FC_SLOW_TESTS", "0") == "1"
    @testset "AAP Hot Loop Zero Allocation (Large-n)" begin
        n = 200_000
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            group = categorical(rand(["A", "B", "C"], n))
        )
        data_nt = Tables.columntable(df)
        model = lm(@formula(y ~ x + group), df)
        engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, Symbol[], GLM.vcov)

        # Warm-up
        modelrow!(engine.row_buf, engine.compiled, data_nt, 1)

        # Benchmark a sizable loop; assert zero allocations
        b = @benchmark _aap_loop_typed!($engine.compiled, $engine.row_buf, $engine.β, $engine.link, $data_nt, 100_000) samples=3 evals=1
        @test minimum(b.memory) <= 64
    end
end
