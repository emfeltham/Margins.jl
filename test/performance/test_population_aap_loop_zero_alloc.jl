# test_population_aap_loop_zero_alloc.jl
#
# Allocation of the population adjusted-prediction (AAP) path: the per-row prediction loop
# on an engine's compiled formula is allocation-free, and a warm
# `population_margins(...; type=:predictions)` call allocates the same amount at n = 1,000 and
# n = 50,000 (O(1) in rows), unweighted and weighted.
#
# (Rewritten in 3.0.0: it used a removed 5-argument `get_or_build_engine` and was not run by
# runtests.jl.)

using Test
using DataFrames, Tables, CategoricalArrays, Random
using GLM, StatsModels
using FormulaCompiler, Margins
using LinearAlgebra: dot

function _aap_loop!(compiled, row_buf::Vector{Float64}, β::Vector{Float64}, link,
                    data_nt::NamedTuple, rows)
    acc = 0.0
    @inbounds for row in rows
        modelrow!(row_buf, compiled, data_nt, row)
        acc += GLM.linkinv(link, dot(row_buf, β))
    end
    return acc
end

function _aap_data(n)
    rng = MersenneTwister(n)
    df = DataFrame(x = randn(rng, n), k = rand(rng, 1:5, n),
                   group = categorical(rand(rng, ["A", "B", "C"], n)))
    df.y = rand(rng, n) .< 1 ./ (1 .+ exp.(-(0.4 .* df.x .+ 0.1 .* df.k)))
    df.w = rand(rng, n) .+ 0.1
    return df
end

@testset "AAP: allocation independent of rows" begin
    df = _aap_data(5_000)
    model = glm(@formula(y ~ x + k + group), df, Binomial(), LogitLink())
    data_nt = Margins._prepare_margins_data(Tables.columntable(df), model)
    engine = Margins.build_engine(Margins.PopulationUsage, Margins.NoDerivatives, model, data_nt,
                                  Symbol[], GLM.vcov, :ad)
    # (measured inside a function: on Julia 1.10, @allocated at testset scope counts boxing)
    alloc_loop(e, d, rows) = @allocated _aap_loop!(e.compiled, e.row_buf, e.β, e.link, d, rows)
    _aap_loop!(engine.compiled, engine.row_buf, engine.β, engine.link, data_nt, 1:10)
    alloc_loop(engine, data_nt, 1:10)
    @test alloc_loop(engine, data_nt, 1:5_000) == 0

    allocs = Dict{Tuple{Int, Bool}, Int}()
    for n in (1_000, 50_000)
        d = _aap_data(n)
        m = glm(@formula(y ~ x + k + group), d, Binomial(), LogitLink())
        for weighted in (false, true)
            w = weighted ? :w : nothing
            population_margins(m, d; type=:predictions, weights=w)
            population_margins(m, d; type=:predictions, weights=w)
            allocs[(n, weighted)] = @allocated population_margins(m, d; type=:predictions, weights=w)
        end
    end
    for weighted in (false, true)
        # The integer column k is converted to Float64 once and memoized, so a warm call does
        # not copy it: allocation must be far below one column's worth (8 × 50,000 bytes)
        @test allocs[(50_000, weighted)] < allocs[(1_000, weighted)] + 20_000
        @test allocs[(50_000, weighted)] < 8 * 50_000 ÷ 4
    end
end
