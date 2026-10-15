# test/regression/test_data_prep_memo.jl
#
# Regression tests for the memo of converted data columns (`_NARROWED_COLUMNS`), which keeps
# repeated calls on `Union{Missing,T}` columns and on model-categorical Int/String columns
# hitting the engine cache:
#  * the staleness check on a cache hit ran on abstractly typed values and allocated O(n)
#    per call (28.8 MB per warm call at n = 200,000); it now runs behind a function barrier;
#  * `clear_engine_cache!()` did not empty the memo.

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Random
using Margins

@testset "data-preparation memo: O(1) warm calls, cleared with the engine cache" begin
    Random.seed!(1); n = 200_000
    df = DataFrame(x = allowmissing(randn(n)), k = rand(1:3, n), s = rand(["u", "v"], n))
    df.y = df.x .+ randn(n)
    m = lm(@formula(y ~ x + k + s), dropmissing(df); contrasts = Dict(:k => DummyCoding()))
    population_margins(m, df; vars=[:x]); population_margins(m, df; vars=[:x])
    a_conv = @allocated population_margins(m, df; vars=[:x])

    dfp = DataFrame(x = randn(n), k = categorical(rand(1:3, n)), s = categorical(rand(["u", "v"], n)))
    dfp.y = dfp.x .+ randn(n)
    mp = lm(@formula(y ~ x + k + s), dfp)
    population_margins(mp, dfp; vars=[:x]); population_margins(mp, dfp; vars=[:x])
    a_plain = @allocated population_margins(mp, dfp; vars=[:x])

    # O(1): far below one column's worth of memory (8n bytes = 1.6 MB), and comparable
    # to plain CategoricalArray data
    @test a_conv < 8n ÷ 10
    @test a_conv < 4 * a_plain + 200_000

    @test !isempty(Margins._NARROWED_COLUMNS)
    Margins.clear_engine_cache!()
    @test isempty(Margins._NARROWED_COLUMNS)
    @test isempty(Margins.ENGINE_CACHE)
end
