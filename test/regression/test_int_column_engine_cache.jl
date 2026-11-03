# test/regression/test_int_column_engine_cache.jl
#
# Regression test: continuous Int (and Float32) columns missed the engine cache.
# `_convert_numeric_to_float64` built a new Float64 copy of every non-Float64 numeric column
# on each call, and the engine cache revalidates by column identity, so every call rebuilt
# the engine and copied the column (O(n) allocation per warm call). The copy is now memoized
# per source array (`_memoized_transform`), reused while the source's contents are unchanged.

using Test
using DataFrames, GLM, StatsModels, Random, Statistics, Tables
using Margins

@testset "continuous Int/Float32 columns hit the engine cache" begin
    Random.seed!(3); n = 100_000
    d = DataFrame(x = randn(n), k = rand(0:20, n), f = Float32.(randn(n)))
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.3 .* d.x .+ 0.05 .* d.k .- 0.2 .* d.f)))
    m = glm(@formula(y ~ x + k + f), d, Binomial(), LogitLink())

    builds = Ref(0)
    count_builds(f) = (before = Margins._cached_engines(); r = f();
                       builds[] += count(e -> !any(b -> b === e, before), Margins._cached_engines()); r)

    Margins.clear_engine_cache!()
    r1 = DataFrame(count_builds(() -> population_margins(m, d)))
    e1 = only(Margins._cached_engines())
    r2 = DataFrame(count_builds(() -> population_margins(m, d)))
    r3 = DataFrame(count_builds(() -> population_margins(m, d)))
    @test builds[] == 1                                  # built once
    @test only(Margins._cached_engines()) === e1         # and reused
    @test r1.estimate == r2.estimate == r3.estimate
    @test r1.se == r2.se == r3.se
    # The converted column is the one the engine holds, not a fresh copy each call
    @test e1.data_nt.k === Margins._convert_numeric_to_float64(Tables.columntable(d)).k

    # O(1) when warm: far below one converted column (8n bytes)
    a = @allocated population_margins(m, d)
    @test a < 8n ÷ 10

    # Reference: the same data with Float64 columns
    dd = transform(d, :k => ByRow(Float64) => :k, :f => ByRow(Float64) => :f)
    rf = DataFrame(population_margins(m, dd))
    @test r1.estimate ≈ rf.estimate rtol=1e-12
    @test r1.se ≈ rf.se rtol=1e-12

    # In-place mutation of the source is picked up (and matches a fresh Float64 frame)
    d.k[1:1000] .= 20
    dd.k[1:1000] .= 20.0
    r4 = DataFrame(population_margins(m, d))
    @test r4.estimate != r1.estimate
    @test r4.estimate ≈ DataFrame(population_margins(m, dd)).estimate rtol=1e-12
    d.f[1:1000] .= 2.5f0
    dd.f[1:1000] .= 2.5
    @test DataFrame(population_margins(m, d)).estimate ≈ DataFrame(population_margins(m, dd)).estimate rtol=1e-12

    # Values that do not round-trip through Float64 (|k| > 2^53) are never taken from a
    # stale copy: the staleness check compares exactly
    big = [2^53 + 1, 3]
    c1 = Margins._convert_numeric_to_float64((k = big,)).k
    c2 = Margins._convert_numeric_to_float64((k = big,)).k
    @test c1 == Float64.(big) && c2 == Float64.(big)

    # clear_engine_cache! also drops the memoized conversions
    Margins.clear_engine_cache!()
    @test isempty(Margins._NARROWED_COLUMNS)
end
