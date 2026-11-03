# test/validation/test_no_engine_cache.jl
#
# Margins 3.0 keeps no cache of its own: every call builds its own engine. Through 3.0
# development a bounded engine cache (LRU, `set_engine_cache_capacity!`) kept engines — and
# through them the datasets they were built from — across calls; before that it was
# unbounded (a 40-replicate bootstrap at n = 50,000 retained ~320 MB). These tests check what
# replaces it: nothing Margins holds keeps a dataset alive, repeated calls are identical, and
# `clear_engine_cache!()` remains as an alias that clears FormulaCompiler's compiled cache.

using Test
using DataFrames, GLM, StatsModels, Random, Statistics
using Margins

function _bound_data(seed; n = 400)
    rng = MersenneTwister(seed)
    d = DataFrame(x = randn(rng, n), z = randn(rng, n))
    d.y = rand(rng, n) .< 1 ./ (1 .+ exp.(-(0.5 .* d.x .- 0.3 .* d.z)))
    return d
end
_bound_model(d) = glm(@formula(y ~ x + z), d, Binomial(), LogitLink())

# One bootstrap-like replicate; returns a WeakRef to a column of its data that the model does
# not hold (the model frame keeps its own columns, and FormulaCompiler's cache keeps models)
function _replicate!(seed)
    d = _bound_data(seed)
    m = _bound_model(d)
    dd = copy(d)
    population_margins(m, dd; vars=[:x])
    return WeakRef(dd.x)
end

@testset "no engine cache" begin
    @testset "bootstrap loop: data not retained" begin
        refs = [_replicate!(s) for s in 100:140]
        GC.gc(); GC.gc()
        @test all(r -> r.value === nothing, refs)
    end

    @testset "repeated calls are identical" begin
        d = _bound_data(31); m = _bound_model(d)
        r1 = DataFrame(population_margins(m, d))
        r2 = DataFrame(population_margins(m, d))
        @test r1.estimate == r2.estimate
        @test r1.se == r2.se
        g = means_grid(d)
        @test DataFrame(profile_margins(m, d, g)).se == DataFrame(profile_margins(m, d, g)).se
    end

    @testset "build_engine rejects unknown variables" begin
        d = _bound_data(51); m = _bound_model(d)
        nt = Margins._prepare_margins_data(Margins.Tables.columntable(d), m)
        @test_throws Margins.MarginsError Margins.build_engine(Margins.PopulationUsage, m, nt, [:nope], GLM.vcov, :ad)
    end

    @testset "clear_engine_cache! clears FormulaCompiler's cache" begin
        df_fc = DataFrame(y = randn(50), x = randn(50))
        m_fc = lm(@formula(y ~ x), df_fc)
        population_margins(m_fc, df_fc; type = :effects)
        @test Margins.FormulaCompiler.cache_info().entries >= 1
        @test Margins.clear_engine_cache!() === nothing
        @test Margins.FormulaCompiler.cache_info().entries == 0
        # results unchanged after clearing
        r = DataFrame(population_margins(m_fc, df_fc))
        Margins.clear_engine_cache!()
        @test DataFrame(population_margins(m_fc, df_fc)).se == r.se
    end

    @testset "removed cache names: UndefVarError with a hint" begin
        for name in (:get_cache_stats, :set_engine_cache_capacity!, :engine_cache_capacity,
                     :get_or_build_engine, :ENGINE_CACHE, :VALIDATED_MODELS_CACHE)
            @test !isdefined(Margins, name)
            err = try
                getfield(Margins, name)
                nothing
            catch e
                e
            end
            @test err isa UndefVarError
            @test occursin("removed in Margins 3.0", sprint(showerror, err))
        end
    end
end
