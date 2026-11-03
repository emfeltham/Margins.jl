# test/validation/test_engine_cache_bound.jl
#
# The engine cache is bounded (LRU over cache keys, `set_engine_cache_capacity!`). Before
# 3.0.0 it kept every engine — and through it every dataset — for the whole session (a
# 40-replicate bootstrap at n = 50,000 retained ~320 MB).

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

# One bootstrap-like replicate; returns a WeakRef to its data column
function _replicate!(seed)
    d = _bound_data(seed)
    population_margins(_bound_model(d), d; vars=[:x])
    return WeakRef(d.x)
end

@testset "engine cache: bounded LRU" begin
    @test Margins.engine_cache_capacity() == 8
    old = Margins.set_engine_cache_capacity!(3)
    try
        @test old == 8
        @test Margins.engine_cache_capacity() == 3
        @test_throws ArgumentError Margins.set_engine_cache_capacity!(-1)

        @testset "bound holds; most recently used kept" begin
            Margins.clear_engine_cache!()
            ds = [_bound_data(s) for s in 1:6]; ms = _bound_model.(ds)
            for (m, d) in zip(ms, ds)
                population_margins(m, d; vars=[:x])
                @test Margins.get_cache_stats().entries <= 3
            end
            kept = Margins._cached_engines()
            @test length(kept) == 3
            @test Set(objectid.(e.model for e in kept)) == Set(objectid.(ms[4:6]))
        end

        @testset "LRU order: a hit refreshes its entry" begin
            Margins.set_engine_cache_capacity!(2)
            Margins.clear_engine_cache!()
            ds = [_bound_data(s) for s in 11:13]; ms = _bound_model.(ds)
            population_margins(ms[1], ds[1]; vars=[:x]); e1 = only(Margins._cached_engines())
            population_margins(ms[2], ds[2]; vars=[:x])
            population_margins(ms[1], ds[1]; vars=[:x])        # hit: m1 most recent
            population_margins(ms[3], ds[3]; vars=[:x])        # evicts m2, not m1
            kept = Margins._cached_engines()
            @test any(e -> e === e1, kept)
            @test Set(objectid.(e.model for e in kept)) == Set(objectid.(ms[[1, 3]]))
        end

        @testset "same key, new data replaces (does not accumulate)" begin
            Margins.set_engine_cache_capacity!(16)
            Margins.clear_engine_cache!()
            d = _bound_data(21); m = _bound_model(d)
            for i in 1:5
                population_margins(m, d[shuffle(MersenneTwister(i), 1:nrow(d)), :]; vars=[:x])
            end
            @test Margins.get_cache_stats().entries == 1
        end

        @testset "bootstrap loop: memory released" begin
            Margins.set_engine_cache_capacity!(4)
            Margins.clear_engine_cache!()
            refs = [_replicate!(s) for s in 100:140]
            @test Margins.get_cache_stats().entries <= 4
            GC.gc(); GC.gc()
            # Datasets of evicted replicates are no longer reachable from the cache
            @test count(r -> r.value === nothing, refs) >= length(refs) - 4
        end

        @testset "capacity 0 disables caching; results unchanged" begin
            Margins.set_engine_cache_capacity!(16)
            Margins.clear_engine_cache!()
            d = _bound_data(31); m = _bound_model(d)
            r_cached = DataFrame(population_margins(m, d))
            Margins.set_engine_cache_capacity!(0)
            @test Margins.get_cache_stats().entries == 0          # evicted immediately
            r_uncached = DataFrame(population_margins(m, d))
            @test Margins.get_cache_stats().entries == 0
            @test r_uncached.estimate == r_cached.estimate
            @test r_uncached.se == r_cached.se
        end

        @testset "warm repeated calls: one engine, no extra allocation" begin
            Margins.set_engine_cache_capacity!(16)
            Margins.clear_engine_cache!()
            d = _bound_data(41; n = 20_000); m = _bound_model(d)
            population_margins(m, d; vars=[:x]); e = only(Margins._cached_engines())
            population_margins(m, d; vars=[:x])
            a = @allocated population_margins(m, d; vars=[:x])
            @test only(Margins._cached_engines()) === e
            @test a < 8 * 20_000 ÷ 4          # far below one column: O(1) in rows
        end

        @testset "failed build releases its reservation" begin
            Margins.clear_engine_cache!()
            d = _bound_data(51); m = _bound_model(d)
            nt = Margins._prepare_margins_data(Margins.Tables.columntable(d), m)
            @test_throws Margins.MarginsError Margins._checkout_engine(Margins.PopulationUsage,
                Margins.HasDerivatives, m, nt, [:nope], GLM.vcov, :ad)
            @test Margins.get_cache_stats().entries == 0
            @test isempty(Margins.ENGINE_CACHE)
            # check-out / check-in bookkeeping
            (e, key) = Margins._checkout_engine(Margins.PopulationUsage, Margins.HasDerivatives,
                                                m, nt, [:x], GLM.vcov, :ad)
            @test Margins.ENGINE_CACHE[key].checked_out == 1
            @test isempty(Margins._cached_engines())             # not idle while in use
            Margins._checkin_engine!(key, e)
            @test Margins.ENGINE_CACHE[key].checked_out == 0
            @test only(Margins._cached_engines()) === e
        end

        @testset "clear_engine_cache! empties engines and data memo" begin
            d = _bound_data(61); d.k = rand(MersenneTwister(1), 1:4, nrow(d))
            m = glm(@formula(y ~ x + k), d, Binomial(), LogitLink())
            population_margins(m, d)
            @test Margins.get_cache_stats().entries >= 1
            @test !isempty(Margins._NARROWED_COLUMNS)
            # A call in progress keeps its engine; clearing meanwhile discards it on return
            nt = Margins._prepare_margins_data(Margins.Tables.columntable(d), m)
            (e, key) = Margins._checkout_engine(Margins.PopulationUsage, Margins.HasDerivatives,
                                                m, nt, [:x], GLM.vcov, :ad)
            Margins.clear_engine_cache!()
            @test isempty(Margins.ENGINE_CACHE) && isempty(Margins._NARROWED_COLUMNS)
            Margins._checkin_engine!(key, e)
            @test isempty(Margins.ENGINE_CACHE)
        end
    finally
        Margins.set_engine_cache_capacity!(old)
        Margins.clear_engine_cache!()
    end

    # clear_engine_cache! also clears FormulaCompiler's compiled-formula cache
    @testset "clear_engine_cache! clears FormulaCompiler's cache" begin
        df_fc = DataFrame(y = randn(50), x = randn(50))
        m_fc = lm(@formula(y ~ x), df_fc)
        population_margins(m_fc, df_fc; type = :effects)
        @test Margins.FormulaCompiler.cache_info().entries >= 1
        Margins.clear_engine_cache!()
        @test Margins.FormulaCompiler.cache_info().entries == 0
    end
end
