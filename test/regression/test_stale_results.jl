# test/regression/test_stale_results.jl
#
# Regression tests for stale results from Margins' process-wide caches, removed in 3.0
# (every call now builds its own engine, validates its inputs and evaluates `vcov`):
#
#  * B1: the engine cache was keyed on the model object, so after MixedModels `refit!` (which
#    mutates the model in place) the next call returned the OLD estimate and SE.
#  * B2: a `vcov` matrix was keyed by `hash(V)`, and `hash` of a large array samples its
#    entries, so an in-place edit of an unsampled entry left the key unchanged and the next
#    call returned the old SE. At p = 201 most single-entry edits are invisible to `hash`.
#  * `VALIDATED_MODELS_CACHE` used the same sampled-hash key, so an edited matrix that is no
#    longer valid (non-finite) skipped validation.
#
# Each expected value comes from a fresh fit / closed form, never from a second Margins call
# on the same objects.

using Test
using DataFrames, GLM, StatsModels, Random, LinearAlgebra
using MixedModels
using Margins

@testset "no stale results after refit! or in-place vcov edits" begin

    @testset "B1: MixedModels refit! gives the new estimate" begin
        rng = MersenneTwister(20261019)
        n, ng = 600, 30
        d = DataFrame(x = randn(rng, n), g = string.(rand(rng, 1:ng, n)))
        u = randn(rng, ng)
        gi = parse.(Int, d.g)
        d.y = 0.5 .* d.x .+ u[gi] .+ randn(rng, n)
        y_new = 4.0 .* d.x .+ u[gi] .+ randn(rng, n)
        f = @formula(y ~ x + (1 | g))

        lmm = fit(MixedModel, f, d; progress = false)
        r_old = DataFrame(population_margins(lmm, d; vars = [:x]))
        @test r_old.estimate[1] ≈ coef(lmm)[2] atol = 1e-10       # identity link: AME = β_x

        refit!(lmm, y_new; progress = false)
        d_new = transform(d, :x => (_ -> y_new) => :y)
        fresh = fit(MixedModel, f, d_new; progress = false)
        r_new = DataFrame(population_margins(lmm, d; vars = [:x]))
        @test r_new.estimate[1] ≈ coef(fresh)[2] atol = 1e-6       # was the stale r_old value
        @test r_new.se[1] ≈ sqrt(vcov(fresh)[2, 2]) rtol = 1e-4
        @test abs(r_new.estimate[1] - r_old.estimate[1]) > 1       # 0.5 vs 4

        # predictions and profile margins too
        p_new = DataFrame(profile_margins(lmm, d, cartesian_grid(x = [1.0]); type = :predictions))
        @test p_new.estimate[1] ≈ coef(fresh)[1] + coef(fresh)[2] atol = 1e-6
    end

    # A linear model with p = 201 coefficients: the AME of x_k is β_k with SE sqrt(V[k, k])
    rng = MersenneTwister(7)
    n, q = 1_000, 200
    X = randn(rng, n, q)
    d = DataFrame(X, Symbol.("x", 1:q))
    d.y = X * (0.01 .* (1:q)) .+ randn(rng, n)
    f = term(:y) ~ term(1) + sum(term.(Symbol.("x", 1:q)))
    m = lm(f, d)
    V0 = Matrix(vcov(m))

    # A diagonal entry (k ≥ 2: a slope) whose in-place edit `hash` does not see
    function _hash_invisible_diagonal(V)
        h = hash(V)
        for k in 2:size(V, 1)
            old = V[k, k]
            V[k, k] = 4old
            invisible = hash(V) == h
            V[k, k] = old
            invisible && return k
        end
        return nothing
    end

    @testset "B2: in-place edit of a vcov matrix changes the next SE" begin
        V = copy(V0)
        k = _hash_invisible_diagonal(V)
        found = k !== nothing
        # The premise of the bug: some single-entry edits are invisible to `hash` (true for
        # the array hash of Julia 1.10-1.13). If a future `hash` sees every entry, the test
        # still checks the SE, it just no longer distinguishes the old behaviour.
        found || @info "hash(V) sees every diagonal edit on this Julia; B2 premise not exercised"
        k = something(k, 2)
        var = Symbol("x", k - 1)
        r1 = DataFrame(population_margins(m, d; vars = [var], vcov = V))
        @test r1.se[1] ≈ sqrt(V0[k, k]) rtol = 1e-10

        h = hash(V)
        V[k, k] *= 4                                                # in place, SE doubles
        found && @test hash(V) == h                                 # the edit the old key missed
        r2 = DataFrame(population_margins(m, d; vars = [var], vcov = V))
        @test r2.se[1] ≈ 2 * sqrt(V0[k, k]) rtol = 1e-10            # was r1.se[1]
        @test r2.estimate[1] ≈ r1.estimate[1] atol = 1e-12

        # profile margins read the same Σ
        g = cartesian_grid(; NamedTuple{(var,)}(([0.0],))...)
        V[k, k] = V0[k, k]
        s1 = DataFrame(profile_margins(m, d, g; vars = [var], vcov = V)).se[1]
        V[k, k] *= 9
        s2 = DataFrame(profile_margins(m, d, g; vars = [var], vcov = V)).se[1]
        @test s2 ≈ 3s1 rtol = 1e-10
    end

    @testset "changed vcov is revalidated" begin
        V = copy(V0)
        k = something(_hash_invisible_diagonal(V), 2)
        var = Symbol("x", k - 1)
        population_margins(m, d; vars = [var], vcov = V)              # valid: passes
        V[k, k] = NaN                                                 # now invalid, in place
        @test_throws ArgumentError population_margins(m, d; vars = [var], vcov = V)
        @test_throws ArgumentError profile_margins(m, d, means_grid(d); vars = [var], vcov = V)
        V[k, k] = V0[k, k]
        V[k, 1] += 1.0                                                # asymmetric
        @test_throws ArgumentError population_margins(m, d; vars = [var], vcov = V)
    end

    @testset "vcov is evaluated once per call" begin
        calls = Ref(0)
        counted = mdl -> (calls[] += 1; vcov(mdl))
        dm = DataFrame(x = randn(MersenneTwister(1), 200), z = randn(MersenneTwister(2), 200))
        dm.y = dm.x .+ dm.z .+ randn(MersenneTwister(3), 200)
        mm = lm(@formula(y ~ x * z), dm)
        population_margins(mm, dm; vcov = counted)
        @test calls[] == 1
        calls[] = 0
        population_margins(mm, dm; vars = [:x], scenarios = (z = [-1.0, 1.0],), vcov = counted)
        @test calls[] == 1
        calls[] = 0
        profile_margins(mm, dm, means_grid(dm); vcov = counted)
        @test calls[] == 1
        calls[] = 0
        second_differences(mm, dm, :x, :z; vcov = counted)
        @test calls[] == 1
        calls[] = 0
        second_differences(mm, dm, :x, :z; levels = [-1.0, 1.0], vcov = counted)
        @test calls[] == 1
        calls[] = 0
        second_differences(mm, dm, :x, :z; at = 0.0, vcov = counted)
        @test calls[] == 1
    end
end
