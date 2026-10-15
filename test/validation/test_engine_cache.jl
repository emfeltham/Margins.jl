# test/validation/test_engine_cache.jl
#
# Regression tests for engine-cache data binding.
#
# Motivation: `ENGINE_CACHE` is keyed on the data's column *names*, but a `MarginsEngine`
# is bound to the column objects it was built from — its derivative and contrast evaluators
# store those arrays and dereference them at call time. Two datasets sharing a schema and a
# fitted model therefore collided on one cache entry, and the second caller silently got the
# first caller's rows (or a BoundsError, if the second dataset was longer). The fix
# revalidates a cache hit with `_engine_data_matches` before reusing it.
#
# Every expected value here is computed from `modelmatrix`/`coef`/`vcov` with plain array
# code. Do NOT switch to "compute it again with Margins on the other data" — that oracle is
# exactly what this bug poisons.

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, LinearAlgebra, Random
using Margins

# Closed-form AME of a continuous variable entering linearly, over `rows`
function _ame_truth(X, β, Σ, link, ix, rows, scale)
    f = bb -> begin
        η = X * bb
        d = scale === :response ? [GLM.mueta(link, η[r]) * bb[ix] for r in rows] :
                                  fill(bb[ix], length(rows))
        mean(d)
    end
    h = 1e-6
    grad = map(eachindex(β)) do j
        e = zeros(length(β)); e[j] = h
        (f(β .+ e) - f(β .- e)) / (2h)
    end
    return (f(β), sqrt(max(0.0, dot(grad, Σ, grad))))
end

# Closed-form average prediction over `rows`
function _pred_truth(X, β, Σ, link, rows, scale)
    η = X * β
    ŷ = scale === :response ? GLM.linkinv.(link, η) : η
    est = mean(@view ŷ[rows])
    grad = zeros(length(β))
    for r in rows
        factor = scale === :response ? GLM.mueta(link, η[r]) : 1.0
        grad .+= (factor / length(rows)) .* @view X[r, :]
    end
    return (est, sqrt(max(0.0, dot(grad, Σ, grad))))
end

function _cache_data(; n=240, seed=42)
    Random.seed!(seed)
    x = abs.(randn(n)) .+ 1.0
    g = categorical(rand(["a", "b", "c"], n))
    η = 0.3 .+ 0.8 .* x .+ 0.4 .* (g .== "b")
    y = rand(n) .< 1 ./ (1 .+ exp.(-η))
    return DataFrame(y=y, x=x, g=g, w=rand(n) .+ 0.25)
end

@testset "engine cache is bound to its data" begin

    df = _cache_data()
    m = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())
    X = modelmatrix(m); β = coef(m); Σ = GLM.vcov(m)
    link = GLM.Link(m.model)
    ix = findfirst(==("x"), coefnames(m))
    a_rows = findall(==("a"), string.(df.g))
    sub = df[a_rows, :]

    @testset "full then subset" begin
        # Before the fix the second call returned rows 1:60 of `df`.
        for scale in (:response, :link)
            full = DataFrame(population_margins(m, df; type=:effects, vars=[:x], scale=scale))
            (fe, fs) = _ame_truth(X, β, Σ, link, ix, 1:nrow(df), scale)
            @test full.estimate[1] ≈ fe atol=1e-9
            @test full.se[1] ≈ fs atol=1e-6

            part = DataFrame(population_margins(m, sub; type=:effects, vars=[:x], scale=scale))
            (se_, ss) = _ame_truth(X, β, Σ, link, ix, a_rows, scale)
            @test part.estimate[1] ≈ se_ atol=1e-9
            @test part.se[1] ≈ ss atol=1e-6
            @test part.n[1] == nrow(sub)
        end
    end

    @testset "subset then full" begin
        # Before the fix this order threw BoundsError (the cached evaluator held 60 rows).
        Margins.clear_engine_cache!()
        part = DataFrame(population_margins(m, sub; type=:effects, vars=[:x]))
        full = DataFrame(population_margins(m, df; type=:effects, vars=[:x]))
        (pe, _) = _ame_truth(X, β, Σ, link, ix, a_rows, :response)
        (fe, _) = _ame_truth(X, β, Σ, link, ix, 1:nrow(df), :response)
        @test part.estimate[1] ≈ pe atol=1e-9
        @test full.estimate[1] ≈ fe atol=1e-9
    end

    @testset "loop over subgroups in one session" begin
        # The workflow the bug breaks: one fitted model, many slices, no cache clearing.
        for lev in ("a", "b", "c")
            rows = findall(==(lev), string.(df.g))
            slice = df[rows, :]
            r = DataFrame(population_margins(m, slice; type=:effects, vars=[:x]))
            (want_est, want_se) = _ame_truth(X, β, Σ, link, ix, rows, :response)
            @test r.estimate[1] ≈ want_est atol=1e-9
            @test r.se[1] ≈ want_se atol=1e-6
            @test r.n[1] == length(rows)
        end
        # ...and going back to the full data still works
        full = DataFrame(population_margins(m, df; type=:effects, vars=[:x]))
        (fe, _) = _ame_truth(X, β, Σ, link, ix, 1:nrow(df), :response)
        @test full.estimate[1] ≈ fe atol=1e-9
    end

    @testset "predictions and profile margins" begin
        for scale in (:response, :link)
            p_sub = DataFrame(population_margins(m, sub; type=:predictions, scale=scale))
            (want_est, want_se) = _pred_truth(X, β, Σ, link, a_rows, scale)
            @test p_sub.estimate[1] ≈ want_est atol=1e-10
            @test p_sub.se[1] ≈ want_se atol=1e-10

            p_full = DataFrame(population_margins(m, df; type=:predictions, scale=scale))
            (fe, fs) = _pred_truth(X, β, Σ, link, 1:nrow(df), scale)
            @test p_full.estimate[1] ≈ fe atol=1e-10
            @test p_full.se[1] ≈ fs atol=1e-10
        end

        # profile_margins evaluates at the slice's own means, so the two must differ
        prof_sub = DataFrame(profile_margins(m, sub, means_grid(sub); type=:effects, vars=[:x]))
        prof_full = DataFrame(profile_margins(m, df, means_grid(df); type=:effects, vars=[:x]))
        @test !isapprox(prof_sub.estimate[1], prof_full.estimate[1]; atol=1e-8)
        # MEM at the slice's mean x: g'(η̄)·β_x with η̄ built from the slice's own means
        x̄ = mean(sub.x)
        props = [mean(string.(sub.g) .== lev) for lev in levels(sub.g)]
        η̄ = β[1] + β[ix] * x̄ + β[3] * props[2] + β[4] * props[3]
        @test prof_sub.estimate[1] ≈ GLM.mueta(link, η̄) * β[ix] atol=1e-8
    end

    @testset "categorical contrasts on a slice" begin
        # Exercises the ContrastEvaluator, which is also bound to the data
        b_rows = findall(!=("c"), string.(df.g))
        slice = df[b_rows, :]
        r = DataFrame(population_margins(m, slice; type=:effects, vars=[:g], scale=:link))
        # Link-scale baseline contrasts are exactly the coefficients, on any subset
        @test r.estimate[findfirst(==("b - a"), r.contrast)] ≈ β[3] atol=1e-10
        @test r.n[1] == length(b_rows)
    end

    @testset "cache still caches, and does not grow" begin
        Margins.clear_engine_cache!()
        data_nt = Margins._convert_numeric_to_float64(Margins.Tables.columntable(df))
        sub_nt = Margins._convert_numeric_to_float64(Margins.Tables.columntable(sub))
        args = (Margins.PopulationUsage, Margins.HasDerivatives, m)

        e1 = Margins.get_or_build_engine(args..., data_nt, [:x], GLM.vcov, :ad)
        e2 = Margins.get_or_build_engine(args..., data_nt, [:x], GLM.vcov, :ad)
        @test e1 === e2                                    # same data: cache hit
        @test Margins.get_cache_stats().entries == 1

        e3 = Margins.get_or_build_engine(args..., sub_nt, [:x], GLM.vcov, :ad)
        @test e3 !== e1                                    # different data: rebuilt
        @test Margins._engine_data_matches(e3, sub_nt)
        @test !Margins._engine_data_matches(e3, data_nt)
        @test Margins.get_cache_stats().entries == 1        # replaced, not accumulated

        # Repeated identical calls must not leak entries
        for _ in 1:5
            population_margins(m, df; type=:effects, vars=[:x])
        end
        @test Margins.get_cache_stats().entries <= 2        # effects engine (+ any predictions)
    end

    @testset "Int columns: rebuilt each call, still correct and bounded" begin
        # _convert_numeric_to_float64 makes a fresh array for non-Float64 numerics, so these
        # calls always miss revalidation. Results must stay correct and the cache bounded.
        Margins.clear_engine_cache!()
        Random.seed!(7); n = 200
        di = DataFrame(x=abs.(randn(n)) .+ 1.0, k=rand(1:10, n))
        di.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* di.x .+ 0.05 .* di.k)))
        mi = glm(@formula(y ~ x + k), di, Binomial(), LogitLink())
        Xi = modelmatrix(mi); βi = coef(mi); Σi = GLM.vcov(mi)
        linki = GLM.Link(mi.model)
        ixi = findfirst(==("x"), coefnames(mi))

        (want_est, want_se) = _ame_truth(Xi, βi, Σi, linki, ixi, 1:n, :response)
        for _ in 1:3
            r = DataFrame(population_margins(mi, di; type=:effects, vars=[:x]))
            @test r.estimate[1] ≈ want_est atol=1e-9
            @test r.se[1] ≈ want_se atol=1e-6
        end
        @test Margins.get_cache_stats().entries == 1

        half = di[1:100, :]
        (h_est, _) = _ame_truth(Xi, βi, Σi, linki, ixi, 1:100, :response)
        @test DataFrame(population_margins(mi, half; type=:effects, vars=[:x])).estimate[1] ≈ h_est atol=1e-9
    end

    @testset "same schema, different categorical level order" begin
        # Exercises FormulaCompiler's separate compiled-formula cache, which keys on a
        # semantic column-type signature. Both datasets must give their own correct answer.
        Margins.clear_engine_cache!()
        df2 = _cache_data(seed=99)
        df2.g = categorical(string.(df2.g); levels=["c", "b", "a"])
        m2 = glm(@formula(y ~ x + g), df2, Binomial(), LogitLink())
        X2 = modelmatrix(m2); β2 = coef(m2); Σ2 = GLM.vcov(m2)
        link2 = GLM.Link(m2.model)
        ix2 = findfirst(==("x"), coefnames(m2))

        r1 = DataFrame(population_margins(m, df; type=:effects, vars=[:x]))
        r2 = DataFrame(population_margins(m2, df2; type=:effects, vars=[:x]))
        (w1, _) = _ame_truth(X, β, Σ, link, ix, 1:nrow(df), :response)
        (w2, _) = _ame_truth(X2, β2, Σ2, link2, ix2, 1:nrow(df2), :response)
        @test r1.estimate[1] ≈ w1 atol=1e-9
        @test r2.estimate[1] ≈ w2 atol=1e-9

        # Link-scale contrasts must follow each frame's own baseline
        c2 = DataFrame(population_margins(m2, df2; type=:effects, vars=[:g], scale=:link))
        for row in eachrow(c2)
            to, from = split(row.contrast, " - ")
            idx = findfirst(s -> occursin("g:", s) && endswith(replace(s, " " => ""), to), coefnames(m2))
            @test from == "c"                       # first level is the baseline
            @test row.estimate ≈ β2[idx] atol=1e-10
        end
    end
end
