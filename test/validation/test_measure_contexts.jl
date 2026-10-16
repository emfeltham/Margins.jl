# test/validation/test_measure_contexts.jl
#
# Regression tests for `measure` under `scenarios` / `groups`, and for the weighted
# context path.
#
# Motivation: `measure` used to be accepted and then silently dropped whenever a context
# was supplied — `_population_margins_with_contexts` threaded it into the metadata only —
# so `population_margins(...; measure=:elasticity, groups=:g)` returned raw AMEs labeled
# `AME (elasticity)`. No test in the suite paired `measure` with a context, so nothing
# caught it. The weighted context path additionally threw `BoundsError`, because the
# per-context weight *subset* was being indexed by absolute row index.
#
# Oracle strategy: every expected value here is computed from `modelmatrix(m)`, `coef(m)`
# and `GLM.vcov(m)` with plain array code — nothing from Margins. The gradient is obtained
# by central-differencing the estimate with respect to β, which validates the delta-method
# SE for every measure/scale combination without hand-deriving d²μ/dη².
#
# Do NOT switch these to a "compute it on the subset DataFrame" oracle: the engine cache
# is keyed on column *names* only, so a second dataset with the same schema silently
# reuses the first engine (see ISSUES.md).

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, LinearAlgebra, Random
using Margins

# Weighted mean over a subset of absolute row indices
_wmean(v, rows, w) = isnothing(w) ? mean(@view v[rows]) :
    sum(w[r] * v[r] for r in rows) / sum(w[r] for r in rows)

"""
Closed-form (estimate, se) for the marginal effect of a continuous `var` over `rows`,
under `measure` and `scale`. `X` is the design matrix already reflecting any scenario
override. Measures are (weighted) means of unit-level quantities, as in Stata `margins,
eyex()/dyex()/eydx()` and R `avg_slopes(slope = ...)` (v2.2.0; earlier versions used a
ratio of means — see test/validation/test_elasticity_unit_definitions.jl). Gradients come
from a central difference in β, so this covers the quotient-rule terms without deriving
them by hand.
"""
function _measure_oracle(X, β, Σ, link, xcol, ix, rows, measure, scale, w)
    function estimate(bb)
        η = X * bb
        # Unit marginal effect: ∂η/∂x = β_x on the link scale, g'(η)·β_x on the response scale
        d = if scale === :response
            [GLM.mueta(link, η[r]) * bb[ix] for r in eachindex(η)]
        else
            fill(bb[ix], length(η))
        end
        measure === :effect && return _wmean(d, rows, w)
        measure === :semielasticity_dyex && return _wmean(d .* xcol, rows, w)

        ŷ = scale === :response ? GLM.linkinv.(link, η) : η
        measure === :elasticity && return _wmean(d .* xcol ./ ŷ, rows, w)
        measure === :semielasticity_eydx && return _wmean(d ./ ŷ, rows, w)
        error("unhandled measure $measure")
    end

    h = 1e-6
    grad = map(eachindex(β)) do j
        e = zeros(length(β)); e[j] = h
        (estimate(β .+ e) - estimate(β .- e)) / (2h)
    end
    return (estimate(β), sqrt(max(0.0, dot(grad, Σ, grad))))
end

function _context_data(; n=300, seed=515)
    Random.seed!(seed)
    x = abs.(randn(n)) .+ 1.0          # strictly positive, so elasticities are well behaved
    z = randn(n)
    g = categorical(rand(["a", "b", "c"], n))
    h = categorical(rand(["p", "q"], n))
    η = 0.3 .+ 0.8 .* x .+ 0.5 .* z .+ 0.4 .* (g .== "b")
    y = rand(n) .< 1 ./ (1 .+ exp.(-η))
    return DataFrame(y=y, x=x, z=z, g=g, h=h, w=rand(n) .+ 0.25)
end

const _MEASURES = (:elasticity, :semielasticity_dyex, :semielasticity_eydx)

@testset "measure under scenarios and groups" begin

    df = _context_data()
    models = (
        ("logit",  glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())),
        ("probit", glm(@formula(y ~ x + z + g), df, Binomial(), ProbitLink())),
    )

    @testset "groups: $mname, $measure, scale=$scale" for (mname, m) in models,
                                                          measure in _MEASURES,
                                                          scale in (:response, :link)
        X = modelmatrix(m); β = coef(m); Σ = GLM.vcov(m)
        link = GLM.Link(m.model)
        ix = findfirst(==("x"), coefnames(m))

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                           measure=measure, scale=scale, groups=:g))
        @test nrow(res) == 3
        for lev in ("a", "b", "c")
            rows = findall(==(lev), string.(df.g))
            (want_est, want_se) = _measure_oracle(X, β, Σ, link, df.x, ix, rows,
                                                  measure, scale, nothing)
            r = only(eachrow(res[string.(res.g) .== lev, :]))
            @test r.estimate ≈ want_est atol=1e-9
            @test r.se ≈ want_se atol=1e-6 rtol=1e-6   # central difference on β (link-scale unit
            # elasticities divide by η, which can be near 0, so SEs can be large)
            @test r.n == length(rows)
        end
    end

    @testset "weighted groups: $measure" for measure in _MEASURES
        # Regression test for the weight-convention bug: this combination used to throw
        # BoundsError before reaching any arithmetic.
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        X = modelmatrix(m); β = coef(m); Σ = GLM.vcov(m)
        link = GLM.Link(m.model)
        ix = findfirst(==("x"), coefnames(m))

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                           measure=measure, groups=:g, weights=:w))
        for lev in ("a", "b", "c")
            rows = findall(==(lev), string.(df.g))
            (want_est, want_se) = _measure_oracle(X, β, Σ, link, df.x, ix, rows,
                                                  measure, :response, df.w)
            r = only(eachrow(res[string.(res.g) .== lev, :]))
            @test r.estimate ≈ want_est atol=1e-9
            @test r.se ≈ want_se atol=1e-6
        end

        # Weights must actually matter here
        unw = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                           measure=measure, groups=:g))
        @test !isapprox(sort(res.estimate), sort(unw.estimate); atol=1e-8)
    end

    @testset "scenarios: $measure, scale=$scale" for measure in _MEASURES,
                                                     scale in (:response, :link)
        # Override a continuous term that enters linearly, so the oracle's design matrix
        # is the original one with that column replaced.
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        β = coef(m); Σ = GLM.vcov(m)
        link = GLM.Link(m.model)
        ix = findfirst(==("x"), coefnames(m))
        iz = findfirst(==("z"), coefnames(m))

        for zval in (-1.0, 0.0, 1.5)
            X = copy(modelmatrix(m)); X[:, iz] .= zval
            rows = 1:nrow(df)
            (want_est, want_se) = _measure_oracle(X, β, Σ, link, df.x, ix, rows,
                                                  measure, scale, nothing)
            res = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                               measure=measure, scale=scale,
                                               scenarios=(z=zval,)))
            @test res.estimate[1] ≈ want_est atol=1e-9
            @test res.se[1] ≈ want_se atol=1e-6
        end
    end

    @testset "scenarios and groups combined" begin
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        β = coef(m); Σ = GLM.vcov(m)
        link = GLM.Link(m.model)
        ix = findfirst(==("x"), coefnames(m))
        iz = findfirst(==("z"), coefnames(m))
        X = copy(modelmatrix(m)); X[:, iz] .= 0.0

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                           measure=:elasticity, scenarios=(z=0.0,), groups=:h))
        @test nrow(res) == 2
        for lev in ("p", "q")
            rows = findall(==(lev), string.(df.h))
            (want_est, want_se) = _measure_oracle(X, β, Σ, link, df.x, ix, rows,
                                                  :elasticity, :response, nothing)
            r = only(eachrow(res[string.(res.h) .== lev, :]))
            @test r.estimate ≈ want_est atol=1e-9
            @test r.se ≈ want_se atol=1e-6
        end
    end

    @testset "context normalizers are context-specific" begin
        # The exact signature of the bug: a transformed measure must differ from the raw
        # AME, and each group must average over its own rows (so the three groups differ
        # from the single whole-sample elasticity).
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())

        raw = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=:g))
        for measure in _MEASURES
            transformed = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                                       measure=measure, groups=:g))
            for lev in ("a", "b", "c")
                a = only(eachrow(raw[string.(raw.g) .== lev, :]))
                b = only(eachrow(transformed[string.(transformed.g) .== lev, :]))
                @test !isapprox(a.estimate, b.estimate; atol=1e-8)
                @test !isapprox(a.se, b.se; atol=1e-8)
            end
        end

        # measure=:effect must be untouched by the change
        for scale in (:response, :link)
            r = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                             scale=scale, groups=:g))
            X = modelmatrix(m); β = coef(m); Σ = GLM.vcov(m)
            link = GLM.Link(m.model); ix = findfirst(==("x"), coefnames(m))
            for lev in ("a", "b", "c")
                rows = findall(==(lev), string.(df.g))
                (want_est, want_se) = _measure_oracle(X, β, Σ, link, df.x, ix, rows,
                                                      :effect, scale, nothing)
                row = only(eachrow(r[string.(r.g) .== lev, :]))
                @test row.estimate ≈ want_est atol=1e-9
                @test row.se ≈ want_se atol=1e-6
            end
        end
    end

    @testset "weighted predictions with groups" begin
        # Second half of the weight-convention fix: this path also threw BoundsError.
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        X = modelmatrix(m); β = coef(m); Σ = GLM.vcov(m)
        link = GLM.Link(m.model)
        η = X * β

        for scale in (:response, :link)
            res = DataFrame(population_margins(m, df; type=:predictions, scale=scale,
                                               groups=:g, weights=:w))
            ŷ = scale === :response ? GLM.linkinv.(link, η) : η
            for lev in ("a", "b", "c")
                rows = findall(==(lev), string.(df.g))
                want_est = _wmean(ŷ, rows, df.w)
                # ∂ȳ/∂β = weighted mean of g'(η)X (response) or X (link)
                W = sum(df.w[r] for r in rows)
                grad = zeros(length(β))
                for r in rows
                    factor = scale === :response ? GLM.mueta(link, η[r]) : 1.0
                    grad .+= (df.w[r] * factor / W) .* @view X[r, :]
                end
                r = only(eachrow(res[string.(res.g) .== lev, :]))
                @test r.estimate ≈ want_est atol=1e-10
                @test r.se ≈ sqrt(dot(grad, Σ, grad)) atol=1e-10
            end
        end
    end

    @testset "elasticity measures require continuous variables" begin
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())

        # Explicitly requested discrete variable: error rather than a mislabeled contrast
        for measure in _MEASURES
            @test_throws ArgumentError population_margins(m, df; type=:effects,
                                                          vars=[:g], measure=measure)
            @test_throws ArgumentError population_margins(m, df; type=:effects,
                                                          vars=[:x, :g], measure=measure)
            @test_throws ArgumentError population_margins(m, df; type=:effects, vars=[:g],
                                                          measure=measure, groups=:h)
        end
        # measure=:effect is of course still fine for discrete variables
        @test nrow(DataFrame(population_margins(m, df; type=:effects, vars=[:g]))) == 2

        # Auto-detected vars: drop the discrete ones, keep working
        auto = DataFrame(population_margins(m, df; type=:effects, measure=:elasticity))
        @test Set(auto.variable) == Set(["x", "z"])
        @test all(isfinite, auto.se)
    end
end
