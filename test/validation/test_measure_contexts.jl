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
eyex()/dyex()/eydx()` and R `avg_slopes(slope = ...)` (v2.3.0; earlier versions used a
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

    # `measure` under groups alone, weighted groups and scenarios alone is checked against
    # closed forms for 7 links × 2 scales × 3 weightings in test_feature_matrix_oracle.jl
    # (paths "groups=h", "groups=(z,3)", "scenarios z=0.5"; estimates rtol 1e-6) and pinned
    # to 1e-12 by test_golden_equivalence.jl. This testset keeps an analytic (no finite
    # difference in x) check at atol 1e-9 of every measure, scale and weighting with a
    # scenario and groups at once.
    @testset "scenarios and groups combined: $measure, scale=$scale, weights=$wname" for
            measure in (_MEASURES..., :effect), scale in (:response, :link),
            (wname, wcol) in (("none", nothing), ("w", :w))
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        β = coef(m); Σ = GLM.vcov(m)
        link = GLM.Link(m.model)
        ix = findfirst(==("x"), coefnames(m))
        iz = findfirst(==("z"), coefnames(m))
        X = copy(modelmatrix(m)); X[:, iz] .= 0.0
        w = isnothing(wcol) ? nothing : df[!, wcol]
        wkw = isnothing(wcol) ? (;) : (; weights=wcol)

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], measure, scale,
                                           scenarios=(z=0.0,), groups=:h, wkw...))
        @test nrow(res) == 2
        for lev in ("p", "q")
            rows = findall(==(lev), string.(df.h))
            (want_est, want_se) = _measure_oracle(X, β, Σ, link, df.x, ix, rows,
                                                  measure, scale, w)
            r = only(eachrow(res[string.(res.over_h) .== lev, :]))
            @test r.estimate ≈ want_est atol=1e-9
            @test r.se ≈ want_se atol=1e-6 rtol=1e-6
            @test r.n == length(rows)
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
                r = only(eachrow(res[string.(res.over_g) .== lev, :]))
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
