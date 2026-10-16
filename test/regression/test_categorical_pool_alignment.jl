# test/regression/test_categorical_pool_alignment.jl
#
# Regression test: data whose categorical pool differs from the fitted model's levels.
#
# A subset re-`categorical`-ized after dropping the reference level has a pool without that
# level. `population_margins(m, subset; vars=[:g])` failed with "Baseline level 'a' not
# found", which was caught, logged as a warning, and followed by an `UndefRefError` from the
# partially filled results table. Contrasts are now taken against the model's reference
# level (from the fitted schema), with the missing levels appended to the column's pool.
# A reordered pool must give the same contrasts as the original.
#
# Oracle: mean over the data rows of linkinv(X(g = to)β) − linkinv(X(g = from)β), with the
# design rows built by `modelcols` on the model's schema; SEs by central differences in β.

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, LinearAlgebra, Random, Tables
using Margins

@testset "categorical pools that differ from the model's levels" begin
    Random.seed!(1); n = 3000
    d = DataFrame(x = randn(n), g = categorical(rand(["a", "b", "c"], n)))
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.5 .* d.x .+ [0.0, 1.0, -1.0][levelcode.(d.g)])))
    m = glm(@formula(y ~ x + g), d, Binomial(), LogitLink())
    β = coef(m); V = vcov(m); L = LogitLink()

    function oracle(dd, to, from)
        X(lv) = modelcols(formula(m).rhs,
                          (x = dd.x, g = categorical(fill(lv, nrow(dd)), levels=["a", "b", "c"]), y = dd.y))
        X1, X0 = X(to), X(from)
        est(b) = mean(GLM.linkinv.(Ref(L), X1 * b) .- GLM.linkinv.(Ref(L), X0 * b))
        h = 1e-6
        grad = [(est(β .+ h .* (1:length(β) .== j)) - est(β .- h .* (1:length(β) .== j))) / 2h for j in eachindex(β)]
        return est(β), sqrt(dot(grad, V, grad))
    end

    check(res, dd) = for r in eachrow(res)
        to, from = String.(split(r.contrast, " - "))
        (oe, ose) = oracle(dd, to, from)
        @test r.estimate ≈ oe rtol=1e-8
        @test r.se ≈ ose rtol=1e-5
    end

    # reference level absent from the data's pool
    d3 = d[string.(d.g) .!= "a", :]; d3.g = categorical(string.(d3.g))
    @test levels(d3.g) == ["b", "c"]
    for backend in (:ad, :fd)
        res = DataFrame(population_margins(m, d3; vars=[:g], backend))
        @test sort(res.contrast) == ["b - a", "c - a"]
        check(res, d3)
        # predictions on that subset match GLM.predict
        @test only(DataFrame(population_margins(m, d3; type=:predictions)).estimate) ≈ mean(predict(m, d3)) rtol=1e-10
    end

    # reordered pool: same contrasts as the original data
    d2 = copy(d); d2.g = categorical(string.(d.g), levels=["c", "b", "a"])
    r1 = sort(DataFrame(population_margins(m, d; vars=[:g])), :contrast)
    r2 = sort(DataFrame(population_margins(m, d2; vars=[:g])), :contrast)
    @test r1.contrast == r2.contrast == ["b - a", "c - a"]
    @test r1.estimate ≈ r2.estimate rtol=1e-12
    @test r1.se ≈ r2.se rtol=1e-10
    check(r2, d2)
end
