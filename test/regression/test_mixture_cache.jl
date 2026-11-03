# test/regression/test_mixture_cache.jl
#
# Regression: two `profile_margins` calls on the SAME fitted model whose grids hold different
# categorical mixtures returned the FIRST call's mixture for the second, silently.
#
# FormulaCompiler caches compiled formulas by (model, column names, column categories), and a
# mixture column's category is just `:mixture`; but compilation embedded the compile-time
# mixture's levels and weights in the compiled operation's type. The second grid hit the
# cache and was evaluated at the first grid's category shares (`means_grid(data)` then
# `means_grid(data; weights = :w)`, or `complete_reference_grid(...; weights = :none)` then the
# default `:model`). Only clearing the cache or refitting the model gave the right answer.
# Within ONE call it also hit grids holding two different mixtures (evaluated in blocks).
# Fixed in FormulaCompiler 1.5.0: the operation reads each row's mixture from the data.
#
# Every expected value is a hand computation from the coefficients and the grid's own mixture.

using Test
using DataFrames, GLM, CategoricalArrays, Random, Statistics, LinearAlgebra
using Margins

@testset "mixture grids on one model are not answered from the first grid" begin
    rng = MersenneTwister(20261030)
    n = 400
    d = DataFrame(x = randn(rng, n), z = randn(rng, n) .+ 1,
                  g = categorical(rand(rng, ["a", "b", "c"], n)))
    d.w = rand(rng, n) .* 3 .+ 0.1
    d.y = Int.(rand(rng, n) .< 1 ./ (1 .+ exp.(-(0.5 .* d.x .- 0.3 .* d.z .+ (d.g .== "b")))))
    m = glm(@formula(y ~ x + z + g), d, Binomial(); wts = d.w)
    b = coef(m); V = vcov(m)

    # Category shares of a grid cell: a mixture, or a plain level
    shares(c) = c isa CategoricalMixture ?
        Dict(string(l) => w for (l, w) in zip(c.levels, c.weights)) : Dict(string(c) => 1.0)
    xrow(x, z, c) = (s = shares(c); [1.0, x, z, get(s, "b", 0.0), get(s, "c", 0.0)])
    μ(r) = 1 / (1 + exp(-dot(r, b)))
    # hand prediction, its delta-method SE, and the AME of x at the grid row
    pred(x, z, c) = μ(xrow(x, z, c))
    pred_se(x, z, c) = (r = xrow(x, z, c); p = μ(r); g = p * (1 - p) .* r; sqrt(g' * V * g))
    dydx(x, z, c) = (p = pred(x, z, c); b[2] * p * (1 - p))

    predictions(G) = DataFrame(profile_margins(m, d, G; type = :predictions))
    effects(G) = DataFrame(profile_margins(m, d, G; type = :effects, vars = [:x]))
    check_pred(G) = let r = predictions(G)
        [isapprox(r.estimate[i], pred(G.x[i], G.z[i], G.g[i]); rtol = 1e-10) &&
         isapprox(r.se[i], pred_se(G.x[i], G.z[i], G.g[i]); rtol = 1e-6) for i in 1:nrow(G)]
    end
    check_eff(G) = let r = effects(G)
        [isapprox(r.estimate[i], dydx(G.x[i], G.z[i], G.g[i]); rtol = 1e-8) for i in 1:nrow(G)]
    end

    @testset "means_grid unweighted, then weighted shares (the reported pattern)" begin
        G1 = means_grid(d)
        G2 = means_grid(d; weights = :w)
        @test G1.g[1].weights != G2.g[1].weights           # the grids differ only in shares
        @test all(check_pred(G1))
        @test all(check_pred(G2))                          # was G1's shares
        @test all(check_eff(G1))
        @test all(check_eff(G2))
        @test all(check_pred(G1))                          # and back again
    end

    @testset "complete_reference_grid weights = :none, then :model" begin
        Gn = complete_reference_grid(DataFrame(x = [0.5]), m, d; weights = :none)
        Gm = complete_reference_grid(DataFrame(x = [0.5]), m, d)
        @test Gn.g[1].weights != Gm.g[1].weights
        @test all(check_pred(Gn))
        @test all(check_pred(Gm))
        @test all(check_eff(Gm))
    end

    @testset "mixtures over different levels, and a plain level" begin
        mA = mix("a" => 0.5, "b" => 0.25, "c" => 0.25)
        mL = mix("c" => 0.6, "b" => 0.4)
        for g in (mA, mL, mix("b" => 1.0), "c", mA)
            G = DataFrame(x = [0.3], z = [1.2], g = [g])
            @test all(check_pred(G))
            @test all(check_eff(G))
        end
    end

    @testset "one grid holding two different mixtures (one call)" begin
        G = DataFrame(x = [0.0, 0.0, 1.0], z = [1.0, 1.0, 0.5],
                      g = Any[mix("a" => 0.5, "b" => 0.25, "c" => 0.25),
                              mix("a" => 0.1, "b" => 0.1, "c" => 0.8), "b"])
        @test all(check_pred(G))
    end
end
