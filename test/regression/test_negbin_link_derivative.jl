# test/regression/test_negbin_link_derivative.jl
#
# Regression test: response-scale gradients for the negative-binomial canonical link.
#
# Margins took dμ/dη from `GLM.mueta`, which for `NegativeBinomialLink(θ)` returns
# θeᶯ/(1 − eᶯ) — that is μ itself — instead of θeᶯ/(1 − eᶯ)² (at θ = 2, η = −0.5: 3.083
# instead of 7.835). Response-scale prediction SEs (population and profile) and the
# measure gradients were therefore wrong for such models. GLM.jl also plans to remove
# `mueta` (GLM.jl #591). Margins now uses FormulaCompiler's `_dmu_deta` throughout.
#
# Oracle: estimates from `linkinv.(X β)` with `X` from `modelcols`; gradients by central
# differences in β (and in x for the AME); SE = √(∇'V∇).

using Test
using DataFrames, GLM, StatsModels, Statistics, LinearAlgebra, Tables
using Margins

@testset "negative-binomial link: response-scale SEs" begin
    i = 1:400
    d = DataFrame(x = sin.(i .* 0.37), z = cos.(i .* 1.3))
    θ = 2.0
    μtrue = exp.(0.8 .+ 0.5 .* d.x .- 0.3 .* d.z)
    d.y = floor.(μtrue .* (1 .+ 0.8 .* sin.(i .* 2.7)) .+ 0.5)
    L = NegativeBinomialLink(θ)
    m = glm(@formula(y ~ x + z), d, NegativeBinomial(θ), L)
    β = coef(m); V = vcov(m); rhs = formula(m).rhs
    X(dd) = modelcols(rhs, Tables.columntable(dd))
    μ(Xs, b) = GLM.linkinv.(Ref(L), Xs * b)
    ∇(f) = [(f(β .+ 1e-6 .* (1:length(β) .== j)) - f(β .- 1e-6 .* (1:length(β) .== j))) / 2e-6 for j in eachindex(β)]
    se(f) = (g = ∇(f); sqrt(dot(g, V, g)))

    # population prediction (AAP)
    X0 = X(d)
    aap(b) = mean(μ(X0, b))
    r = DataFrame(population_margins(m, d; type=:predictions))
    @test only(r.estimate) ≈ aap(β) rtol=1e-10
    @test only(r.se) ≈ se(aap) rtol=1e-6

    # profile prediction (APM)
    grid = DataFrame(x = [0.5, -0.5], z = [0.0, 0.2])
    rp = DataFrame(profile_margins(m, d, grid; type=:predictions))
    Xg = X(grid)
    for k in 1:2
        f = b -> μ(Xg, b)[k]
        @test rp.estimate[k] ≈ f(β) rtol=1e-10
        @test rp.se[k] ≈ se(f) rtol=1e-6
    end

    # AME of x and an elasticity (exercise _dmu_deta in the measure loop)
    h = 1e-5
    dp = copy(d); dp.x .+= h; dm = copy(d); dm.x .-= h
    Xp, Xm = X(dp), X(dm)
    ame(b) = mean((μ(Xp, b) .- μ(Xm, b)) ./ 2h)
    eyex(b) = mean((μ(Xp, b) .- μ(Xm, b)) ./ 2h .* d.x ./ μ(X0, b))
    for backend in (:ad, :fd)
        ra = DataFrame(population_margins(m, d; vars=[:x], backend))
        @test only(ra.estimate) ≈ ame(β) rtol=1e-6
        @test only(ra.se) ≈ se(ame) rtol=1e-4
        re = DataFrame(population_margins(m, d; vars=[:x], backend, measure=:elasticity))
        @test only(re.estimate) ≈ eyex(β) rtol=1e-6
        @test only(re.se) ≈ se(eyex) rtol=1e-4
    end
end
