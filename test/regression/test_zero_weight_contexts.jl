# test/regression/test_zero_weight_contexts.jl
#
# Regression test: a context (here a `groups` cell) whose observations all have weight 0.
#
# Continuous effects in such a cell were reported as 0.0 ± 0.0 — a precisely estimated null
# rather than "no data" — while categorical effects and predictions raised an error. Every
# effect type must now raise the same `MarginsError`, on every backend, scale and measure
# (the AD fused loop, the generic/FD loops, and the elasticity loop).

using Test
using DataFrames, CategoricalArrays, GLM, Statistics, Random
using Margins

@testset "all-zero weights in a context raise a clear error" begin
    Random.seed!(2); n = 600
    d = DataFrame(x = randn(n), z = randn(n), g = categorical(rand(["a", "b"], n)), w = rand(n) .+ 0.1)
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.5 .* d.x .+ 0.8 .* d.z)))
    m = glm(@formula(y ~ x + z + g), d, Binomial(), LogitLink())
    d.h = categorical(rand(["p", "q"], n))
    w0 = copy(d.w); w0[string.(d.g) .== "a"] .= 0
    w0h = copy(d.w); w0h[string.(d.h) .== "p"] .= 0

    zero_weight_error(f) = try
        f(); nothing
    catch e
        e
    end

    for backend in (:ad, :fd), scale in (:response, :link), measure in (:effect, :elasticity)
        err = zero_weight_error(() -> population_margins(m, d; vars=[:x, :z], weights=w0,
                                                         groups=:g, backend, scale, measure))
        @test err isa Margins.MarginsError
        @test occursin("All weights are zero", sprint(showerror, err))
    end
    # categorical effects and predictions: same error type and message
    for f in (() -> population_margins(m, d; vars=[:g], weights=w0h, groups=:h),
              () -> population_margins(m, d; type=:predictions, weights=w0, groups=:g))
        err = zero_weight_error(f)
        @test err isa Margins.MarginsError
        @test occursin("All weights are zero", sprint(showerror, err))
    end

    # Weights that are zero only for SOME rows of a cell are fine, and match an explicit
    # weighted average over the positive-weight rows of that cell.
    wpart = copy(d.w); wpart[1:2:end] .= 0
    res = DataFrame(population_margins(m, d; vars=[:x], weights=wpart, groups=:g))
    β = coef(m); X = modelmatrix(m); ix = findfirst(==("x"), coefnames(m))
    μ′ = GLM.mueta.(LogitLink(), X * β)
    for lev in ("a", "b")
        rows = findall(==(lev), string.(d.g))
        want = sum(wpart[rows] .* μ′[rows] .* β[ix]) / sum(wpart[rows])
        @test only(res[string.(res.g) .== lev, :estimate]) ≈ want rtol=1e-10
    end
end
