# test/regression/test_profile_measure_quotient_rule.jl
#
# Regression test: profile elasticity SEs when the transformation factor is exactly 1.
#
# `_process_profile_continuous_variable!` applied the quotient-rule term −(ε/ŷ)·∂ŷ/∂β for
# measures that divide by ŷ only inside `if transform_factor != 1.0`. When the factor was
# exactly 1.0 — `:elasticity` at a point with x == ŷ, `:semielasticity_eydx` at ŷ == 1 — the
# term was dropped: right estimate, wrong SE. Profile measures also divided by ŷ = 0
# silently. At a single grid row the unit definition (A4) and the ratio coincide:
# eyex = g·x/ŷ, eydx = g/ŷ, dyex = g·x, with x and ŷ at the grid row.
#
# Oracle: closed form for y ~ x (identity link, g = β₁, ŷ = β₀ + β₁x); SEs by central
# differences in β.

using Test
using DataFrames, GLM, LinearAlgebra
using Margins

@testset "profile measures: quotient rule when the factor is exactly 1" begin
    i = 1:200
    d = DataFrame(x = sin.(i .* 0.7) .* 3)
    d.y = 2.0 .+ 0.5 .* d.x .+ 0.8 .* cos.(i .* 1.9)
    m = lm(@formula(y ~ x), d)
    β = coef(m); V = vcov(m)
    ŷ(b, x) = dot([1.0, x], b)   # the same order of operations as the model row · β

    # Find a grid x (searching a few ulps around the analytic root) at which the computed
    # prediction satisfies `cond` exactly.
    function exact_point(x0, cond)
        x = x0
        for _ in 1:2000
            cond(x) && return x
            x = nextfloat(x)
        end
        x = x0
        for _ in 1:2000
            cond(x) && return x
            x = prevfloat(x)
        end
        return nothing
    end
    se(f) = (h = 1e-6; g = [(f(β .+ h .* (1:2 .== j)) - f(β .- h .* (1:2 .== j))) / 2h for j in 1:2]; sqrt(dot(g, V, g)))

    # eyex with x / ŷ == 1.0
    xa = exact_point(β[1] / (1 - β[2]), x -> x / ŷ(β, x) == 1.0)
    @test !isnothing(xa)
    r = DataFrame(profile_margins(m, d, DataFrame(x = [xa]); vars=[:x], measure=:elasticity))
    f = b -> b[2] * xa / ŷ(b, xa)
    @test only(r.estimate) ≈ f(β) rtol=1e-12
    @test only(r.se) ≈ se(f) rtol=1e-6
    @test !isapprox(only(r.se), sqrt(V[2, 2]); rtol=1e-3)   # the dropped-term value

    # eydx with ŷ == 1.0
    xb = exact_point((1 - β[1]) / β[2], x -> ŷ(β, x) == 1.0)
    @test !isnothing(xb)
    r = DataFrame(profile_margins(m, d, DataFrame(x = [xb]); vars=[:x], measure=:semielasticity_eydx))
    f = b -> b[2] / ŷ(b, xb)
    @test only(r.estimate) ≈ f(β) rtol=1e-12
    @test only(r.se) ≈ se(f) rtol=1e-6

    # dyex is consistent with the unit definition g·x
    r = DataFrame(profile_margins(m, d, DataFrame(x = [1.5]); vars=[:x], measure=:semielasticity_dyex))
    @test only(r.estimate) ≈ β[2] * 1.5 rtol=1e-12
    @test only(r.se) ≈ 1.5 * sqrt(V[2, 2]) rtol=1e-10

    # ŷ == 0 at the grid row: a clear error, not Inf/NaN
    m0 = lm(@formula(y ~ 0 + x), d)
    for measure in (:elasticity, :semielasticity_eydx)
        err = try
            profile_margins(m0, d, DataFrame(x = [0.0]); vars=[:x], measure); nothing
        catch e
            e
        end
        @test err isa Margins.MarginsError
    end
end
