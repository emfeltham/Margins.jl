# test/regression/test_profile_multi_categorical.jl
#
# Regression test: `profile_margins` with two or more categoricals in `vars`.
#
# The hybrid evaluator used for categorical contrasts at grid rows replaced EVERY requested
# categorical column with a placeholder (first level / `false`). Contrasting one categorical
# therefore evaluated the others at the placeholder rather than at the grid row's value
# (a pinned level or a mixture), so estimates and SEs depended on which other categoricals
# happened to be requested (e.g. the effect of b at g = "b" was reported as 0.180 — its
# value at g = "a" — instead of 0.124). Each contrast must hold all other variables at the
# grid row's values.
#
# Oracle: the design row of a profile is the frequency-weighted average of the design rows
# of its discrete components (exact, because the model matrix is multilinear in the
# indicator columns), built with `modelcols` on the model's schema; SEs by central
# differences of the contrast in β.

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, LinearAlgebra, Random, Tables
using Margins

function _mc_data(; n=1500, seed=4)
    Random.seed!(seed)
    d = DataFrame(x = randn(n), b = rand(Bool, n), g = categorical(rand(["a", "b", "c"], n)))
    η = -0.5 .+ 0.6 .* d.x .+ 0.7 .* d.b .+ [0, 1.5, -1.0][levelcode.(d.g)] .+ 0.4 .* d.x .* d.b
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-η))
    return d
end

# Profile: x value, P(b = true), Dict(level => weight) for g
function _mc_xrow(m, levs, x, pb, gmix)
    acc = zeros(length(coef(m)))
    for (lv, pg) in gmix, (bv, wb) in ((true, pb), (false, 1 - pb))
        (pg == 0 || wb == 0) && continue
        row = (x = [x], b = [bv], g = categorical([lv], levels=levs), y = [false])
        acc .+= pg * wb .* vec(modelcols(formula(m).rhs, row))
    end
    return acc
end

function _mc_oracle(m, levs, f; scale)
    link = GLM.Link(m.model)
    μ(xr, β) = (η = dot(xr, β); scale === :link ? η : GLM.linkinv(link, η))
    est(β) = f((xr) -> μ(xr, β))
    β = coef(m); V = vcov(m); h = 1e-6
    grad = [(est(β .+ h .* (1:length(β) .== j)) - est(β .- h .* (1:length(β) .== j))) / (2h)
            for j in eachindex(β)]
    return est(β), sqrt(dot(grad, V, grad))
end

@testset "profile_margins: several categoricals in vars" begin
    d = _mc_data()
    levs = string.(levels(d.g))
    m = glm(@formula(y ~ x * b + g), d, Binomial(), LogitLink())

    @testset "pinned grid, scale=$scale, backend=$backend" for scale in (:response, :link),
                                                              backend in (:ad, :fd)
        grid = cartesian_grid(x = [0.0, 1.5], g = ["b", "c"], b = [true])
        x̄ = mean(d.x)
        for vars in ([:b, :g], [:x, :b, :g], [:g, :b])
            res = DataFrame(profile_margins(m, d, grid; vars, scale, backend))
            for r in eachrow(res)
                gl = string(r.g)
                pin = Dict(gl => 1.0)
                if r.variable == "b"
                    f = μ -> μ(_mc_xrow(m, levs, r.x, 1.0, pin)) - μ(_mc_xrow(m, levs, r.x, 0.0, pin))
                elseif r.variable == "g"
                    to, from = String.(split(r.contrast, " - "))
                    f = μ -> μ(_mc_xrow(m, levs, r.x, 1.0, Dict(to => 1.0))) -
                             μ(_mc_xrow(m, levs, r.x, 1.0, Dict(from => 1.0)))
                else
                    continue
                end
                (oe, ose) = _mc_oracle(m, levs, f; scale)
                @test r.estimate ≈ oe rtol=1e-8 atol=1e-12
                @test r.se ≈ ose rtol=1e-5
            end
        end
    end

    @testset "same answer whatever else is requested" begin
        grid = cartesian_grid(g = ["b"])
        alone = DataFrame(profile_margins(m, d, grid; vars=[:b]))
        both = DataFrame(profile_margins(m, d, grid; vars=[:b, :g]))
        rb = only(eachrow(both[both.variable .== "b", :]))
        @test rb.estimate ≈ only(alone.estimate) rtol=1e-12
        @test rb.se ≈ only(alone.se) rtol=1e-10
    end

    @testset "means grid (mixtures), backend=$backend" for backend in (:ad, :fd)
        res = DataFrame(profile_margins(m, d, means_grid(d); vars=[:x, :b, :g], backend))
        x̄ = mean(d.x); p̄ = mean(d.b)
        gmix = Dict(l => mean(string.(d.g) .== l) for l in levs)
        for r in eachrow(res)
            if r.variable == "b"
                f = μ -> μ(_mc_xrow(m, levs, x̄, 1.0, gmix)) - μ(_mc_xrow(m, levs, x̄, 0.0, gmix))
            elseif r.variable == "g"
                to, from = String.(split(r.contrast, " - "))
                f = μ -> μ(_mc_xrow(m, levs, x̄, p̄, Dict(to => 1.0))) -
                         μ(_mc_xrow(m, levs, x̄, p̄, Dict(from => 1.0)))
            else
                continue
            end
            (oe, ose) = _mc_oracle(m, levs, f; scale=:response)
            @test r.estimate ≈ oe rtol=1e-8 atol=1e-12
            @test r.se ≈ ose rtol=1e-5
        end
    end
end
