# test/validation/test_elasticity_unit_definitions.jl
#
# Population elasticities are averages of UNIT-LEVEL quantities (Stata `margins, eyex()
# dyex() eydx()`, R `marginaleffects::avg_slopes(slope = ...)`):
#
#   :elasticity          (eyex)  mean_i[ gᵢ xᵢ / ŷᵢ ]
#   :semielasticity_dyex (dyex)  mean_i[ gᵢ xᵢ ]          dy / d(ln x)
#   :semielasticity_eydx (eydx)  mean_i[ gᵢ / ŷᵢ ]        d(ln y) / dx
#
# with gᵢ = ∂ŷᵢ/∂x and ŷᵢ the prediction on `scale`. Before v2.2.0 Margins computed a ratio
# of means instead ((x̄/ȳ)·AME, x̄·AME, AME/ȳ), which disagrees with its own docs, Stata
# and R whenever the unit quantities vary.
#
# Two independent references:
#  (a) a closed-form oracle built from `modelcols`/`coef`/`vcov` only — gᵢ by central
#      differences of linkinv.(X(x ± h) β), SEs by central differences of the estimate in β;
#  (b) R marginaleffects 1.0.0 values in
#      test/r_compare_marginaleffects/elasticity/elasticity_reference.csv (regenerate with
#      elasticity_reference.R there; the data below are built without an RNG so that both
#      languages construct identical data).

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, LinearAlgebra, CSV, Tables
using Margins

function _elasticity_data()
    i = 1:300
    d = DataFrame(x = 0.5 .+ mod.(i .* 37, 101) ./ 40,
                  z = cos.(i .* 0.7),
                  g = categorical(["a", "b", "c"][mod.(i, 3) .+ 1]),
                  w = 0.5 .+ mod.(i .* 13, 17) ./ 10)
    d.yb = Float64.(sin.(i .* 1.3) .+ 0.8 .* d.x .- 1.5 .+ 0.5 .* d.z .> 0)
    d.cnt = floor.(exp.(0.3 .+ 0.4 .* d.x .- 0.3 .* d.z) .+ 2 .* (sin.(i .* 2.1) .+ 1))
    d.ylin = 6 .+ 1.5 .* d.x .- 0.7 .* d.z .+ 2 .* sin.(i .* 0.9)
    return d
end

const _SLOPE_TO_MEASURE = Dict("eyex" => :elasticity, "dyex" => :semielasticity_dyex,
                               "eydx" => :semielasticity_eydx)

_wavg(v, rows, w) = isnothing(w) ? mean(v[rows]) : sum(w[rows] .* v[rows]) / sum(w[rows])

"""
Closed-form oracle: (estimate, se) of `measure` for variable :x over `rows`.
"""
function _unit_measure_oracle(m, d, rows, measure, scale, w)
    rhs = formula(m).rhs
    link = m isa StatsModels.TableRegressionModel && m.model isa GLM.LinearModel ?
        IdentityLink() : GLM.Link(m.model)
    X(dd) = modelcols(rhs, Tables.columntable(dd))
    h = 1e-5
    dp = copy(d); dp.x .+= h
    dm = copy(d); dm.x .-= h
    X0, Xp, Xm = X(d), X(dp), X(dm)
    pred(Xs, b) = scale === :response ? GLM.linkinv.(Ref(link), Xs * b) : Xs * b
    function est(b)
        ŷ = pred(X0, b)
        g = (pred(Xp, b) .- pred(Xm, b)) ./ (2h)
        q = measure === :elasticity ? g .* d.x ./ ŷ :
            measure === :semielasticity_dyex ? g .* d.x :
            measure === :semielasticity_eydx ? g ./ ŷ : g
        return _wavg(q, rows, w)
    end
    β = coef(m); V = vcov(m); hb = 1e-6
    grad = [(est(β .+ hb .* (1:length(β) .== j)) - est(β .- hb .* (1:length(β) .== j))) / (2hb)
            for j in eachindex(β)]
    return est(β), sqrt(dot(grad, V, grad))
end

@testset "Population elasticities are means of unit-level quantities" begin
    d = _elasticity_data()
    models = Dict(
        "logit"   => glm(@formula(yb ~ x * z + g), d, Binomial(), LogitLink()),
        "poisson" => glm(@formula(cnt ~ x + z + g), d, Poisson(), LogLink()),
        "lm"      => lm(@formula(ylin ~ x * z + g), d),
    )

    @testset "closed-form oracle: $mname $measure $scale $(wt ? "weighted" : "unweighted")" for
            (mname, m) in models, measure in values(_SLOPE_TO_MEASURE),
            scale in (:response, :link), wt in (false, true), backend in (:ad, :fd)
        w = wt ? d.w : nothing
        kw = wt ? (; weights=:w) : (;)
        res = DataFrame(population_margins(m, d; vars=[:x], measure, scale, backend, kw...))
        (oe, ose) = _unit_measure_oracle(m, d, 1:nrow(d), measure, scale, w)
        @test res.estimate[1] ≈ oe rtol=1e-6
        @test res.se[1] ≈ ose rtol=1e-4

        grp = DataFrame(population_margins(m, d; vars=[:x], measure, scale, backend,
                                           groups=:g, kw...))
        for lev in ("a", "b", "c")
            rows = findall(==(lev), string.(d.g))
            (ge, gse) = _unit_measure_oracle(m, d, rows, measure, scale, w)
            r = only(eachrow(grp[string.(grp.g) .== lev, :]))
            @test r.estimate ≈ ge rtol=1e-6
            @test r.se ≈ gse rtol=1e-4
        end
    end

    @testset "R marginaleffects avg_slopes()" begin
        ref = CSV.read(joinpath(@__DIR__, "..", "r_compare_marginaleffects", "elasticity",
                                "elasticity_reference.csv"), DataFrame)
        @test nrow(ref) == 120
        for r in eachrow(ref)
            kw = (; vars=[:x], measure=_SLOPE_TO_MEASURE[r.slope],
                    scale = r.type == "link" ? :link : :response)
            kw = r.weights == "w" ? merge(kw, (; weights=:w)) : kw
            kw = r.by == "g" ? merge(kw, (; groups=:g)) : kw
            res = DataFrame(population_margins(models[r.model], d; kw...))
            row = r.by == "g" ? only(eachrow(res[string.(res.g) .== r.group, :])) : only(eachrow(res))
            # Logit on the link scale divides by η, which is near 0 for some rows; R's
            # finite-difference slope is then the less accurate side (the closed-form
            # oracle above checks Margins exactly), so allow it a looser tolerance.
            ill = r.model == "logit" && r.type == "link" && r.slope != "dyex"
            @test row.estimate ≈ r.estimate rtol=(ill ? 1e-3 : 1e-6)
            @test row.se ≈ r.std_error rtol=(ill ? 1e-3 : 1e-3)
        end
    end

    @testset "ratio of means is NOT what is computed" begin
        # Regression guard for the pre-2.2.0 definition (x̄/ȳ)·AME
        m = models["logit"]
        ame = DataFrame(population_margins(m, d; vars=[:x])).estimate[1]
        el = DataFrame(population_margins(m, d; vars=[:x], measure=:elasticity)).estimate[1]
        old = mean(d.x) / mean(predict(m)) * ame
        @test !isapprox(el, old; rtol=1e-3)
        @test el ≈ 1.399523 atol=1e-5      # R: avg_slopes(m, variables="x", slope="eyex")
    end

    @testset "measure=:effect unchanged" begin
        m = models["poisson"]
        res = DataFrame(population_margins(m, d; vars=[:x]))
        (oe, ose) = _unit_measure_oracle(m, d, 1:nrow(d), :effect, :response, nothing)
        @test res.estimate[1] ≈ oe rtol=1e-6
        @test res.se[1] ≈ ose rtol=1e-4
    end

    @testset "zero prediction: clear error, not Inf/NaN" begin
        # No intercept, so ŷ = β x is exactly 0 at x = 0 and eyex/eydx are undefined there
        # (R returns NaN/Inf for that unit and propagates it into the average).
        d0 = DataFrame(x = [0.0, 1.0, 2.0, 3.0], y = [0.1, 1.2, 1.9, 3.1])
        m0 = lm(@formula(y ~ 0 + x), d0)
        for measure in (:elasticity, :semielasticity_eydx), backend in (:ad, :fd)
            err = try
                population_margins(m0, d0; vars=[:x], measure, backend); nothing
            catch e
                e
            end
            @test err isa Margins.MarginsError
            @test occursin("exactly 0", sprint(showerror, err))
        end
        # dyex does not divide by ŷ
        r = DataFrame(population_margins(m0, d0; vars=[:x], measure=:semielasticity_dyex))
        @test r.estimate[1] ≈ coef(m0)[1] * mean(d0.x)
    end

    @testset "profile measures unchanged (single row: mean of ratios = ratio of means)" begin
        m = models["poisson"]
        grid = DataFrame(x = [1.0, 2.5], z = [0.0, 0.0], g = categorical(["a", "a"], levels=["a", "b", "c"]))
        res = DataFrame(profile_margins(m, d, grid; vars=[:x], measure=:elasticity))
        β = coef(m); ix = findfirst(==("x"), coefnames(m))
        # Poisson/log link: elasticity at a point = β_x · x
        @test res.estimate ≈ β[ix] .* grid.x rtol=1e-8
    end
end
