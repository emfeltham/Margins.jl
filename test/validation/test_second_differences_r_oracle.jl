# test/validation/test_second_differences_r_oracle.jl
#
# Second differences of nonlinear models against R marginaleffects 1.0.0. A second difference
# is the change of an average marginal effect when the modifier is set to other values for
# everyone (counterfactual grid), tested in R as
#   avg_slopes(m, variables = "x", newdata = datagrid(t = c(0, 1), grid_type = "counterfactual"),
#              by = "t", hypothesis = "b2 - b1 = 0")
# whose delta-method SE uses the joint Jacobian of both averages.
#
# Reference values: test/r_compare_marginaleffects/second_differences/second_differences_reference.csv
# (regenerate with second_differences_reference.R there; the data are built without an RNG so
# that both languages construct identical data). Covers logit and Poisson, response and link
# scale, with and without weights, binary / categorical / continuous modifiers, and all four
# measures (effect, elasticity, dyex, eydx).

using Test
using DataFrames, CategoricalArrays, GLM, CSV, StatsModels, Tables, Statistics
using Margins

function _sd_oracle_data()
    i = 1:300
    d = DataFrame(x = 0.5 .+ mod.(i .* 37, 101) ./ 40,
                  z = cos.(i .* 0.7),
                  g = categorical(["a", "b", "c"][mod.(i, 3) .+ 1]),
                  w = 0.5 .+ mod.(i .* 13, 17) ./ 10,
                  t = Float64.(mod.(i, 2)),
                  sw = 0.5 .+ mod.(i .* 7, 11) ./ 5)
    d.yb = Float64.(sin.(i .* 1.3) .+ 0.8 .* d.x .- 1.5 .+ 0.5 .* d.z .+ 0.4 .* d.x .* d.t .> 0)
    d.cnt = floor.(exp.(0.3 .+ 0.4 .* d.x .- 0.3 .* d.z .+ 0.2 .* d.x .* d.t) .+ 2 .* (sin.(i .* 2.1) .+ 1))
    return d
end

const _SD_MEASURE = Dict("dydx" => :effect, "eyex" => :elasticity,
                         "dyex" => :semielasticity_dyex, "eydx" => :semielasticity_eydx)

@testset "second differences vs R marginaleffects" begin
    d = _sd_oracle_data()
    models = Dict(
        "logit" => glm(@formula(yb ~ x * t + x * w + z + g), d, Binomial(), LogitLink()),
        "poisson" => glm(@formula(cnt ~ x * t + x * w + z + g), d, Poisson(), LogLink()),
    )
    ref = CSV.read(joinpath(@__DIR__, "..", "r_compare_marginaleffects", "second_differences",
                            "second_differences_reference.csv"), DataFrame)
    @test nrow(ref) == 72
    # some CSV versions return string-like columns; normalize
    _num(x) = x isa Real ? Float64(x) : (String(x) == "NA" ? NaN : parse(Float64, String(x)))
    for c in (:model, :kind, :measure, :weights, :type)
        ref[!, c] = String.(ref[!, c])
    end
    for c in (:estimate, :std_error)
        ref[!, c] = _num.(ref[!, c])
    end

    # Julia side of each reference row: (estimate, se)
    function julia_value(m, kind, measure, wt, tp)
        kw = (scale = Symbol(tp), measure = _SD_MEASURE[measure], weights = wt == "sw" ? :sw : nothing, warn = false)
        if kind == "binary_x"
            r = second_differences(m, d, :x, :t; kw...)
        elseif startswith(kind, "binary_g_")
            r = second_differences(m, d, :g, :t; kw...)
            r = r[r.contrast .== "$(last(kind)) - a", :]
        elseif kind == "local_w"
            r = second_differences(m, d, :x, :w; at = 1.2, delta = 0.01, kw...)
        elseif kind == "slope_w"
            r = second_differences(m, d, :x, :w; levels = [0.8, 1.8], kw...)
        end
        @assert nrow(r) == 1
        return r.second_diff[1], r.se[1]
    end

    worst_est, worst_se, nstruct = 0.0, 0.0, 0
    for row in eachrow(ref)
        est, se = julia_value(models[row.model], row.kind, row.measure, row.weights, row.type)
        label = "$(row.model) $(row.kind) $(row.measure) wts=$(row.weights) $(row.type)"
        worst_est = max(worst_est, abs(est - row.estimate) / max(abs(row.estimate), 1e-3))
        @test isapprox(est, row.estimate; rtol = 1e-5, atol = 1e-8) || (println("EST ", label, " ", est, " vs ", row.estimate); false)
        if isnan(row.std_error)
            # R gives no SE where the second difference is zero by construction (a categorical focal
            # variable with no interaction with the modifier on the link scale); Margins reports a zero
            # with SE 0 and NaN inference
            @test se == 0.0
            nstruct += 1
        else
            # R's SE of a local derivative of an elasticity nests two numerical derivatives and divides by
            # the step 2δ = 0.02, which amplifies its error (about 2%); the closed-form oracle below
            # checks that case tightly
            tol = (row.kind == "local_w" && row.measure == "eyex") ? 3e-2 : 5e-4
            @test isapprox(se, row.std_error; rtol = tol) || (println("SE  ", label, " ", se, " vs ", row.std_error); false)
            worst_se = max(worst_se, abs(se - row.std_error) / row.std_error)
        end
    end
    @test nstruct == 8

    # Independent closed-form oracle for the local derivative of the elasticity (Poisson): the
    # AME of x at w₀ ± δ from `modelcols` (central difference in x), the derivative of it, and
    # the SE by central differences in β.
    let m = models["poisson"], rhs = formula(m).rhs
        X(dd) = modelcols(rhs, Tables.columntable(dd))
        function est(β, scale; w0 = 1.2, δ = 0.01, h = 1e-5)
            function ame(wv)
                dd = copy(d); dd.w .= wv
                dp = copy(dd); dp.x .+= h
                dm = copy(dd); dm.x .-= h
                pred(Xs) = scale === :response ? exp.(Xs * β) : Xs * β
                ŷ = pred(X(dd)); g = (pred(X(dp)) .- pred(X(dm))) ./ (2h)
                mean(g .* dd.x ./ ŷ)
            end
            (ame(w0 + δ) - ame(w0 - δ)) / (2δ)
        end
        β = coef(m)
        for scale in (:response, :link)
            grad = map(eachindex(β)) do k
                st = 1e-5 * max(abs(β[k]), 1e-2)
                bp = copy(β); bp[k] += st; bm = copy(β); bm[k] -= st
                (est(bp, scale) - est(bm, scale)) / (2st)
            end
            sd = second_differences(m, d, :x, :w; at = 1.2, delta = 0.01, measure = :elasticity, scale, warn = false)
            @test sd.second_diff[1] ≈ est(β, scale) rtol = 1e-6
            @test sd.se[1] ≈ sqrt(grad' * vcov(m) * grad) rtol = 2e-3
        end
    end

    # R's SE comes from a numerical Jacobian (about 2e-4 relative here). Where the answer is known
    # exactly Margins matches it to rounding: on the link scale the second difference of the
    # AME of x across t is the x:t coefficient, and its SE is that coefficient's SE.
    for mn in ("logit", "poisson")
        m = models[mn]
        k = findfirst(==("x & t"), coefnames(m))
        sd = second_differences(m, d, :x, :t; scale = :link, warn = false)
        @test sd.second_diff[1] ≈ coef(m)[k] rtol = 1e-10
        @test sd.se[1] ≈ sqrt(vcov(m)[k, k]) rtol = 1e-10
    end
    println("max relative difference vs R: estimate ", worst_est, ", se ", worst_se)
end
