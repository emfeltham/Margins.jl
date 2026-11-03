# test/core/test_profile_fill_weights.jl
#
# The values `profile_margins` (and `complete_reference_grid`) fill in for model variables a
# reference grid leaves out use the model's estimation weights by default (`weights=:model`), as
# Stata's `margins` uses the estimation weights for the `at()` values it computes. `weights=:none`
# on `complete_reference_grid` gives the unweighted (2.x) values; unweighted fits are unchanged.

using Test, DataFrames, CategoricalArrays, GLM, Statistics, Logging
using MixedModels, FixedEffectModels
using Margins

@testset "profile fill-ins use the model's estimation weights" begin
    fr(t) = t - floor(t); n = 300; i = 1:n
    d = DataFrame(x = 2 .* fr.(i .* 0.6180339887498949), z = fr.(i .* 0.7548776662466927),
                  b = fr.(i .* 0.41) .< 0.4,
                  g = categorical(["a", "b", "c"][1 .+ mod.(i .* 7 .+ div.(i, 5), 3)]),
                  w = 0.5 .+ 2 .* fr.(i .* 0.3247179572447460), cl = mod.(i, 12))
    d.y = 1 .+ d.x .+ 0.5 .* d.z .+ 0.3 .* d.b .+ 0.2 .* (d.g .== "b") .+ (fr.(i .* i .* 0.123) .- 0.5)
    d.yb = fr.(i .* i .* 0.271) .< 1 ./ (1 .+ exp.(-(d.x .- 1 .+ 0.5 .* d.z)))
    f = @formula(y ~ x + z + b + g)
    mw = lm(f, d; wts = d.w)
    mu = lm(f, d)
    cg = cartesian_grid(x = [0.5, 1.5])
    wmean(v, w) = sum(w .* v) / sum(w)
    shares(w) = [sum(w[d.g .== l]) / sum(w) for l in ["a", "b", "c"]]
    # a mixture as level => share (its level order is a Dict's, see _create_frequency_mixture)
    mixshares(m) = [Dict(string(l) => v for (l, v) in zip(m.levels, m.weights))[k] for k in ("a", "b", "c")]
    samecell(a, b) = a isa CategoricalMixture ? (b isa CategoricalMixture && mixshares(a) == mixshares(b)) : isequal(a, b)
    samegrid(a, b) = names(a) == names(b) && size(a) == size(b) && all(samecell(a[r, c], b[r, c]) for r in 1:nrow(a), c in names(a))

    @testset "weighted fit: weighted means, shares and mixture (hand computation)" begin
        c = complete_reference_grid(cg, mw, d)
        @test names(c) == ["x", "z", "b", "g"]
        @test all(c.z .≈ wmean(d.z, d.w)) && all(c.b .≈ wmean(d.b, d.w))
        @test mixshares(c.g[1]) ≈ shares(d.w)
        # profile_margins fills in the same values: equal to evaluating the hand-built grid
        hand = DataFrame(x = [0.5, 1.5], z = fill(wmean(d.z, d.w), 2), b = fill(wmean(d.b, d.w), 2),
                         g = fill(mix("a" => shares(d.w)[1], "b" => shares(d.w)[2], "c" => shares(d.w)[3]), 2))
        for type in (:predictions, :effects)
            r = profile_margins(mw, d, cg; type)
            rh = profile_margins(mw, d, hand; type)
            @test DataFrame(r).estimate ≈ DataFrame(rh).estimate rtol = 1e-13
            @test DataFrame(r).se ≈ DataFrame(rh).se rtol = 1e-12
            @test r.metadata[:weights_source] === :model && r.metadata[:weighted] === true
        end
        # a logit fit too
        gw = glm(@formula(yb ~ x + z + g), d, Binomial(), LogitLink(); wts = d.w)
        @test all(complete_reference_grid(cg, gw, d).z .≈ wmean(d.z, d.w))
    end

    @testset ":none reproduces the unweighted (2.x) fill-ins exactly" begin
        c0 = complete_reference_grid(cg, mw, d; weights = :none)
        @test all(c0.z .== mean(d.z)) && all(c0.b .== mean(d.b))
        @test mixshares(c0.g[1]) == [count(==(l), d.g) / n for l in ["a", "b", "c"]]
        @test samegrid(c0, complete_reference_grid(cg, mw, d; weights = nothing))
        r0 = profile_margins(mw, d, c0; type = :predictions)
        @test r0.metadata[:weights_source] === :none && r0.metadata[:weighted] === false
        # the weighted default moves the point, not the method: the two differ
        @test DataFrame(r0).estimate != DataFrame(profile_margins(mw, d, cg; type = :predictions)).estimate
    end

    @testset "unweighted fit: bitwise unchanged" begin
        @test samegrid(complete_reference_grid(cg, mu, d), complete_reference_grid(cg, mu, d; weights = :none))
        for type in (:predictions, :effects)
            a = profile_margins(mu, d, cg; type)
            b = profile_margins(mu, d, complete_reference_grid(cg, mu, d; weights = :none); type)
            @test isequal(DataFrame(a).estimate, DataFrame(b).estimate) && isequal(DataFrame(a).se, DataFrame(b).se)
            @test a.metadata[:weights_source] === :none && a.metadata[:weighted] === false
        end
    end

    @testset "zero weights contribute nothing" begin
        w0 = copy(d.w); w0[1:3:end] .= 0.0
        m0 = lm(f, d; wts = w0)
        c = complete_reference_grid(cg, m0, d)
        @test all(c.z .≈ wmean(d.z, w0)) && mixshares(c.g[1]) ≈ shares(w0)
        keep = w0 .> 0
        @test all(c.z .≈ wmean(d.z[keep], w0[keep]))
    end

    @testset "a grid that sets every model variable reads no weights" begin
        full = DataFrame(x = [0.5], z = [0.2], b = [true], g = ["b"])
        r = profile_margins(mw, d, full; type = :predictions)
        @test r.metadata[:weights_source] === :none
        @test DataFrame(r).estimate == DataFrame(profile_margins(mw, d, complete_reference_grid(full, mw, d; weights = :none);
                                                                 type = :predictions)).estimate
    end

    @testset "explicit weights and errors" begin
        @test all(complete_reference_grid(cg, mu, d; weights = :w).z .≈ wmean(d.z, d.w))   # a column, any fit
        # a custom typical function cannot be weighted: an error for a weighted fit, fine with :none
        @test_throws ArgumentError complete_reference_grid(cg, mw, d; typical = maximum)
        @test all(complete_reference_grid(cg, mw, d; typical = maximum, weights = :none).z .== maximum(d.z))
        @test all(complete_reference_grid(cg, mu, d; typical = maximum).z .== maximum(d.z))   # unweighted fit
        # model weights that cannot be matched to the data's rows: an error only when a value is filled in
        dd = vcat(d, d[1:5, :])
        @test_throws ArgumentError profile_margins(mw, dd, cg; type = :predictions)
        @test profile_margins(mw, dd, DataFrame(x = [0.5], z = [0.2], b = [true], g = ["b"]); type = :predictions) isa PredictionsResult
        @test all(complete_reference_grid(cg, mw, dd; weights = :none).z .== mean(dd.z))
    end

    @testset "MixedModels: the fit's weights" begin
        lmm = fit(MixedModel, @formula(y ~ x + z + b + (1 | g)), d; wts = d.w, progress = false)
        c = complete_reference_grid(cartesian_grid(x = [0.5]), lmm, d)
        @test c.z[1] ≈ wmean(d.z, d.w) && c.b[1] ≈ wmean(d.b, d.w)
        r = profile_margins(lmm, d, cartesian_grid(x = [0.5]); type = :predictions)
        @test r.metadata[:weights_source] === :model
        lmm0 = fit(MixedModel, @formula(y ~ x + z + b + (1 | g)), d; progress = false)
        @test complete_reference_grid(cartesian_grid(x = [0.5]), lmm0, d).z[1] == mean(d.z)
    end

    @testset "FixedEffectModels: weights unknown, fill-ins unweighted" begin
        fm = reg(d, @formula(y ~ x + z + b + fe(g)); weights = :w)
        c = with_logger(NullLogger()) do
            complete_reference_grid(cartesian_grid(x = [0.5]), fm, d)
        end
        @test c.z[1] == mean(d.z)
        r = with_logger(NullLogger()) do
            profile_margins(fm, d, cartesian_grid(x = [0.5], g = ["a"]); type = :effects, vars = [:x])
        end
        @test r.metadata[:weights_source] === :none
        @test complete_reference_grid(cartesian_grid(x = [0.5]), fm, d; weights = :w).z[1] ≈ wmean(d.z, d.w)
    end

    @testset "means_grid(model, data): Stata's atmeans (model weights, estimation sample)" begin
        mg = means_grid(mw, d)
        @test names(mg) == ["x", "z", "b", "g"]                  # formula variables only, no response
        @test mg.x[1] ≈ wmean(d.x, d.w) && mg.z[1] ≈ wmean(d.z, d.w) && mg.b[1] ≈ wmean(d.b, d.w)
        @test mixshares(mg.g[1]) ≈ shares(d.w)
        # the documented translation of `margins, dydx(*) atmeans` after a weighted fit equals the
        # effects at a hand-built grid of the weighted means
        hand = DataFrame(x = [wmean(d.x, d.w)], z = [wmean(d.z, d.w)], b = [wmean(d.b, d.w)],
                         g = [mix("a" => shares(d.w)[1], "b" => shares(d.w)[2], "c" => shares(d.w)[3])])
        for type in (:effects, :predictions)
            a = DataFrame(profile_margins(mw, d, means_grid(mw, d); type))
            h = DataFrame(profile_margins(mw, d, hand; type))
            @test a.estimate ≈ h.estimate rtol = 1e-13
            @test a.se ≈ h.se rtol = 1e-12
        end
        # weights=:none and unweighted fits: the unweighted means (= means_grid(data) on the formula's columns)
        m0 = means_grid(mw, d; weights = :none)
        @test m0.x[1] == mean(d.x) && m0.b[1] == mean(d.b)
        @test means_grid(mu, d).z[1] == means_grid(d).z[1]
        @test means_grid(mw, d; weights = :w).z[1] == means_grid(mw, d).z[1]
        @test means_grid(mw, d; typical = median).x[1] == Margins.typical_value(d.x, median; weights = d.w)
        @test_throws ArgumentError means_grid(mw, d; typical = maximum)
        # a variable the model treats as categorical gets its mixture over the model's levels
        dk = copy(d); dk.k = mod.(1:n, 3)
        mk = lm(@formula(y ~ x + k), dk; wts = dk.w, contrasts = Dict(:k => DummyCoding()))
        @test means_grid(mk, dk).k[1] isa CategoricalMixture
    end

    @testset "fill-ins over the estimation sample (FixedEffectModels: singletons, missing values)" begin
        d2 = copy(d)
        d2.hh = [k <= 6 ? 1000 + k : mod(k, 25) for k in 1:n]     # rows 1-6: singleton groups
        zm = allowmissing(d2.z); zm[[10, 50, 90, 200]] .= missing; d2.z = zm
        fm2 = reg(d2, @formula(y ~ x + z + b + fe(hh)); weights = :w, save = :fe)
        es = fm2.esample
        @test count(!, es) == 10 && !any(es[1:6])
        quiet(f) = with_logger(f, NullLogger())
        cg1 = cartesian_grid(x = [0.5])
        c = quiet(() -> complete_reference_grid(cg1, fm2, d2))      # FEM weights unknown: unweighted
        @test c.z[1] == mean(Float64.(d2.z[es])) && c.b[1] == mean(d2.b[es])
        @test c.z[1] != mean(skipmissing(d2.z))
        cw = complete_reference_grid(cg1, fm2, d2; weights = :w)
        @test cw.z[1] ≈ wmean(Float64.(d2.z[es]), d2.w[es]) && cw.b[1] ≈ wmean(d2.b[es], d2.w[es])
        @test complete_reference_grid(cg1, fm2, d2; weights = d2.w).z[1] == cw.z[1]   # a vector for every row
        @test complete_reference_grid(cg1, fm2, d2; weights = :none).z[1] == c.z[1]
        # the table given to reg() is required, as for population_margins
        @test_throws ArgumentError complete_reference_grid(cg1, fm2, d2[es, :])
        # profile_margins fills in the same values (its request already holds the estimation rows)
        r = quiet(() -> profile_margins(fm2, d2, cartesian_grid(x = [0.5], hh = [3]); type = :predictions))
        rh = profile_margins(fm2, d2, DataFrame(x = [0.5], z = [c.z[1]], b = [c.b[1]], hh = [3]); type = :predictions)
        @test DataFrame(r).estimate == DataFrame(rh).estimate
        # means_grid(model, data) over esample too
        mg = means_grid(fm2, d2; weights = :w)
        @test mg.z[1] ≈ wmean(Float64.(d2.z[es]), d2.w[es]) && mg.x[1] ≈ wmean(d2.x[es], d2.w[es])
        @test !("hh" in names(mg))                                       # absorbed fixed effects are not filled
        @test quiet(() -> means_grid(fm2, d2)).x[1] == mean(d2.x[es])     # weights unknown: unweighted
    end
end
