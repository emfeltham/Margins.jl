# test/validation/test_profile_categorical_contrasts.jl
#
# Regression tests for categorical contrasts in `profile_margins`.
#
# Motivation: the feature CLAUDE.md documents as "row-specific baseline contrasts" was
# unreachable. The filter in `_mem_continuous_and_categorical` dropped a categorical if it
# was a mixture *or* if it appeared in the reference grid, and grid completion fills an
# absent categorical with a mixture — so naming the variable in the grid made it
# "scenario-defining" and leaving it out made it a mixture. Every route returned 0 rows, and
# `test/test_profile_pairwise.jl` was excluded from runtests.jl because it would fail.
#
# The two modes now available for a categorical named in `vars`:
#   pinned to discrete levels in the grid -> one contrast per grid row (that row's level vs
#     the baseline), rows already at the baseline contributing none
#   not pinned (a completed mixture)      -> the full level enumeration at each grid row
# `contrasts=:pairwise` always enumerates. Auto-detected `vars` is unchanged: categoricals
# stay scenario-defining or backdrop (see test/regression/test_bool_mixture_skip.jl).
#
# Oracle: for a model that is linear in the parameters where the categorical does not
# interact, a link-scale contrast between two levels is exactly the difference of their
# coefficients, and its SE is the delta-method SE of that difference — independent of the
# profile. Where an interaction is present, the expected value is built from the design rows.

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, LinearAlgebra, Random
using Margins

# Delta-method SE of c'β
_se(c, Σ) = sqrt(max(0.0, dot(c, Σ, c)))

function _contrast_data(; n=300, seed=515)
    Random.seed!(seed)
    x = randn(n)
    z = randn(n)
    g = categorical(rand(["a", "b", "c"], n))
    h = categorical(rand(["p", "q"], n))
    η = 0.4 .+ 0.6 .* x .+ 0.3 .* z .+ 0.5 .* (g .== "b") .- 0.2 .* (g .== "c")
    return DataFrame(y = rand(n) .< 1 ./ (1 .+ exp.(-η)), x=x, z=z, g=g, h=h)
end

# Index of a coefficient by name fragment, tolerating "g: b" / "g:b"
function _ci(model, fragment)
    cn = coefnames(model)
    idx = findfirst(s -> occursin(fragment, replace(String(s), " " => "")), cn)
    @assert !isnothing(idx) "no coefficient matching $fragment in $cn"
    return idx
end

@testset "profile categorical contrasts" begin

    df = _contrast_data()

    @testset "unpinned categorical: full enumeration at the profile ($linkname)" for
            (linkname, m) in (("logit", glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())),
                              ("linear", lm(@formula(y ~ x + z + g), df)))
        β = coef(m); Σ = GLM.vcov(m)
        ib = _ci(m, "g:b"); ic = _ci(m, "g:c")

        # means_grid leaves g as a mixture, so every non-baseline level is contrasted
        res = DataFrame(profile_margins(m, df, means_grid(df); type=:effects, vars=[:g], scale=:link))
        @test nrow(res) == 2
        for (contrast, idx) in (("b - a", ib), ("c - a", ic))
            r = only(eachrow(res[res.contrast .== contrast, :]))
            c = zeros(length(β)); c[idx] = 1.0
            @test r.estimate ≈ β[idx] atol=1e-10
            @test r.se ≈ _se(c, Σ) atol=1e-10
            @test r.variable == "g"
        end

        # Pairwise adds the non-baseline pair
        pw = DataFrame(profile_margins(m, df, means_grid(df); type=:effects, vars=[:g],
                                       scale=:link, contrasts=:pairwise))
        @test nrow(pw) == 3
        r = only(eachrow(pw[pw.contrast .== "c - b", :]))
        c = zeros(length(β)); c[ic] = 1.0; c[ib] = -1.0
        @test r.estimate ≈ β[ic] - β[ib] atol=1e-10
        @test r.se ≈ _se(c, Σ) atol=1e-10
    end

    @testset "pinned categorical: one contrast per grid row" begin
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        β = coef(m); Σ = GLM.vcov(m)
        ib = _ci(m, "g:b"); ic = _ci(m, "g:c")

        grid = cartesian_grid(x=[-1.0, 1.0], g=["a", "b", "c"])
        res = DataFrame(profile_margins(m, df, grid; type=:effects, vars=[:g], scale=:link))

        # 6 grid rows, but the two rows pinned to the baseline contribute nothing
        @test nrow(res) == 4
        @test !any(res.contrast .== "a - a")
        @test sort(unique(res.x)) == [-1.0, 1.0]
        for r in eachrow(res)
            idx = r.contrast == "b - a" ? ib : ic
            c = zeros(length(β)); c[idx] = 1.0
            @test r.estimate ≈ β[idx] atol=1e-10
            @test r.se ≈ _se(c, Σ) atol=1e-10
        end
        # The grid's own level is reported alongside each contrast
        @test Set(string.(res.g)) == Set(["b", "c"])
    end

    @testset "response scale differs and follows the profile" begin
        # On the response scale a contrast is Δμ at that row's covariates, so it must vary
        # across profiles and differ from the link scale.
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        β = coef(m); Σ = GLM.vcov(m)
        link = GLM.Link(m.model)
        ib = _ci(m, "g:b")
        iint = _ci(m, "Intercept"); ix = _ci(m, "x"); iz = _ci(m, "z")

        grid = cartesian_grid(x=[-1.0, 1.5], z=[0.0])
        res = DataFrame(profile_margins(m, df, grid; type=:effects, vars=[:g], scale=:response))
        @test nrow(res) == 4  # 2 profiles × 2 non-baseline levels

        for xval in (-1.0, 1.5)
            r = only(eachrow(res[(res.x .== xval) .& (res.contrast .== "b - a"), :]))
            # η at baseline and at level b, with z = 0 and g's dummies set explicitly
            η_a = β[iint] + β[ix] * xval + β[iz] * 0.0
            η_b = η_a + β[ib]
            @test r.estimate ≈ GLM.linkinv(link, η_b) - GLM.linkinv(link, η_a) atol=1e-9
            # Gradient: g'(η_b)·X_b − g'(η_a)·X_a
            Xa = zeros(length(β)); Xa[iint] = 1.0; Xa[ix] = xval
            Xb = copy(Xa); Xb[ib] = 1.0
            grad = GLM.mueta(link, η_b) .* Xb .- GLM.mueta(link, η_a) .* Xa
            @test r.se ≈ _se(grad, Σ) atol=1e-9
        end
        # Different profiles must give different response-scale contrasts
        est = res[res.contrast .== "b - a", :estimate]
        @test !isapprox(est[1], est[2]; atol=1e-6)
    end

    @testset "interaction: contrast depends on the profile even on the link scale" begin
        m = glm(@formula(y ~ x * g + z), df, Binomial(), LogitLink())
        β = coef(m); Σ = GLM.vcov(m)
        ib = _ci(m, "g:b")
        ixb = findfirst(s -> occursin("x", String(s)) && occursin("g:b", replace(String(s), " " => "")),
                        coefnames(m))
        @test !isnothing(ixb)

        grid = cartesian_grid(x=[0.0, 2.0], z=[0.0])
        res = DataFrame(profile_margins(m, df, grid; type=:effects, vars=[:g], scale=:link))
        for xval in (0.0, 2.0)
            r = only(eachrow(res[(res.x .== xval) .& (res.contrast .== "b - a"), :]))
            c = zeros(length(β)); c[ib] = 1.0; c[ixb] = xval
            @test r.estimate ≈ β[ib] + xval * β[ixb] atol=1e-9
            @test r.se ≈ _se(c, Σ) atol=1e-9
        end
    end

    @testset "continuous and categorical together" begin
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        β = coef(m); Σ = GLM.vcov(m)
        res = DataFrame(profile_margins(m, df, means_grid(df); type=:effects,
                                        vars=[:x, :g], scale=:link))
        @test nrow(res) == 3                       # x plus two g contrasts
        @test Set(res.variable) == Set(["x", "g"])
        xrow = only(eachrow(res[res.variable .== "x", :]))
        @test xrow.estimate ≈ β[_ci(m, "x")] atol=1e-10   # link-scale slope is the coefficient
        @test xrow.contrast == "dy/dx"
    end

    @testset "Bool variables" begin
        Random.seed!(21); n = 200
        dfb = DataFrame(x=randn(n), b=rand(Bool, n))
        dfb.y = 1 .+ 0.4 .* dfb.x .+ 0.7 .* dfb.b .+ 0.1 .* randn(n)
        mb = lm(@formula(y ~ x + b), dfb)
        βb = coef(mb); Σb = GLM.vcov(mb)
        ibb = _ci(mb, "b")
        c = zeros(length(βb)); c[ibb] = 1.0

        # Unpinned (completed to a frequency), pinned as Bool, and pinned as exact 0.0/1.0
        for (label, grid, expected_rows) in (
                ("unpinned", means_grid(dfb), 1),
                ("pinned bool", cartesian_grid(x=[0.0], b=[false, true]), 1),   # false is baseline
                ("pinned float", DataFrame(x=[0.0, 0.0], b=[0.0, 1.0]), 1))
            res = DataFrame(profile_margins(mb, dfb, grid; type=:effects, vars=[:b]))
            @test nrow(res) == expected_rows
            @test all(res.contrast .== "true - false")
            for r in eachrow(res)
                @test r.estimate ≈ βb[ibb] atol=1e-10
                @test r.se ≈ _se(c, Σb) atol=1e-10
            end
        end
    end

    @testset "grid larger than the data" begin
        # The contrast evaluator is built at grid length; splicing in the original column
        # would fail here.
        Random.seed!(5); n = 12
        small = DataFrame(x=randn(n), g=categorical(rand(["a", "b"], n)))
        small.y = 1 .+ small.x .+ (small.g .== "b") .+ 0.1 .* randn(n)
        ms = lm(@formula(y ~ x + g), small)
        grid = cartesian_grid(x=collect(range(-2, 2, length=40)))
        res = DataFrame(profile_margins(ms, small, grid; type=:effects, vars=[:g], scale=:link))
        @test nrow(res) == 40
        @test all(res.estimate .≈ coef(ms)[_ci(ms, "g:b")])
    end

    @testset "auto-detected vars stay unchanged" begin
        # Historical behaviour, pinned down by test/regression/test_bool_mixture_skip.jl:
        # without an explicit vars=, categoricals are never contrasted.
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        auto = DataFrame(profile_margins(m, df, means_grid(df); type=:effects))
        @test Set(auto.variable) == Set(["x", "z"])

        auto_grid = DataFrame(profile_margins(m, df, cartesian_grid(g=["a", "b"]); type=:effects))
        @test Set(auto_grid.variable) == Set(["x", "z"])
    end

    @testset "requesting a categorical not in the model errors" begin
        m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())
        @test_throws Exception profile_margins(m, df, means_grid(df); type=:effects, vars=[:nonexistent])
    end
end
