# test/validation/test_link_scale.jl
#
# Regression tests for `scale=:link`.
#
# Motivation: `scale` used to be accepted and then silently ignored on several code paths,
# so `scale=:link` returned response-scale numbers under a link-scale label. The existing
# suite could not see it because every test that paired `scale=:link` with a categorical
# `vars=` used an identity-link `lm`, where the two scales coincide. These tests use
# non-identity links (logit, probit) so the scales genuinely differ.
#
# Key identity exploited throughout: for a model that is linear in the parameters and where
# the variable of interest does not interact with anything, the LINK-scale effect is exactly
# the coefficient, and its delta-method SE is exactly the coefficient's SE.
#   - continuous x:      ∂η/∂x   = β_x
#   - categorical level: Δη      = β_level  (baseline contrast)

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, LinearAlgebra, Random
using Margins

# Locate a coefficient by name fragment, tolerating "g: b" / "g:b" spellings
function _coef_index(model, fragment::AbstractString)
    cn = coefnames(model)
    idx = findfirst(s -> occursin(fragment, replace(String(s), " " => "")), cn)
    @assert !isnothing(idx) "coefficient matching $fragment not found in $cn"
    return idx
end

function _link_scale_data(; n=400, seed=42)
    Random.seed!(seed)
    x = randn(n)
    g = categorical(rand(["a", "b", "c"], n))
    h = categorical(rand(["p", "q"], n))
    η = 0.3 .+ 0.8 .* x .+ 0.5 .* (g .== "b") .- 0.4 .* (g .== "c")
    y = rand(n) .< 1 ./ (1 .+ exp.(-η))
    return DataFrame(y=y, x=x, g=g, h=h, w=rand(n) .+ 0.1)
end

@testset "scale=:link is honored (non-identity link)" begin

    @testset "population categorical contrasts equal coefficients on link scale" begin
        for (dist, link) in ((Binomial(), LogitLink()), (Binomial(), ProbitLink()))
            df = _link_scale_data()
            m = glm(@formula(y ~ x + g), df, dist, link)

            res = DataFrame(population_margins(m, df; type=:effects, vars=[:x, :g], scale=:link))

            # Continuous: dy/dx on the link scale is exactly β_x
            xrow = only(findall(==("x"), res.variable))
            @test res.estimate[xrow] ≈ coef(m)[_coef_index(m, "x")] atol=1e-10
            @test res.se[xrow] ≈ stderror(m)[_coef_index(m, "x")] atol=1e-10

            # Categorical: baseline contrasts on the link scale are exactly the coefficients
            for (contrast, fragment) in (("b - a", "g:b"), ("c - a", "g:c"))
                row = only(findall(r -> r.variable == "g" && r.contrast == contrast, eachrow(res)))
                idx = _coef_index(m, fragment)
                @test res.estimate[row] ≈ coef(m)[idx] atol=1e-10
                @test res.se[row] ≈ stderror(m)[idx] atol=1e-10
            end
        end
    end

    @testset "link and response categorical effects differ under a non-identity link" begin
        # This is the assertion the old suite was missing entirely: the two scales were
        # returning bit-identical categorical numbers.
        df = _link_scale_data()
        m = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())

        link_res = DataFrame(population_margins(m, df; type=:effects, vars=[:g], scale=:link))
        resp_res = DataFrame(population_margins(m, df; type=:effects, vars=[:g], scale=:response))

        @test link_res.contrast == resp_res.contrast  # same rows, so comparison is meaningful
        for i in eachindex(link_res.estimate)
            @test !isapprox(link_res.estimate[i], resp_res.estimate[i]; atol=1e-6)
            @test !isapprox(link_res.se[i], resp_res.se[i]; atol=1e-6)
        end
        # Response-scale contrasts are probability differences, hence bounded
        @test all(abs.(resp_res.estimate) .< 1.0)
    end

    @testset "pairwise contrasts on link scale" begin
        df = _link_scale_data()
        m = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:g],
                                           scale=:link, contrasts=:pairwise))
        @test nrow(res) == 3  # 3 levels ⇒ 3 unique pairs

        βb = coef(m)[_coef_index(m, "g:b")]
        βc = coef(m)[_coef_index(m, "g:c")]
        expected = Dict("b - a" => βb, "c - a" => βc, "c - b" => βc - βb)
        for r in eachrow(res)
            @test r.estimate ≈ expected[r.contrast] atol=1e-10
        end
    end

    @testset "weighted categorical contrasts on link scale" begin
        # On the link scale the per-row contrast is constant, so any weighting must give
        # the same answer as the unweighted case (and equal the coefficient).
        df = _link_scale_data()
        m = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:g],
                                           scale=:link, weights=:w))
        for (contrast, fragment) in (("b - a", "g:b"), ("c - a", "g:c"))
            row = only(findall(==(contrast), res.contrast))
            idx = _coef_index(m, fragment)
            @test res.estimate[row] ≈ coef(m)[idx] atol=1e-10
            @test res.se[row] ≈ stderror(m)[idx] atol=1e-10
        end
    end

    @testset "categorical contrasts on link scale under scenarios" begin
        df = _link_scale_data()
        m = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:g],
                                           scale=:link, scenarios=(x=0.0,)))
        for (contrast, fragment) in (("b - a", "g:b"), ("c - a", "g:c"))
            row = only(findall(==(contrast), res.contrast))
            idx = _coef_index(m, fragment)
            @test res.estimate[row] ≈ coef(m)[idx] atol=1e-10
            @test res.se[row] ≈ stderror(m)[idx] atol=1e-10
        end
    end

    @testset "categorical contrasts on link scale under groups" begin
        df = _link_scale_data()
        m = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:g],
                                           scale=:link, groups=:h))
        @test nrow(res) == 4  # 2 groups × 2 non-baseline levels
        # No g×h interaction, so every group reproduces the coefficients exactly
        for (contrast, fragment) in (("b - a", "g:b"), ("c - a", "g:c"))
            idx = _coef_index(m, fragment)
            for r in eachrow(res)
                r.contrast == contrast || continue
                @test r.estimate ≈ coef(m)[idx] atol=1e-10
                @test r.se ≈ stderror(m)[idx] atol=1e-10
            end
        end
    end

    @testset "elasticity quotient-rule gradient uses the link scale" begin
        # ∂ȳ/∂β must be X̄ on the link scale, not the mueta-weighted mean design row.
        # For y ~ x + z the link-scale AME of x is β_x, ȳ = η̄ = X̄β, and
        #   ε = x̄ β_x / η̄  ⇒  ∂ε/∂β = (x̄/η̄) eₓ - (x̄ β_x / η̄²) X̄
        Random.seed!(7)
        n = 400
        df = DataFrame(x=abs.(randn(n)) .+ 1.0, z=randn(n))
        df.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.3 .+ 0.8 .* df.x .+ 0.4 .* df.z)))
        m = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())

        β = coef(m); Σ = GLM.vcov(m); X = modelmatrix(m)
        ix = _coef_index(m, "x")
        x̄ = mean(df.x); η̄ = mean(X * β); X̄ = vec(mean(X, dims=1))
        eₓ = zeros(length(β)); eₓ[ix] = 1.0

        expected_estimate = x̄ / η̄ * β[ix]
        expected_grad = (x̄ / η̄) .* eₓ .- (x̄ * β[ix] / η̄^2) .* X̄
        expected_se = sqrt(dot(expected_grad, Σ, expected_grad))

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x],
                                           measure=:elasticity, scale=:link))
        @test res.estimate[1] ≈ expected_estimate atol=1e-10
        @test res.se[1] ≈ expected_se atol=1e-10

        # The old bug used the mueta-weighted design mean; assert we are not doing that
        X̄_mueta = vec(mean(GLM.mueta.(LogitLink(), X * β) .* X, dims=1))
        buggy_grad = (x̄ / η̄) .* eₓ .- (expected_estimate / η̄) .* X̄_mueta
        buggy_se = sqrt(dot(buggy_grad, Σ, buggy_grad))
        @test !isapprox(res.se[1], buggy_se; atol=1e-8)
    end

    @testset "identity link: scales coincide (no regression)" begin
        # All of the fixes must be no-ops for lm, where mueta ≡ 1.
        Random.seed!(99)
        n = 200
        g = categorical(rand(["A", "B", "C"], n))
        x = randn(n)
        df = DataFrame(x=x, g=g,
                       y = 1 .+ 0.5 .* x .+ (g .== "B") .* 0.3 .- (g .== "C") .* 0.2 .+ 0.01 .* randn(n))
        m = lm(@formula(y ~ x + g), df)

        for vars in ([:x], [:g], [:x, :g])
            a = DataFrame(population_margins(m, df; type=:effects, vars=vars, scale=:link))
            b = DataFrame(population_margins(m, df; type=:effects, vars=vars, scale=:response))
            @test a.estimate ≈ b.estimate atol=1e-12
            @test a.se ≈ b.se atol=1e-12
        end

        # And elasticities agree across scales under the identity link too
        for measure in (:elasticity, :semielasticity_eydx)
            a = DataFrame(population_margins(m, df; type=:effects, vars=[:x], measure=measure, scale=:link))
            b = DataFrame(population_margins(m, df; type=:effects, vars=[:x], measure=measure, scale=:response))
            @test a.estimate ≈ b.estimate atol=1e-12
            @test a.se ≈ b.se atol=1e-12
        end
    end
end
