# test_prediction_intervals.jl
#
# Tests for prediction intervals (interval=:prediction) on profile predictions.
# See notes/PREDICTION_INTERVALS_PLAN.md for the statistical design.
#
# A prediction interval widens the confidence interval by the observation-level
# residual variance:
#   CI:  ŷ ± z·√(SE²)
#   PI:  ŷ ± z·√(SE² + σ̂²),   σ̂² = RSS/(n−p) = deviance/dof_residual
# matching R's predict(lm, interval = "prediction") (with a normal quantile).
#
# The core test validates against an INDEPENDENT linear-algebra construction
# (design row + vcov + residual variance), not against the package's own
# reported se — so a wrong gradient/design point at the grid would be caught.

using Test
using Random
using Statistics
using LinearAlgebra: dot
using DataFrames, GLM, StatsModels
using Distributions: Normal, quantile
using Margins

@testset "Prediction Intervals" begin
    Random.seed!(20260629)
    n = 200
    df = DataFrame(x = randn(n), z = randn(n))
    df.y = 1.0 .+ 2.0 .* df.x .- 0.5 .* df.z .+ randn(n)
    m = lm(@formula(y ~ x + z), df)

    grid = DataFrame(x = [-1.0, 0.0, 1.0], z = [0.5, 0.0, -0.5])
    z975 = quantile(Normal(), 0.975)

    @testset "PI matches independent linear-algebra construction" begin
        # Independent ground truth: for ŷ₀ = x₀'β on a linear model,
        #   se(ŷ₀) = √(x₀' Σ x₀),  Σ = vcov(m)
        #   PI half-width = z · √(se² + σ̂²),  σ̂² = deviance/dof_residual
        # None of these quantities are read back from the package output.
        Σ = vcov(m)
        β = coef(m)               # order: (Intercept), x, z
        σ² = deviance(m) / dof_residual(m)
        @test σ² > 0

        rp = DataFrame(profile_margins(m, df, grid; type=:predictions, interval=:prediction))

        for i in 1:nrow(grid)
            x0 = [1.0, grid.x[i], grid.z[i]]
            est_indep = dot(x0, β)
            se_indep  = sqrt(dot(x0, Σ * x0))
            half      = z975 * sqrt(se_indep^2 + σ²)

            @test isapprox(rp.estimate[i], est_indep; atol=1e-8)
            @test isapprox(rp.se[i],       se_indep;  atol=1e-8)   # se = mean-prediction SE
            @test isapprox(rp.ci_lower[i], est_indep - half; atol=1e-8)
            @test isapprox(rp.ci_upper[i], est_indep + half; atol=1e-8)
        end
    end

    @testset "CI is the σ²→0 limit of PI (confidence path unchanged)" begin
        Σ = vcov(m)
        β = coef(m)
        rc = DataFrame(profile_margins(m, df, grid; type=:predictions))  # default = confidence
        for i in 1:nrow(grid)
            x0 = [1.0, grid.x[i], grid.z[i]]
            se_indep = sqrt(dot(x0, Σ * x0))
            half = z975 * se_indep   # NO residual term
            @test isapprox(rc.ci_lower[i], dot(x0, β) - half; atol=1e-8)
            @test isapprox(rc.ci_upper[i], dot(x0, β) + half; atol=1e-8)
        end
    end

    @testset "se/estimate untouched; only bounds widen" begin
        rc = DataFrame(profile_margins(m, df, grid; type=:predictions))
        rp = DataFrame(profile_margins(m, df, grid; type=:predictions, interval=:prediction))
        @test rc.estimate == rp.estimate
        @test rc.se == rp.se
        # Strictly wider on both sides, by the same residual inflation every row
        @test all(rp.ci_lower .< rc.ci_lower)
        @test all(rp.ci_upper .> rc.ci_upper)
        σ² = deviance(m) / dof_residual(m)
        for i in 1:nrow(rc)
            ci_half = (rc.ci_upper[i] - rc.ci_lower[i]) / 2
            pi_half = (rp.ci_upper[i] - rp.ci_lower[i]) / 2
            # pi_half = z√(se²+σ²), ci_half = z·se  ⇒  pi_half² - ci_half² = z²σ²
            @test isapprox(pi_half^2 - ci_half^2, z975^2 * σ²; atol=1e-7)
        end
    end

    @testset "ci_alpha controls the level" begin
        r95 = DataFrame(profile_margins(m, df, grid; type=:predictions, interval=:prediction))
        r90 = DataFrame(profile_margins(m, df, grid; type=:predictions, interval=:prediction, ci_alpha=0.10))
        # Narrower confidence level → narrower prediction interval
        @test all(r90.ci_upper .< r95.ci_upper)
        @test all(r90.ci_lower .> r95.ci_lower)
        # And the right quantile is used: ratio of half-widths is z90/z95 only after
        # removing σ², so check directly against the analytical half-widths.
        σ² = deviance(m) / dof_residual(m)
        Σ = vcov(m); β = coef(m)
        z90 = quantile(Normal(), 0.95)
        for i in 1:nrow(grid)
            x0 = [1.0, grid.x[i], grid.z[i]]
            se = sqrt(dot(x0, Σ * x0))
            @test isapprox(r90.ci_upper[i], dot(x0, β) + z90*sqrt(se^2 + σ²); atol=1e-8)
        end
    end

    @testset "default is confidence" begin
        rc   = DataFrame(profile_margins(m, df, grid; type=:predictions))
        rdef = DataFrame(profile_margins(m, df, grid; type=:predictions, interval=:confidence))
        @test rc.ci_lower == rdef.ci_lower
        @test rc.ci_upper == rdef.ci_upper
    end

    @testset "Normal-family GLM with identity link is eligible" begin
        # Exercises the GlmResp+Normal branch of _is_gaussian_identity; statistically
        # identical to OLS, so the prediction intervals must coincide with the lm fit.
        gm = glm(@formula(y ~ x + z), df, Normal(), IdentityLink())
        rg = DataFrame(profile_margins(gm, df, grid; type=:predictions, interval=:prediction))
        rl = DataFrame(profile_margins(m,  df, grid; type=:predictions, interval=:prediction))
        @test isapprox(rg.ci_lower, rl.ci_lower; rtol=1e-6)
        @test isapprox(rg.ci_upper, rl.ci_upper; rtol=1e-6)
    end

    @testset "error paths (reason-matched)" begin
        # GLMs are not eligible — must fail for the Gaussian-eligibility reason,
        # not some unrelated ArgumentError. Fit outside @test_throws so only the
        # profile_margins call is under test.
        gm = glm(@formula((y > 1) ~ x + z), df, Binomial(), LogitLink())
        @test_throws "Gaussian" profile_margins(gm, df, grid; type=:predictions, interval=:prediction)

        # Effects have no prediction-interval analogue
        @test_throws "type=:predictions" profile_margins(m, df, grid; type=:effects, vars=[:x], interval=:prediction)

        # Invalid interval symbol
        @test_throws "must be :confidence or :prediction" profile_margins(m, df, grid; type=:predictions, interval=:bogus)
    end
end
