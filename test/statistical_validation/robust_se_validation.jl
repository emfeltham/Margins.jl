# robust_se_validation.jl
#
# Robust and clustered standard errors (CovarianceMatrices.jl estimators passed as `vcov=`),
# checked exactly against sandwich matrices built by hand from X, the residuals and the link:
#
#   lm, HC0–HC3     V = (X'X)⁻¹ X' Ω X (X'X)⁻¹, Ω = diag(ωᵢ eᵢ²) with ω = 1, n/(n−k),
#                   1/(1−hᵢ), 1/(1−hᵢ)²
#   lm, CR0         V = (X'X)⁻¹ (Σ_g s_g s_g') (X'X)⁻¹,   s_g = Σ_{i∈g} xᵢ eᵢ
#   CR1             V_CR0 · G/(G−1) · (n−1)/(n−k), as Stata, fixest and sandwich::vcovCL(type =
#                   "HC1"); two-way: V_a + V_b − V_ab, each term with its own G (CovarianceMatrices
#                   before 0.32.2 applied G/(G−1) twice)
#   logit, HC0/HC1  V = B (Σ uᵢuᵢ') B · {1, n/(n−k)},      B = (X'WX)⁻¹, W = diag(μᵢ(1−μᵢ)),
#                   uᵢ = xᵢ (yᵢ − μᵢ)
#   logit, CR0/CR1  V = B (Σ_g u_g u_g') B,                u_g = Σ_{i∈g} uᵢ (CR1 scaled as above)
#
# and every Margins SE in the 2×2 framework is √(g'Vg) with the gradient g written out from
# X, β and the link derivative (μ' = μ(1−μ), μ'' = μ(1−μ)(1−2μ) for the logit).
# CovarianceMatrices' own matrices are compared with the hand ones too, so a convention change
# upstream shows here rather than as a silent SE change.

using Test
using Random
using DataFrames
using Statistics
using GLM
using StatsModels
using Margins
using CovarianceMatrices
using LinearAlgebra

_rse_delta(g, V) = sqrt(g' * V * g)

# CR1's finite-sample factor for one clustering: G/(G−1) · (n−1)/(n−k)
_rse_cr1_factor(cluster, n, k) = (G = length(unique(cluster)); G / (G - 1) * (n - 1) / (n - k))

# CovarianceMatrices < 0.32.2 (what Julia 1.10/1.11 resolve) applies G/(G−1) twice in CR1; Margins
# corrects it (`_evaluate_vcov`), so only upstream's own CR1 matrix depends on the version
const _RSE_CM_CR1_FIXED = pkgversion(CovarianceMatrices) >= v"0.32.2"

# lm sandwich by hand: HC0–HC3 (Ω diagonal) or CR0 (cluster sums of scores)
function _rse_lm_sandwich(model, est; cluster = nothing)
    X = modelmatrix(model); e = GLM.residuals(model); n, k = size(X)
    B = inv(X' * X)
    if est in (:CR0, :CR1)
        S = zeros(k, k)
        for c in unique(cluster)
            s = vec(sum(X[cluster .== c, :] .* e[cluster .== c]; dims = 1))
            S .+= s * s'
        end
        return B * S * B .* (est === :CR1 ? _rse_cr1_factor(cluster, n, k) : 1.0)
    end
    h = [dot(X[i, :], B * X[i, :]) for i in 1:n]
    ω = est === :HC0 ? ones(n) : est === :HC1 ? fill(n / (n - k), n) :
        est === :HC2 ? 1 ./ (1 .- h) : 1 ./ (1 .- h) .^ 2
    return B * (X' * Diagonal(ω .* e .^ 2) * X) * B
end

# logit sandwich by hand, from X, β and y (no GLM internals)
function _rse_logit_sandwich(model, y, est; cluster = nothing)
    X = modelmatrix(model); b = coef(model); n, k = size(X)
    μ = 1 ./ (1 .+ exp.(-(X * b)))
    B = inv(X' * ((μ .* (1 .- μ)) .* X))
    U = X .* (y .- μ)
    if est in (:CR0, :CR1)
        S = zeros(k, k)
        for c in unique(cluster)
            u = vec(sum(U[cluster .== c, :]; dims = 1))
            S .+= u * u'
        end
        return B * S * B .* (est === :CR1 ? _rse_cr1_factor(cluster, n, k) : 1.0)
    end
    return B * (U' * U) * B .* (est === :HC1 ? n / (n - k) : 1.0)
end

# lm: every quadrant's SE from V. AME/MEM of x = β_x (g = e_x); AAP and the APM at the means
# both have g = column means of X (y ~ x + z, so means_grid is the mean row)
function _rse_check_lm_quadrants(model, data, vc, V)
    X = modelmatrix(model); jx = findfirst(==("x"), coefnames(model))
    ex = zeros(size(X, 2)); ex[jx] = 1.0
    xbar = vec(mean(X; dims = 1))
    @test only(DataFrame(population_margins(model, data; vars = [:x], vcov = vc)).se) ≈ _rse_delta(ex, V) rtol = 1e-10
    @test only(DataFrame(population_margins(model, data; type = :predictions, vcov = vc)).se) ≈ _rse_delta(xbar, V) rtol = 1e-10
    @test only(DataFrame(profile_margins(model, data, means_grid(data); vars = [:x], vcov = vc)).se) ≈ _rse_delta(ex, V) rtol = 1e-10
    @test only(DataFrame(profile_margins(model, data, means_grid(data); type = :predictions, vcov = vc)).se) ≈ _rse_delta(xbar, V) rtol = 1e-10
end

@testset "Robust SE Validation" begin

    @testset "lm: HC0–HC3 sandwich by hand, all four quadrants" begin
        data = make_heteroskedastic_data(n = 400, heteroskedasticity_type = :linear)
        model = lm(@formula(y ~ x + z), data)
        base = DataFrame(population_margins(model, data; vars = [:x]))
        for (name, est) in ((:HC0, HR0()), (:HC1, HR1()), (:HC2, HR2()), (:HC3, HR3()))
            @testset "$name" begin
                V = _rse_lm_sandwich(model, name)
                @test CovarianceMatrices.vcov(est, model) ≈ V rtol = 1e-10
                _rse_check_lm_quadrants(model, data, est, V)
                r = DataFrame(population_margins(model, data; vars = [:x], vcov = est))
                @test r.estimate == base.estimate        # vcov= changes SEs only
                @test !isapprox(r.se[1], base.se[1]; rtol = 1e-2)   # heteroskedastic data: robust ≠ model SE
            end
        end
    end

    @testset "lm: CR0 and CR1 clustered SEs by hand" begin
        rng = MersenneTwister(06515)
        n = 240
        region = repeat(1:6, inner = n ÷ 6)       # 6 regions × 40
        year = repeat(1:8, outer = n ÷ 8)         # 8 years, crossed with region
        unbal = vcat(fill(1, 120), fill(2, 60), fill(3, 35), fill(4, 25))   # unbalanced
        data = DataFrame(x = randn(rng, n), z = randn(rng, n), region = region, year = year, unbal = unbal)
        data.y = 2.0 .+ 0.5 .* data.x .+ 0.3 .* data.z .+ (randn(rng, 6) .* 0.6)[region] .+
                 (randn(rng, 8) .* 0.5)[year] .+ 0.2 .* randn(rng, n)
        model = lm(@formula(y ~ x + z), data)
        ses = Float64[]
        for cl in (:region, :year, :unbal)
            @testset "cluster = $cl" begin
                V = _rse_lm_sandwich(model, :CR0; cluster = data[!, cl])
                @test CovarianceMatrices.vcov(CR0(data[!, cl]), model) ≈ V rtol = 1e-10
                _rse_check_lm_quadrants(model, data, CR0(data[!, cl]), V)
                push!(ses, only(DataFrame(population_margins(model, data; vars = [:x], vcov = CR0(data[!, cl]))).se))
                V1 = _rse_lm_sandwich(model, :CR1; cluster = data[!, cl])
                G = length(unique(data[!, cl]))
                @test CovarianceMatrices.vcov(CR1(data[!, cl]), model) ≈ (_RSE_CM_CR1_FIXED ? V1 : V1 .* (G / (G - 1))) rtol = 1e-10
                _rse_check_lm_quadrants(model, data, CR1(data[!, cl]), V1)
            end
        end
        @testset "two-way CR1 (region × year)" begin
            # Cameron–Gelbach–Miller: V_region + V_year − V_region∩year, each one-way CR1 term with
            # its own number of clusters (6, 8 and 48 here), as sandwich::vcovCL does
            inter = string.(data.region, "_", data.year)
            V = sum(sgn .* _rse_lm_sandwich(model, :CR1; cluster = c)
                    for (sgn, c) in ((1, data.region), (1, data.year), (-1, inter)))
            vc = CR1((data.region, data.year))
            _RSE_CM_CR1_FIXED && @test CovarianceMatrices.vcov(vc, model) ≈ V rtol = 1e-10
            _rse_check_lm_quadrants(model, data, vc, V)
        end
        model_se = only(DataFrame(population_margins(model, data; vars = [:x])).se)
        @test all(s -> !isapprox(s, model_se; rtol = 1e-2), ses)    # each clustering is used
        @test !isapprox(ses[1], ses[2]; rtol = 1e-2)
        # an unknown cluster column is an error, not a silent fallback to the model vcov
        @test_throws Exception population_margins(model, data; vars = [:x], vcov = CR0(:nonexistent_var))
    end

    @testset "logit: HC0, HC1, CR0 and CR1 by hand, all four quadrants" begin
        rng = MersenneTwister(20261102)
        n = 500
        data = DataFrame(x = randn(rng, n), z = randn(rng, n), cl = rand(rng, 1:15, n))
        data.y = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-(-0.3 .+ 0.8 .* data.x .+ 0.4 .* data.z .+
                                                          (randn(rng, 15) .* 0.7)[data.cl]))))
        # GLM.jl's vcov (CovarianceMatrices' bread) is (X'WX)⁻¹ at the weights BEFORE the last
        # IRLS step, so it equals the hand bread at β̂ only up to that step's size (3e-7 here
        # under --check-bounds=yes, whose arithmetic stops IRLS one step earlier). Refitting from
        # a converged start makes the last step ~1e-15
        tight = (rtol = 1e-15, atol = 0.0)
        model0 = glm(@formula(y ~ x + z), data, Binomial(), LogitLink(); tight...)
        model = glm(@formula(y ~ x + z), data, Binomial(), LogitLink(); start = coef(model0), tight...)
        X = modelmatrix(model); b = coef(model); k = size(X, 2)
        jx = findfirst(==("x"), coefnames(model)); ex = zeros(k); ex[jx] = 1.0
        μ = 1 ./ (1 .+ exp.(-(X * b))); μ1 = μ .* (1 .- μ); μ2 = μ1 .* (1 .- 2 .* μ)
        g_ame = vec(mean(μ1 .* ex' .+ b[jx] .* μ2 .* X; dims = 1))     # ∂/∂β mean(β_x μ'(ηᵢ))
        g_aap = vec(mean(μ1 .* X; dims = 1))                             # ∂/∂β mean(μ(ηᵢ))
        xbar = vec(mean(X; dims = 1)); μb = 1 / (1 + exp(-dot(xbar, b)))
        μb1 = μb * (1 - μb); μb2 = μb1 * (1 - 2μb)
        g_mem = μb1 .* ex .+ b[jx] * μb2 .* xbar                         # at the mean row
        g_apm = μb1 .* xbar
        @test only(DataFrame(population_margins(model, data; vars = [:x])).estimate) ≈ b[jx] * mean(μ1) rtol = 1e-12
        for (name, vc) in ((:HC0, HC0()), (:HC1, HC1()), (:CR0, CR0(data.cl)), (:CR1, CR1(data.cl)))
            @testset "$name" begin
                V = _rse_logit_sandwich(model, data.y, name; cluster = data.cl)
                Vcm = CovarianceMatrices.vcov(vc, model)
                if name === :CR1 && !_RSE_CM_CR1_FIXED
                    G = length(unique(data.cl))
                    @test Vcm ≈ V .* (G / (G - 1)) rtol = 1e-10     # the upstream bug Margins corrects
                    Vcm = V
                end
                @test Vcm ≈ V rtol = 1e-10
                se(r) = only(DataFrame(r).se)
                for (r, g) in ((population_margins(model, data; vars = [:x], vcov = vc), g_ame),
                               (population_margins(model, data; type = :predictions, vcov = vc), g_aap),
                               (profile_margins(model, data, means_grid(data); vars = [:x], vcov = vc), g_mem),
                               (profile_margins(model, data, means_grid(data); type = :predictions, vcov = vc), g_apm))
                    @test se(r) ≈ _rse_delta(g, Vcm) rtol = 1e-10    # Margins' delta method with V
                    @test se(r) ≈ _rse_delta(g, V) rtol = 1e-10      # against the hand sandwich
                end
            end
        end
    end
end
