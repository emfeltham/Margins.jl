# test/validation/test_second_differences_mixed_fem.jl
#
# Smoke tests of the model-level `second_differences` on MixedModels (an LMM and a Bernoulli GLMM)
# and FixedEffectModels (including a fit whose estimation sample drops rows), across `at =
# :observed` (the default), `at = :mean`, `levels`, a binary modifier, `groups =` and the
# result-level form. Each runs, gives finite SEs, AD ≈ FD (`backend`), and equals a hand
# computation:
#   - LMM and FEM, y ~ x * z + …: the cross-derivative is the x & z coefficient (estimate and SE)
#     for every form, and the x & t coefficient for the binary modifier t;
#   - GLMM: the AME of x at the shifted modifier from the fixed-effects linear predictor (Margins
#     predicts from the fixed effects), its central difference across ±δ, and the SE by central
#     differences in β with vcov(m).

using Test
using DataFrames, CategoricalArrays, Statistics, Random
using MixedModels, FixedEffectModels, GLM
using Margins

function _sd_mixed_data()
    i = 1:600
    d = DataFrame(x = 0.5 .+ mod.(i .* 37, 101) ./ 40, z = cos.(i .* 0.7) .+ 1,
                  t = Float64.(mod.(i, 2)), g = categorical(["a", "b", "c"][mod.(i .* 5, 3) .+ 1]),
                  subj = "s" .* string.(mod.(i, 31) .+ 1))
    # 31 subjects, coprime with the periods of t and g, so neither is constant within a subject
    # (collinear with a subject fixed effect)
    u = 0.4 .* sin.(mod.(i, 31) .* 1.7)              # subject effect
    d.y = 1 .+ 0.5 .* d.x .+ 0.3 .* d.z .+ 0.4 .* d.x .* d.z .- 0.3 .* d.t .+ 0.2 .* d.x .* d.t .+
          0.3 .* (d.g .== "b") .+ u .+ 0.8 .* sin.(i .* 2.3)
    d.yb = Float64.(sin.(i .* 1.3) .+ 0.6 .* d.x .- 1.4 .+ 0.4 .* d.z .+ 0.5 .* d.x .* (d.z .- 1) .+
                    0.3 .* d.t .+ u .> 0)
    d.tc = categorical(ifelse.(d.t .== 1, "t1", "t0"))   # t as a grouping column (result-level form)
    return d
end

_finite_se(sd) = all(isfinite, sd.se) && all(>(0), sd.se)

# The coefficient of `term` and its SE
function _coef_se(m, term)
    k = findfirst(==(term), coefnames(m))
    return coef(m)[k], sqrt(vcov(m)[k, k])
end

# The result-level second difference across the observed subgroups of t in y ~ x * z + x * t + …:
# AME(x | rows with t = 1) − AME(x | rows with t = 0) = β_xt + β_xz (z̄₁ − z̄₀), with its SE
function _observed_subgroup_sd(m, z, t)
    names_ = coefnames(m)
    c = zeros(length(names_))
    c[findfirst(==("x & t"), names_)] = 1
    c[findfirst(==("x & z"), names_)] = mean(z[t .== 1]) - mean(z[t .== 0])
    return c' * coef(m), sqrt(c' * vcov(m) * c)
end

# Every form of the second difference of x across z (and across the binary t) on model m, each
# checked against AD ≈ FD and returned for the hand comparisons
function _sd_forms(m, d; kw...)
    forms = (
        observed = second_differences(m, d, :x, :z; warn = false, kw...),
        mean     = second_differences(m, d, :x, :z; at = :mean, warn = false, kw...),
        levels   = second_differences(m, d, :x, :z; levels = [0.5, 1.5], warn = false, kw...),
        binary   = second_differences(m, d, :x, :t; warn = false, kw...),
        groups   = second_differences(m, d, :x, :z; groups = :g, warn = false, kw...),
        grouped_binary = second_differences(m, d, :x, :t; groups = :g, warn = false, kw...),
    )
    for (name, sd) in pairs(forms)
        @test nrow(sd) == (name in (:groups, :grouped_binary) ? 3 : 1)
        @test all(isfinite, sd.second_diff)
        @test _finite_se(sd)
    end
    # AD vs FD (the focal derivative); FD's step limits the agreement
    for (name, f) in pairs((observed = (; ), mean = (at = :mean,), levels = (levels = [0.5, 1.5],)))
        ad = forms[name]
        fd = second_differences(m, d, :x, :z; backend = :fd, warn = false, f..., kw...)
        @test fd.second_diff ≈ ad.second_diff rtol = 1e-6
        @test fd.se ≈ ad.se rtol = 1e-6
    end
    return forms
end

@testset "second differences on MixedModels and FixedEffectModels" begin
    d = _sd_mixed_data()

    @testset "LMM: the cross-derivative is the interaction coefficient" begin
        m = fit(MixedModel, @formula(y ~ x * z + x * t + g + (1 | subj)), d; progress = false)
        forms = _sd_forms(m, d)
        b, se = _coef_se(m, "x & z")
        for name in (:observed, :mean, :levels, :groups)
            @test all(forms[name].second_diff .≈ b)
            @test all(isapprox.(forms[name].se, se; rtol = 1e-8))
        end
        bt, set = _coef_se(m, "x & t")
        for name in (:binary, :grouped_binary)
            @test all(isapprox.(forms[name].second_diff, bt; rtol = 1e-10))
            @test all(isapprox.(forms[name].se, set; rtol = 1e-10))
        end
        @test forms.observed.df[1] == Inf     # LMM: z inference
        # result-level form: the effect across the observed subgroups of t (different rows, so the
        # subgroups' different mean z enters through x & z)
        r = population_margins(m, d; vars = [:x], groups = :tc)
        sdr = second_differences(r, :x, :tc)
        est, se_r = _observed_subgroup_sd(m, d.z, d.t)
        @test sdr.second_diff[1] ≈ est rtol = 1e-10
        @test sdr.se[1] ≈ se_r rtol = 1e-10
    end

    @testset "GLMM (Bernoulli): hand computation on the fixed-effects predictor" begin
        m = fit(MixedModel, @formula(yb ~ x * z + t + (1 | subj)), d, Bernoulli(); progress = false)
        forms = _sd_forms(m, d)
        β = coef(m)
        names_ = coefnames(m)
        @test names_ == ["(Intercept)", "x", "z", "t", "x & z"]
        # Fixed-effects design at z shifted by s; AME of x is mean(μ'(η) ∂η/∂x)
        X(s) = hcat(ones(nrow(d)), d.x, d.z .+ s, d.t, d.x .* (d.z .+ s))
        μ′(η) = (p = 1 / (1 + exp(-η)); p * (1 - p))
        ame(b, s) = mean(μ′.(X(s) * b) .* (b[2] .+ b[5] .* (d.z .+ s)))
        δ = forms.observed.delta_used[1]
        @test δ ≈ 0.01 * std(d.z) rtol = 1e-12
        hand(b) = (ame(b, δ) - ame(b, -δ)) / (2δ)
        grad = map(eachindex(β)) do k
            h = 1e-6 * max(abs(β[k]), 1e-2)
            bp = copy(β); bp[k] += h
            bm = copy(β); bm[k] -= h
            (hand(bp) - hand(bm)) / (2h)
        end
        @test forms.observed.second_diff[1] ≈ hand(β) rtol = 1e-9
        @test forms.observed.se[1] ≈ sqrt(grad' * vcov(m) * grad) rtol = 1e-5
        # at = :mean is the same with every row's z set to mean(z) ± δ
        zbar = mean(d.z)
        Xat(v) = hcat(ones(nrow(d)), d.x, fill(v, nrow(d)), d.t, d.x .* v)
        ameat(b, v) = mean(μ′.(Xat(v) * b) .* (b[2] .+ b[5] .* v))
        @test forms.mean.eval_point[1] ≈ zbar
        @test forms.mean.second_diff[1] ≈ (ameat(β, zbar + δ) - ameat(β, zbar - δ)) / (2δ) rtol = 1e-9
        # levels: (AME(z = 1.5) − AME(z = 0.5)) / 1
        @test forms.levels.second_diff[1] ≈ ameat(β, 1.5) - ameat(β, 0.5) rtol = 1e-10
        # groups = :g: each stratum equals the ungrouped call on that stratum's rows (same Σ, same δ)
        for (k, lv) in enumerate(levels(d.g))
            rows = d.g .== lv
            Xg(s) = X(s)[rows, :]
            ameg(b, s) = mean(μ′.(Xg(s) * b) .* (b[2] .+ b[5] .* (d.z[rows] .+ s)))
            row = forms.groups[string.(forms.groups.over_g) .== lv, :]
            @test row.second_diff[1] ≈ (ameg(β, δ) - ameg(β, -δ)) / (2δ) rtol = 1e-9
        end
        @test forms.observed.df[1] == Inf
        # result-level form runs
        sdr = second_differences(population_margins(m, d; vars = [:x], groups = :tc), :x, :tc)
        @test nrow(sdr) == 1 && _finite_se(sdr)
    end

    @testset "FixedEffectModels, with rows outside the estimation sample" begin
        fd = copy(d)
        allowmissing!(fd, :z)
        # two singleton subjects (dropped by drop_singletons) and a missing z
        fd.subj[1] = "single1"; fd.subj[2] = "single2"
        fd.z[3] = missing
        fd.z[1] = 50.0                  # far from the rest: it would move mean(z) if it were used
        m = reg(fd, @formula(y ~ x * z + x * t + g + fe(subj)), Vcov.cluster(:subj); save = :fe)
        @test sum(m.esample) == nrow(fd) - 3
        forms = _sd_forms(m, fd; weights = :none)
        b, se = _coef_se(m, "x & z")
        for name in (:observed, :mean, :levels, :groups)
            @test all(isapprox.(forms[name].second_diff, b; rtol = 1e-7))
            @test all(isapprox.(forms[name].se, se; rtol = 1e-7))
        end
        bt, set = _coef_se(m, "x & t")
        for name in (:binary, :grouped_binary)
            @test all(isapprox.(forms[name].second_diff, bt; rtol = 1e-10))
            @test all(isapprox.(forms[name].se, set; rtol = 1e-10))
        end
        es = m.esample
        @test forms.mean.eval_point[1] ≈ mean(fd.z[es])
        @test forms.observed.delta_used[1] ≈ 0.01 * std(fd.z[es]) rtol = 1e-12
        # FEM: t inference with the cluster-adjusted dof_residual
        @test forms.observed.df[1] == dof_residual(m)
        # groups: the strata hold only estimation-sample rows (n per stratum)
        r = population_margins(m, fd; vars = [:x], groups = :g, weights = :none)
        @test sum(DataFrame(r).n) == sum(es)
        sdr = second_differences(population_margins(m, fd; vars = [:x], groups = :tc, weights = :none), :x, :tc)
        est, se_r = _observed_subgroup_sd(m, fd.z[es], fd.t[es])
        @test sdr.second_diff[1] ≈ est rtol = 1e-10
        @test sdr.se[1] ≈ se_r rtol = 1e-10
    end
end
