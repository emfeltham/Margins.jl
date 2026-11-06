# test_random_effects.jl
# `random_effects = :zero | :conditional | :marginal` for MixedModels fits.
# References are closed forms from MixedModels' own quantities: `fitted(m)` (which includes
# Z b̂), `fixef`, `ranef` / `raneftables` and `vcov`.

using Test, DataFrames, MixedModels, Random, Statistics, Distributions, LinearAlgebra
using CategoricalArrays
using Margins
using Margins: MarginsError
using GLM: lm
import GLM, Tables

# Crossed raters and villages
Random.seed!(20261104)
const N_RE = 800
re_df = let
    rater = rand(string.('a':'l'), N_RE)
    village = rand(["v$i" for i in 1:8], N_RE)
    x = randn(N_RE); w = randn(N_RE)
    ra = Dict(r => 0.7randn() for r in unique(rater))
    rs = Dict(r => 0.4randn() for r in unique(rater))     # rater slopes in w
    va = Dict(v => 0.7randn() for v in unique(village))
    η = -0.2 .+ 0.8 .* x .+ 0.3 .* w .+ [ra[r] for r in rater] .+ [va[v] for v in village]
    y = 0.5 .+ x .+ (0.3 .+ [rs[r] for r in rater]) .* w .+ [ra[r] for r in rater] .+
        [va[v] for v in village] .+ randn(N_RE)
    exposure = rand(N_RE) .+ 0.5
    DataFrame(; rater, village, x, w, y,
              yb = Float64.(rand(N_RE) .< 1 ./ (1 .+ exp.(-η))),
              count = Float64.(rand.(Poisson.(exposure .* exp.(0.3 .+ 0.2 .* x .+ [va[v] for v in village])))),
              logexp = log.(exposure),
              rater_id = [findfirst(==(r), string.('a':'l')) for r in rater])
end

lmm_int = fit(MixedModel, @formula(y ~ x + w + (1 | rater) + (1 | village)), re_df; progress=false)
lmm_slope = fit(MixedModel, @formula(y ~ x + w + (1 + w | rater) + (1 | village)), re_df; progress=false)
glmm = fit(MixedModel, @formula(yb ~ x + w + (1 | rater) + (1 | village)), re_df, Bernoulli();
           progress=false, fast=true)

# b̂ of `level` for factor `f`, coefficient `coef`. For a GLMM, λu from the fit (what `fitted`
# uses): `ranef` reads the working LMM, which can lag (1e-9 after a fast = true probit fit, 1e-2
# after fast = false)
blup(m, f, level, coef="(Intercept)") = (t = getproperty(raneftables(m), f);
    getproperty(t[findfirst(==(level), getproperty(t, f))], Symbol(coef)))
function blup(m::GeneralizedLinearMixedModel, f, level, coef="(Intercept)")
    k = findfirst(r -> string(r.trm) == string(f), m.LMM.reterms)
    r = m.LMM.reterms[k]
    return (Matrix(r.λ) * m.u[k])[findfirst(==(coef), r.cnames), findfirst(==(level), r.levels)]
end

est(r) = DataFrame(r).estimate
se(r) = DataFrame(r).se

@testset "random_effects (MixedModels)" begin
    @testset "LMM: conditional predictions are the fitted values" begin
        for m in (lmm_int, lmm_slope)
            r = population_margins(m, re_df; type=:predictions, random_effects=:conditional)
            @test est(r)[1] ≈ mean(fitted(m)) atol=1e-10
            r0 = population_margins(m, re_df; type=:predictions)
            @test est(r0)[1] ≈ mean(m.X * fixef(m)) atol=1e-10
            # re_se = :fixed: the delta-method SE on β only, the same gradient (x̄) as at zero
            rf = population_margins(m, re_df; type=:predictions, random_effects=:conditional, re_se=:fixed)
            @test est(rf) ≈ est(r) atol=1e-12
            @test se(rf)[1] ≈ se(r0)[1] atol=1e-12
        end
    end

    @testset "LMM: identity link, effects do not change (random intercepts)" begin
        r = population_margins(lmm_int, re_df; vars=[:x, :w], random_effects=:conditional)
        r0 = population_margins(lmm_int, re_df; vars=[:x, :w])
        @test est(r) ≈ est(r0) atol=1e-12
        @test se(r) ≈ se(r0) atol=1e-12
        @test est(r)[1] ≈ fixef(lmm_int)[2] atol=1e-10
    end

    @testset "GLMM: AME and its delta-method SE against the closed form" begin
        β = fixef(glmm); V = vcov(glmm); X = glmm.X
        μ = fitted(glmm)                      # includes Z b̂
        r = population_margins(glmm, re_df; vars=[:x], random_effects=:conditional, re_se=:fixed)
        @test est(r)[1] ≈ mean(β[2] .* μ .* (1 .- μ)) atol=1e-10
        d = μ .* (1 .- μ)
        g = vec(mean(β[2] .* (d .* (1 .- 2 .* μ)) .* X; dims=1))
        g[2] += mean(d)
        @test se(r)[1] ≈ sqrt(g' * V * g) atol=1e-10
        # :zero differs (typical rater and village)
        μ0 = 1 ./ (1 .+ exp.(-(X * β)))
        r0 = population_margins(glmm, re_df; vars=[:x])
        @test est(r0)[1] ≈ mean(β[2] .* μ0 .* (1 .- μ0)) atol=1e-10
        @test abs(est(r)[1] - est(r0)[1]) > 1e-3
        # link scale: η offsets do not change ∂η/∂x
        @test est(population_margins(glmm, re_df; vars=[:x], scale=:link, random_effects=:conditional)) ≈
              [β[2]] atol=1e-12
    end

    @testset "GLMM: predictions, groups, re_terms" begin
        r = population_margins(glmm, re_df; type=:predictions, random_effects=:conditional)
        @test est(r)[1] ≈ mean(fitted(glmm)) atol=1e-10
        # within villages
        rg = population_margins(glmm, re_df; type=:predictions, random_effects=:conditional, groups=:village)
        dg = DataFrame(rg)
        for row in eachrow(dg)
            rows = re_df.village .== row.over_village
            @test row.estimate ≈ mean(fitted(glmm)[rows]) atol=1e-10
        end
        # condition on village only: rater effects 0
        rv = population_margins(glmm, re_df; type=:predictions, random_effects=:conditional, re_terms=:village)
        ηv = glmm.X * fixef(glmm) .+ [blup(glmm, :village, v) for v in re_df.village]
        @test est(rv)[1] ≈ mean(1 ./ (1 .+ exp.(-ηv))) atol=1e-10
        @test rv.metadata[:re_terms] == [:village]
        # an integer grouping column refers to the same levels after Float64 conversion
        gi = fit(MixedModel, @formula(yb ~ x + (1 | rater_id)), re_df, Bernoulli(); progress=false, fast=true)
        @test est(population_margins(gi, re_df; type=:predictions, random_effects=:conditional))[1] ≈
              mean(fitted(gi)) atol=1e-10
    end

    @testset "GLMM fit with fast = false: b̂ of the fitted linear predictor" begin
        # MixedModels' ranef reads the working LMM, which a fast = false fit leaves at an earlier
        # state; Margins uses λu (what fitted(m) uses)
        gf = fit(MixedModel, @formula(yb ~ x + w + (1 + w | rater) + (1 | village)), re_df, Bernoulli();
                 fast=false, progress=false)
        for kw in ((;), (re_se=:fixed,))
            r = population_margins(gf, re_df; type=:predictions, random_effects=:conditional, kw...)
            @test est(r)[1] ≈ mean(fitted(gf)) atol=1e-12
        end
        gp = profile_margins(gf, re_df, DataFrame(x=[0.0], w=[0.0], rater=["c"], village=["v3"]);
                             type=:predictions, random_effects=:conditional)
        bu = Dict(string(r.trm) => (r, Matrix(r.λ) * u) for (r, u) in zip(gf.LMM.reterms, gf.u))
        b_of(f, l) = (p = bu[f]; p[2][1, findfirst(==(l), p[1].levels)])
        @test est(gp)[1] ≈ 1 / (1 + exp(-(fixef(gf)[1] + b_of("rater", "c") + b_of("village", "v3")))) atol=1e-12
    end

    @testset "GLMM with an offset: both enter η" begin
        pm = fit(MixedModel, @formula(count ~ x + (1 | village)), re_df, Poisson(); offset=re_df.logexp,
                 progress=false, fast=true)
        r = population_margins(pm, re_df; type=:predictions, random_effects=:conditional)
        @test est(r)[1] ≈ mean(fitted(pm)) atol=1e-8
    end

    @testset "random slopes: fitted values, effects of other variables, blocks" begin
        # effect of x (no random slope) on the response scale of an identity link: β_x
        r = population_margins(lmm_slope, re_df; vars=[:x], random_effects=:conditional)
        @test est(r)[1] ≈ fixef(lmm_slope)[2] atol=1e-10
        # w has a rater slope: its conditional effect includes each rater's slope b̂_w
        bw = [blup(lmm_slope, :rater, r, "w") for r in re_df.rater]
        r0w = population_margins(lmm_slope, re_df; vars=[:w])
        rw = population_margins(lmm_slope, re_df; vars=[:w], random_effects=:conditional, re_se=:fixed)
        @test est(rw)[1] ≈ fixef(lmm_slope)[3] + mean(bw) atol=1e-8
        @test est(population_margins(lmm_slope, re_df; vars=[:w], random_effects=:conditional)) ≈ est(rw) atol=1e-8
        @test se(rw)[1] ≈ se(r0w)[1] atol=1e-12          # b̂ fixed: the same β-gradient
        # auto-detected variables include it too
        @test length(est(population_margins(lmm_slope, re_df; random_effects=:conditional))) == 2
        # ... and conditioning on the villages only leaves it out
        rw = population_margins(lmm_slope, re_df; vars=[:w], random_effects=:conditional, re_terms=[:village])
        @test est(rw)[1] ≈ fixef(lmm_slope)[3] atol=1e-10
        # GLMM: ∂μ/∂w = μ(1 − μ)(β_w + b̂_w) per row, and its β-gradient
        gs = fit(MixedModel, @formula(yb ~ x + w + (1 + w | rater) + (1 | village)), re_df, Bernoulli();
                 progress=false, fast=true)
        β = fixef(gs); V = vcov(gs); X = gs.X
        μ = fitted(gs); d1 = μ .* (1 .- μ); d2 = d1 .* (1 .- 2 .* μ)
        sl = β[3] .+ [blup(gs, :rater, r, "w") for r in re_df.rater]
        e = population_margins(gs, re_df; vars=[:w], random_effects=:conditional, re_se=:fixed)
        @test est(e)[1] ≈ mean(d1 .* sl) atol=1e-8
        @test est(population_margins(gs, re_df; vars=[:w], random_effects=:conditional)) ≈ est(e) atol=1e-8
        g = vec(mean(d2 .* sl .* X; dims=1)); g[3] += mean(d1)
        @test se(e)[1] ≈ sqrt(g' * V * g) atol=1e-8
        # the finite-difference backend agrees
        @test est(population_margins(gs, re_df; vars=[:w], random_effects=:conditional, re_se=:fixed, backend=:fd))[1] ≈
              est(e)[1] rtol=1e-6
        # a categorical variable with a random slope: under re_se = :fixed too (the wrapper), the
        # :joint point estimate with the β-only SE of its gradient's β block
        dc = copy(re_df); dc.t = categorical(ifelse.(dc.w .> 0, "hi", "lo"))
        mc = fit(MixedModel, @formula(y ~ x + t + (1 + t | rater)), dc; progress=false)
        ctf = population_margins(mc, dc; vars=[:t], random_effects=:conditional, re_se=:fixed)
        ctj = population_margins(mc, dc; vars=[:t], random_effects=:conditional)
        @test est(ctf) ≈ est(ctj) atol=1e-12
        gβ = ctf.gradients[1, 1:length(fixef(mc))]
        @test se(ctf)[1] ≈ sqrt(gβ' * vcov(mc) * gβ) rtol=1e-10
        @test se(ctf)[1] != se(ctj)[1]
        # a user vcov under :fixed through the wrapper: Σ_β is that vcov
        ct2 = population_margins(mc, dc; vars=[:t], random_effects=:conditional, re_se=:fixed, vcov=m -> 4 .* vcov(m))
        @test se(ct2)[1] ≈ 2 * se(ctf)[1] rtol=1e-10
        ctm = population_margins(mc, dc; vars=[:t], random_effects=:conditional, re_se=:fixed, vcov=4 .* vcov(mc))
        @test se(ctm)[1] ≈ 2 * se(ctf)[1] rtol=1e-10
        @test est(population_margins(mc, dc; vars=[:x], random_effects=:conditional))[1] ≈ fixef(mc)[2] atol=1e-10
        # link scale: β_w + mean b̂_w
        el = population_margins(gs, re_df; vars=[:w], scale=:link, random_effects=:conditional)
        @test est(el)[1] ≈ mean(sl) atol=1e-8
        # elasticity: (∂μ/∂w)·w/μ
        ee = population_margins(gs, re_df; vars=[:w], measure=:elasticity, random_effects=:conditional)
        @test est(ee)[1] ≈ mean(d1 .* sl .* re_df.w ./ μ) atol=1e-8
        # at profile rows (village not set: 0)
        gr = DataFrame(x=[0.0, 0.0], w=[0.5, -1.0], rater=["b", "c"])
        pe = profile_margins(gs, re_df, gr; vars=[:w], random_effects=:conditional)
        ηg = [β[1] + β[3] * w + blup(gs, :rater, r) + blup(gs, :rater, r, "w") * w for (w, r) in zip(gr.w, gr.rater)]
        μg = 1 ./ (1 .+ exp.(-ηg))
        @test est(pe) ≈ μg .* (1 .- μg) .* (β[3] .+ [blup(gs, :rater, r, "w") for r in gr.rater]) atol=1e-8
        # setting w counterfactually under re_se = :fixed: the :joint estimates, β-only SEs
        scf = population_margins(lmm_slope, re_df; type=:predictions, scenarios=(w=[0.0, 1.0],),
                                 random_effects=:conditional, re_se=:fixed)
        scj = population_margins(lmm_slope, re_df; type=:predictions, scenarios=(w=[0.0, 1.0],),
                                 random_effects=:conditional)
        @test est(scf) ≈ est(scj) atol=1e-12
        pβ = length(fixef(lmm_slope))
        for i in 1:2
            gβ = scf.gradients[i, 1:pβ]
            @test se(scf)[i] ≈ sqrt(gβ' * vcov(lmm_slope) * gβ) rtol=1e-10
        end
        sdf = second_differences(lmm_slope, re_df, :x, :w; levels=[0.0, 1.0], random_effects=:conditional,
                                 re_se=:fixed, warn=false)
        sdj = second_differences(lmm_slope, re_df, :x, :w; levels=[0.0, 1.0], random_effects=:conditional, warn=false)
        @test sdf.second_diff ≈ sdj.second_diff atol=1e-12
        # ... but a profile grid may set it: z is evaluated at the grid row
        g = DataFrame(x=[0.0, 0.0], w=[1.0, 2.0], rater=["b", "c"])
        p = profile_margins(lmm_slope, re_df, g; type=:predictions, random_effects=:conditional)
        β = fixef(lmm_slope)
        expect = [β[1] + β[3] * w + blup(lmm_slope, :rater, r) + blup(lmm_slope, :rater, r, "w") * w
                  for (w, r) in zip(g.w, g.rater)]
        @test est(p) ≈ expect atol=1e-10
        @test p.metadata[:re_terms] == [:rater]
        @test "rater" in names(DataFrame(p))
    end

    @testset "re_se = :fixed above the wrapper's limit: the offset path" begin
        # 1100 raters with random slopes: 2 × 1101 wrapper columns, above the limit of 2000, so
        # :fixed uses the offset path (β-only gradients) and keeps its restrictions; :joint errors
        rng = Xoshiro(7)
        nb = 1100; nper = 4
        g = repeat(["r$i" for i in 1:nb]; inner=nper)
        xb = randn(rng, nb * nper); wb = randn(rng, nb * nper)
        a0 = randn(rng, nb); a1 = 0.5 .* randn(rng, nb)
        yb = 1 .+ xb .+ (0.5 .+ repeat(a1; inner=nper)) .* wb .+ repeat(a0; inner=nper) .+ randn(rng, nb * nper)
        big = DataFrame(; g, x=xb, w=wb, y=yb)
        mb = fit(MixedModel, @formula(y ~ x + w + (1 + w | g)), big; progress=false)
        @test !Margins._re_wrap_fixed(mb, [:g])
        r = population_margins(mb, big; type=:predictions, random_effects=:conditional, re_se=:fixed)
        @test est(r)[1] ≈ mean(fitted(mb)) atol=1e-10
        @test size(r.gradients, 2) == length(fixef(mb))
        @test se(r)[1] ≈ sqrt(vec(mean(mb.X; dims=1))' * vcov(mb) * vec(mean(mb.X; dims=1))) rtol=1e-10
        @test_throws MarginsError population_margins(mb, big; type=:predictions, scenarios=(w=[0.0, 1.0],),
                                                     random_effects=:conditional, re_se=:fixed)
        @test_throws ArgumentError population_margins(mb, big; type=:predictions, random_effects=:conditional)
        # the limit counts every factor, and its message does not suggest re_terms
        msg = try population_margins(mb, big; type=:predictions, random_effects=:conditional); "" catch e; sprint(showerror, e) end
        @test occursin("every factor", msg) && !occursin("re_terms", msg)
        # a mixture in the grouping column on the offset path: the share-weighted b̂, and in the
        # effect of the slope variable the share-weighted slope
        gm = DataFrame(x=[0.0], w=[1.0], g=[mix("r1" => 0.25, "r2" => 0.75)])
        pm = profile_margins(mb, big, gm; type=:predictions, random_effects=:conditional, re_se=:fixed)
        bmix(c) = 0.25 * blup(mb, :g, "r1", c) + 0.75 * blup(mb, :g, "r2", c)
        @test est(pm)[1] ≈ fixef(mb)[1] + fixef(mb)[3] + bmix("(Intercept)") + bmix("w") atol=1e-10
        @test pm.metadata[:re_terms] == [:g]
        em = profile_margins(mb, big, gm; vars=[:w], random_effects=:conditional, re_se=:fixed)
        @test est(em)[1] ≈ fixef(mb)[3] + bmix("w") atol=1e-8
        # below the limit the wrapper is used: gradients over (β, b)
        @test Margins._re_wrap_fixed(lmm_slope, [:rater, :village])
        rs = population_margins(lmm_slope, re_df; type=:predictions, random_effects=:conditional, re_se=:fixed)
        @test size(rs.gradients, 2) > length(fixef(lmm_slope))
        @test se(rs)[1] ≈ sqrt(vec(mean(lmm_slope.X; dims=1))' * vcov(lmm_slope) * vec(mean(lmm_slope.X; dims=1))) rtol=1e-10
    end

    @testset "the (β, b) wrapper across formula forms, weights, REML, singular and rank-deficient fits" begin
        jwrap(m, d=re_df) = (spec = Margins._validate_random_effects(m, :conditional, :zero, nothing, :auto, :auto);
                             Margins._re_wrapper_model(m, Tables.columntable(d), spec, GLM.vcov)[1])
        function wrapper_ok(m, d=re_df; wts=nothing)
            μ̄ = isnothing(wts) ? mean(fitted(m)) : sum(wts .* fitted(m)) / sum(wts)
            rj = population_margins(m, d; type=:predictions, random_effects=:conditional)
            rf = population_margins(m, d; type=:predictions, random_effects=:conditional, re_se=:fixed)
            w = jwrap(m, d)
            @test est(rj)[1] ≈ μ̄ atol=1e-10
            @test est(rf) ≈ est(rj) atol=1e-12
            @test w.C[1:w.p, 1:w.p] ≈ vcov(m) rtol=1e-8
        end
        dd = copy(re_df); dd.wt = 0.5 .+ rand(Xoshiro(3), nrow(dd))
        # zerocorr, and a slope amalgamated with the intercept term of the same factor
        # (`(0 + w | rater)` has no intercept column)
        wrapper_ok(fit(MixedModel, @formula(y ~ x + w + zerocorr(1 + w | rater) + (1 | village)), re_df; progress=false))
        am = fit(MixedModel, @formula(y ~ x + w + (1 | rater) + (0 + w | rater) + (1 | village)), re_df; progress=false)
        wrapper_ok(am)
        @test est(population_margins(am, re_df; vars=[:w], random_effects=:conditional))[1] ≈
              fixef(am)[3] + mean(blup(am, :rater, r, "w") for r in re_df.rater) atol=1e-8
        @test est(population_margins(am, re_df; vars=[:w], random_effects=:conditional, re_se=:fixed)) ≈
              est(population_margins(am, re_df; vars=[:w], random_effects=:conditional)) atol=1e-12
        wrapper_ok(fit(MixedModel, @formula(y ~ x + w + (0 + w | rater) + (1 | village)), re_df; progress=false))
        # REML
        wrapper_ok(fit(MixedModel, @formula(y ~ x + w + (1 + w | rater) + (1 | village)), re_df; REML=true, progress=false))
        # a GLMM with an offset
        wrapper_ok(fit(MixedModel, @formula(count ~ x + (1 | village)), re_df, Poisson(); offset=re_df.logexp,
                       progress=false, fast=true))
        # a singular fit (no village variance in the data): its b̂ and their variance are 0
        dn = copy(re_df); dn.yn = 1 .+ dn.x .+ randn(Xoshiro(11), nrow(dn)) .+ [blup(lmm_int, :rater, r) for r in dn.rater]
        sg = fit(MixedModel, @formula(yn ~ x + (1 | rater) + (1 | village)), dn; progress=false)
        @test issingular(sg)
        wrapper_ok(sg, dn)
        # :marginal integrates over the rater effects only (the village variance is 0); identity link: = :zero
        @test est(population_margins(sg, dn; type=:predictions, random_effects=:marginal)) ≈
              est(population_margins(sg, dn; type=:predictions)) atol=1e-12
        # a categorical random slope under :marginal (probit): τᵢ² = zᵢ'Σzᵢ with zᵢ = (1, [t = lo])
        dc = copy(re_df); dc.t = categorical(ifelse.(dc.w .> 0, "hi", "lo"))
        mp = fit(MixedModel, @formula(yb ~ x + t + (1 + t | rater)), dc, Bernoulli(), ProbitLink(); progress=false, fast=true)
        S = Matrix(mp.reterms[1].λ * mp.reterms[1].λ')
        τ² = [z' * S * z for z in eachrow([ones(nrow(dc)) Float64.(dc.t .== "lo")])]
        @test est(population_margins(mp, dc; type=:predictions, random_effects=:marginal))[1] ≈
              mean(cdf.(Normal(), (mp.X * fixef(mp)) ./ sqrt.(1 .+ τ²))) atol=1e-10
        # rank-deficient: :fixed through the wrapper; :joint errors
        dr = copy(re_df); dr.x2 = 2 .* dr.x
        rd = fit(MixedModel, @formula(y ~ x + x2 + w + (1 | rater)), dr; progress=false)
        rr = population_margins(rd, dr; type=:predictions, random_effects=:conditional, re_se=:fixed, warn=false)
        @test est(rr)[1] ≈ mean(fitted(rd)) atol=1e-10
        @test se(rr)[1] ≈ se(population_margins(rd, dr; type=:predictions, warn=false))[1] rtol=1e-10
        @test_throws ArgumentError population_margins(rd, dr; type=:predictions, random_effects=:conditional, warn=false)

        # weighted LMM: Henderson's equations with W, and the weighted average of the default
        # weights = :model
        lw = fit(MixedModel, @formula(y ~ x + w + (1 | rater) + (1 | village)), dd; wts=dd.wt, progress=false)
        wrapper_ok(lw, dd; wts=dd.wt)
        raters = sort(unique(dd.rater)); villages = sort(unique(dd.village))
        Z = [Float64.(dd.rater .== permutedims(raters)) Float64.(dd.village .== permutedims(villages))]
        X = lw.X; W = Diagonal(dd.wt); σ = lw.σ
        Ginv = Diagonal([fill(1 / lw.σs.rater[1]^2, length(raters)); fill(1 / lw.σs.village[1]^2, length(villages))])
        Ch = σ^2 * inv([X'W*X X'W*Z; Z'W*X Z'W*Z + σ^2 * Ginv])
        @test Ch[1:3, 1:3] ≈ vcov(lw) rtol=1e-6
        g = vec(sum(dd.wt .* [X Z]; dims=1)) ./ sum(dd.wt)
        @test se(population_margins(lw, dd; type=:predictions, random_effects=:conditional))[1] ≈
              sqrt(g' * Ch * g) rtol=1e-6
        # weighted GLMM
        gw = fit(MixedModel, @formula(yb ~ x + (1 | rater) + (1 | village)), dd, Bernoulli(); wts=dd.wt,
                 progress=false, fast=true)
        wrapper_ok(gw, dd; wts=dd.wt)
    end

    @testset "nested factors" begin
        m = fit(MixedModel, @formula(y ~ x + (1 | village / rater)), re_df; progress=false)
        r = population_margins(m, re_df; type=:predictions, random_effects=:conditional)
        @test est(r)[1] ≈ mean(fitted(m)) atol=1e-10
        @test Set(r.metadata[:re_terms]) == Set([:village, Symbol("village & rater")])
    end

    @testset "profile grids: levels set, levels unset, new levels" begin
        β = fixef(glmm)
        g = DataFrame(x=[0.0, 0.0], w=[0.0, 0.0], village=["v1", "v2"])
        p = profile_margins(glmm, re_df, g; type=:predictions, random_effects=:conditional)
        @test est(p) ≈ [1 / (1 + exp(-(β[1] + blup(glmm, :village, v)))) for v in g.village] atol=1e-10
        @test p.metadata[:re_terms] == [:village]          # rater not set: 0
        # a grid without grouping columns gives the :zero estimates
        g0 = DataFrame(x=[0.0, 1.0])
        @test est(profile_margins(glmm, re_df, g0; type=:predictions, random_effects=:conditional)) ≈
              est(profile_margins(glmm, re_df, g0; type=:predictions)) atol=1e-12
        # a level absent from the fit
        gn = DataFrame(x=[0.0], w=[0.0], village=["vNEW"], rater=["a"])
        pn = profile_margins(glmm, re_df, gn; type=:predictions, random_effects=:conditional)
        @test est(pn)[1] ≈ 1 / (1 + exp(-(β[1] + blup(glmm, :rater, "a")))) atol=1e-10
        @test_throws ArgumentError profile_margins(glmm, re_df, gn; type=:predictions,
                                                   random_effects=:conditional, re_new=:error)
        # effects at grid rows: the response-scale effect uses μ at the row's η
        pe = profile_margins(glmm, re_df, g; vars=[:x], random_effects=:conditional)
        μg = [1 / (1 + exp(-(β[1] + blup(glmm, :village, v)))) for v in g.village]
        @test est(pe) ≈ β[2] .* μg .* (1 .- μg) atol=1e-10
        # under :zero a varying grouping column is still rejected (it would change nothing)
        @test_throws ArgumentError profile_margins(glmm, re_df, g; type=:predictions)
    end

    @testset "second differences" begin
        # AME of x at w = 1 minus at w = 0, each row keeping its own rater and village effects
        β = fixef(glmm)
        zb = log.(fitted(glmm) ./ (1 .- fitted(glmm))) .- glmm.X * β
        ame_at(wv) = (η = β[1] .+ β[2] .* re_df.x .+ β[3] * wv .+ zb; μ = 1 ./ (1 .+ exp.(-η));
                      mean(β[2] .* μ .* (1 .- μ)))
        sd = second_differences(glmm, re_df, :x, :w; levels=[0.0, 1.0], random_effects=:conditional, warn=false)
        @test sd.second_diff[1] ≈ ame_at(1.0) - ame_at(0.0) atol=1e-10
        sd0 = second_differences(glmm, re_df, :x, :w; levels=[0.0, 1.0], warn=false)
        @test abs(sd.second_diff[1] - sd0.second_diff[1]) > 1e-4
    end

    @testset "metadata and printing" begin
        r = population_margins(glmm, re_df; vars=[:x], random_effects=:conditional)
        @test r.metadata[:random_effects] === :conditional
        @test r.metadata[:re_terms] == [:rater, :village] || r.metadata[:re_terms] == [:village, :rater]
        @test occursin("Random effects: predicted for", sprint(show, r))
        r0 = population_margins(glmm, re_df; vars=[:x])
        @test r0.metadata[:random_effects] === :zero && isempty(r0.metadata[:re_terms])
        @test occursin("(a typical group)", sprint(show, r0))
        rv = population_margins(glmm, re_df; vars=[:x], random_effects=:conditional, re_terms=:village)
        @test occursin("predicted for village; 0 for rater", sprint(show, rv))
        rl = population_margins(lm(@formula(y ~ x), re_df), re_df; vars=[:x])
        @test rl.metadata[:random_effects] === :zero && !haskey(rl.metadata, :re_factors)
        @test !occursin("Random effects", sprint(show, rl))
    end

    @testset ":conditional joint SEs: the covariance of (β̂, b̂ − b)" begin
        E = Base.get_extension(Margins, :MixedModelsExt)
        # (1) LMM: Henderson's mixed-model equations built from scratch (G from VarCorr, Z from the
        # data), against the wrapper's covariance through the SE of the conditional AAP
        m = lmm_int
        σ = m.σ
        raters = sort(unique(re_df.rater)); villages = sort(unique(re_df.village))
        Zr = Float64.(re_df.rater .== permutedims(raters)); Zv = Float64.(re_df.village .== permutedims(villages))
        Z = [Zr Zv]; X = m.X
        Ginv = Diagonal([fill(1 / m.σs.rater[1]^2, length(raters)); fill(1 / m.σs.village[1]^2, length(villages))])
        M = [X'X X'Z; Z'X Z'Z + σ^2 * Ginv]
        Ch = σ^2 * inv(M)                                  # Cov((β̂, b̂) − (β, b)), Henderson
        @test Ch[1:3, 1:3] ≈ vcov(m) rtol=1e-8
        r = population_margins(m, re_df; type=:predictions, random_effects=:conditional)
        g = [vec(mean(X; dims=1)); vec(mean(Z; dims=1))]
        @test se(r)[1] ≈ sqrt(g' * Ch * g) rtol=1e-8
        @test r.metadata[:re_se] === :joint
        # AME of x on the identity scale: ∂/∂b = 0, so the β-only SE
        ex = population_margins(m, re_df; vars=[:x], random_effects=:conditional)
        @test se(ex)[1] ≈ sqrt(vcov(m)[2, 2]) rtol=1e-10
        # predictions at a profile row for rater "c", village "v3": x'β + b_c + b_v3
        gr = DataFrame(x=[0.5], w=[0.0], rater=["c"], village=["v3"])
        p = profile_margins(m, re_df, gr; type=:predictions, random_effects=:conditional)
        gp = [1.0; 0.5; 0.0; Float64.(raters .== "c"); Float64.(villages .== "v3")]
        @test se(p)[1] ≈ sqrt(gp' * Ch * gp) rtol=1e-8
        # a contrast between two result rows uses the joint covariance too
        gr2 = DataFrame(x=[0.0, 0.0], w=[0.0, 0.0], rater=["c", "d"])
        p2 = profile_margins(m, re_df, gr2; type=:predictions, random_effects=:conditional)
        gd = [zeros(3); Float64.(raters .== "d") .- Float64.(raters .== "c"); zeros(length(villages))]
        @test contrast(p2, 2, 1).se ≈ sqrt(gd' * Ch * gd) rtol=1e-8

        # (2) conditioning the joint covariance on β gives MixedModels' condVar, Var(b − b̂ | β):
        # C_bb − C_bβ C_ββ⁻¹ C_βb (LMM, random slopes: 2 × 2 blocks per rater)
        specl = Margins._validate_random_effects(lmm_slope, :conditional, :zero, nothing, :auto, :auto)
        (wl, _) = Margins._re_wrapper_model(lmm_slope, Tables.columntable(re_df), specl, GLM.vcov)
        Cl = wl.C; pl = wl.p
        @test Cl[1:pl, 1:pl] ≈ vcov(lmm_slope) rtol=1e-8
        S = Cl[pl+1:end, pl+1:end] .- Cl[pl+1:end, 1:pl] * (Cl[1:pl, 1:pl] \ Cl[1:pl, pl+1:end])
        cv = MixedModels.condVar(lmm_slope)
        cnl = coefnames(wl)[pl+1:end]
        kr = findfirst(r -> string(r.trm) == "rater", lmm_slope.reterms)
        for (j, l) in enumerate(lmm_slope.reterms[kr].levels)
            i0 = findfirst(==("__re_rater: $(l)"), cnl); i1 = findfirst(==("__re_rater: $(l) & w"), cnl)
            @test S[[i0, i1], [i0, i1]] ≈ cv[kr][:, :, j] rtol=1e-6
        end
        # GLMM: the β block is vcov(m)
        spec = Margins._validate_random_effects(glmm, :conditional, :zero, nothing, :auto, :auto)
        (w, _) = Margins._re_wrapper_model(glmm, Tables.columntable(re_df), spec, GLM.vcov)
        C = w.C; p_ = w.p
        @test C[1:p_, 1:p_] ≈ vcov(glmm) rtol=1e-10
        # the AME's joint gradient: ∂/∂b of mean(β_x μ(1 − μ)) = mean(β_x μ(1 − μ)(1 − 2μ) zᵢ)
        β = fixef(glmm); μ = fitted(glmm); d2 = μ .* (1 .- μ) .* (1 .- 2 .* μ)
        e = population_margins(glmm, re_df; vars=[:x], random_effects=:conditional)
        gβ = vec(mean(β[2] .* d2 .* glmm.X; dims=1)); gβ[2] += mean(μ .* (1 .- μ))
        Zw = hcat([Float64.(E._hidden_column(w, k, Tables.columntable(re_df), nrow(re_df), :zero) .== l)
                   for k in 1:2 for l in Margins.CategoricalArrays.levels(E._hidden_column(w, k, Tables.columntable(re_df), nrow(re_df), :zero))]...)
        gb = vec(mean(β[2] .* d2 .* Zw; dims=1))
        gj = [gβ; gb]
        @test se(e)[1] ≈ sqrt(gj' * C * gj) rtol=1e-8
        @test se(e)[1] != se(population_margins(glmm, re_df; vars=[:x], random_effects=:conditional, re_se=:fixed))[1]

        # (3) on the joint path z is part of the compiled formula, so categorical random slopes,
        # scenarios on slope variables and second differences over them are supported
        dc = copy(re_df); dc.t = categorical(ifelse.(dc.w .> 0, "hi", "lo"))
        mc = fit(MixedModel, @formula(y ~ x + t + (1 + t | rater)), dc; progress=false)
        ct = population_margins(mc, dc; vars=[:t], random_effects=:conditional)
        bt = [blup(mc, :rater, r, "t: lo") for r in dc.rater]
        @test est(ct)[1] ≈ fixef(mc)[3] + mean(bt) atol=1e-8
        sc = population_margins(lmm_slope, re_df; type=:predictions, scenarios=(w=[0.0, 1.0],),
                                random_effects=:conditional)
        βs = fixef(lmm_slope)
        pred_at(wv) = mean(βs[1] .+ βs[2] .* re_df.x .+ βs[3] * wv .+ [blup(lmm_slope, :rater, r) + blup(lmm_slope, :rater, r, "w") * wv
                                                                  for r in re_df.rater] .+ [blup(lmm_slope, :village, v) for v in re_df.village])
        @test est(sc) ≈ [pred_at(0.0), pred_at(1.0)] atol=1e-8
        @test nrow(second_differences(lmm_slope, re_df, :x, :w; levels=[0.0, 1.0], random_effects=:conditional, warn=false)) == 1

        # (4) the options
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, vcov=m -> vcov(m))
        @test population_margins(glmm, re_df; vars=[:x], random_effects=:conditional, re_se=:fixed,
                                 vcov=m -> 2 .* vcov(m)) isa EffectsResult
        @test_throws ArgumentError population_margins(glmm, re_df; re_se=:joint)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:marginal, re_se=:fixed)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_se=:both)
        # no hidden column leaks into results or auto-detected variables
        ra = population_margins(glmm, re_df; random_effects=:conditional)
        @test Set(DataFrame(ra).variable) == Set(["x", "w"])
        @test !any(startswith("__re_"), names(DataFrame(p)))
    end

    @testset ":marginal, probit: closed form E[Φ(η + τZ)] = Φ(η/√(1 + τ²))" begin
        Random.seed!(7)
        d = copy(re_df)
        d.t = categorical(rand(["lo", "hi"], nrow(d)))
        ra = Dict(r => randn() for r in unique(d.rater)); va = Dict(v => 0.8randn() for v in unique(d.village))
        ηt = -0.2 .+ 0.8 .* d.x .+ 0.4 .* (d.t .== "hi") .+ [ra[r] for r in d.rater] .+ [va[v] for v in d.village]
        d.yp = Float64.(rand(nrow(d)) .< cdf.(Normal(), ηt))
        m = fit(MixedModel, @formula(yp ~ x + t + (1 | rater) + (1 | village)), d, Bernoulli(), ProbitLink();
                progress=false, fast=true)
        β = fixef(m); V = vcov(m); X = m.X
        τ² = m.σs.rater[1]^2 + m.σs.village[1]^2
        c = sqrt(1 + τ²)
        φ(t) = pdf(Normal(), t); Φ(t) = cdf(Normal(), t)
        η = X * β
        # AAP and its SE
        r = population_margins(m, d; type=:predictions, random_effects=:marginal)
        @test est(r)[1] ≈ mean(Φ.(η ./ c)) atol=1e-12
        g = vec(mean(φ.(η ./ c) ./ c .* X; dims=1))
        @test se(r)[1] ≈ sqrt(g' * V * g) atol=1e-12
        # AME of x and its SE
        ix = findfirst(==("x"), coefnames(m))
        e = population_margins(m, d; vars=[:x], random_effects=:marginal)
        @test est(e)[1] ≈ mean(β[ix] / c .* φ.(η ./ c)) atol=1e-12
        ge = vec(mean(β[ix] / c .* (-(η ./ c) .* φ.(η ./ c)) ./ c .* X; dims=1))
        ge[ix] += mean(φ.(η ./ c)) / c
        @test se(e)[1] ≈ sqrt(ge' * V * ge) atol=1e-12
        # categorical contrast hi − lo
        it = findfirst(==("t: lo"), coefnames(m))
        X1 = copy(X); X0 = copy(X); X1[:, it] .= 1; X0[:, it] .= 0   # "hi" is the base level
        ct = population_margins(m, d; vars=[:t], random_effects=:marginal)
        @test est(ct)[1] ≈ mean(Φ.((X1 * β) ./ c) .- Φ.((X0 * β) ./ c)) atol=1e-12
        # profile predictions at grid rows
        gr = DataFrame(x=[-1.0, 0.0, 1.0], t=["hi", "hi", "lo"])
        p = profile_margins(m, d, gr; type=:predictions, random_effects=:marginal)
        ηg = [β[1] + β[ix] * x + (t == "lo" ? β[it] : 0.0) for (x, t) in zip(gr.x, gr.t)]
        @test est(p) ≈ Φ.(ηg ./ c) atol=1e-12
        # second difference of the AME of x between t levels
        sd = second_differences(m, d, :x, :t; random_effects=:marginal, warn=false)
        ame_t(lv) = (Xt = lv == "lo" ? X1 : X0; ηt = Xt * β; mean(β[ix] / c .* φ.(ηt ./ c)))
        @test abs(sd.second_diff[1]) ≈ abs(ame_t("lo") - ame_t("hi")) atol=1e-12
        # integrating over the villages only: τ² = σ_village²
        cv = sqrt(1 + m.σs.village[1]^2)
        rv = population_margins(m, d; type=:predictions, random_effects=:marginal, re_terms=:village)
        @test est(rv)[1] ≈ mean(Φ.(η ./ cv)) atol=1e-12
        # metadata and printing
        @test r.metadata[:random_effects] === :marginal
        @test Set(r.metadata[:re_terms]) == Set([:rater, :village])
        @test r.metadata[:re_sd] ≈ sqrt(τ²)
        @test r.metadata[:re_nodes] >= 16
        @test occursin("averaged over", sprint(show, r))
        @test occursin("averaged over village", sprint(show, rv)) && occursin("0 for rater", sprint(show, rv))
        # the link scale is unchanged
        @test est(population_margins(m, d; vars=[:x], scale=:link, random_effects=:marginal)) ≈ [β[ix]] atol=1e-12
    end

    @testset ":marginal, probit with random slopes: τᵢ² = zᵢ'Σzᵢ per row" begin
        Random.seed!(11)
        d = copy(re_df)
        rs = Dict(r => (randn(), 0.6randn()) for r in unique(d.rater))
        va = Dict(v => 0.5randn() for v in unique(d.village))
        ηt = [-0.1 + 0.7x + (0.3 + rs[r][2]) * w + rs[r][1] + va[v]
              for (x, w, r, v) in zip(d.x, d.w, d.rater, d.village)]
        d.yp = Float64.(rand(nrow(d)) .< cdf.(Normal(), ηt))
        m = fit(MixedModel, @formula(yp ~ x + w + (1 + w | rater) + (1 | village)), d, Bernoulli(),
                ProbitLink(); progress=false, fast=true)
        β = fixef(m); V = vcov(m); X = m.X
        Λr = only(r.λ for r in m.reterms if string(r.trm) == "rater")
        Σr = Matrix(Λr) * Matrix(Λr)'
        σv² = m.σs.village[1]^2
        τ²(w) = Σr[1, 1] + 2w * Σr[1, 2] + w^2 * Σr[2, 2] + σv²
        c = sqrt.(1 .+ τ².(d.w))
        φ(t) = pdf(Normal(), t); Φ(t) = cdf(Normal(), t)
        η = X * β
        r = population_margins(m, d; type=:predictions, random_effects=:marginal)
        @test est(r)[1] ≈ mean(Φ.(η ./ c)) atol=1e-12
        g = vec(mean(φ.(η ./ c) ./ c .* X; dims=1))
        @test se(r)[1] ≈ sqrt(g' * V * g) atol=1e-12
        # effect of x (no random slope)
        e = population_margins(m, d; vars=[:x], random_effects=:marginal)
        @test est(e)[1] ≈ mean(β[2] ./ c .* φ.(η ./ c)) atol=1e-12
        # profile rows: τ at the grid's w
        gr = DataFrame(x=[0.0, 0.0, 1.0], w=[-1.0, 1.0, 2.0])
        p = profile_margins(m, d, gr; type=:predictions, random_effects=:marginal)
        @test est(p) ≈ [Φ((β[1] + β[2]x + β[3]w) / sqrt(1 + τ²(w))) for (x, w) in zip(gr.x, gr.w)] atol=1e-12
        @test p.metadata[:re_sd] ≈ sqrt(maximum(τ².(gr.w)))
        # the effect of w: μ̄ = Φ(η/c(w)), c = √(1 + τ²(w)), so
        # ∂μ̄/∂w = φ(t)(β_w/c − η c′/c²), t = η/c, c′ = (Σ₀₁ + wΣ₁₁)/c
        cp = (Σr[1, 2] .+ d.w .* Σr[2, 2]) ./ c
        t = η ./ c
        ew = population_margins(m, d; vars=[:w], random_effects=:marginal)
        dw = φ.(t) .* (β[3] ./ c .- η .* cp ./ c .^ 2)
        @test est(ew)[1] ≈ mean(dw) atol=1e-12          # ∂τ/∂w from the Jacobian: exact
        gw = vec(mean((-t .* φ.(t)) ./ c .* (β[3] ./ c .- η .* cp ./ c .^ 2) .* X .-
                      φ.(t) .* cp ./ c .^ 2 .* X; dims=1))
        gw[3] += mean(φ.(t) ./ c)
        @test se(ew)[1] ≈ sqrt(gw' * V * gw) atol=1e-12
        @test est(population_margins(m, d; vars=[:w], random_effects=:marginal, backend=:fd))[1] ≈ mean(dw) rtol=1e-6
        # at profile rows
        pw = profile_margins(m, d, gr; vars=[:w], random_effects=:marginal)
        ηp = [β[1] + β[2]x + β[3]w for (x, w) in zip(gr.x, gr.w)]
        cg = sqrt.(1 .+ τ².(gr.w)); cgp = (Σr[1, 2] .+ gr.w .* Σr[2, 2]) ./ cg
        @test est(pw) ≈ φ.(ηp ./ cg) .* (β[3] ./ cg .- ηp .* cgp ./ cg .^ 2) atol=1e-12
        # population scenarios on w: τ follows the counterfactual w (z is in the compiled row)
        μ̄at(wv, b=β) = mean(Φ.((b[1] .+ b[2] .* d.x .+ b[3] * wv) ./ sqrt(1 + τ²(wv))))
        sc = population_margins(m, d; type=:predictions, scenarios=(w=[0.0, 2.5],), random_effects=:marginal)
        @test est(sc) ≈ [μ̄at(0.0), μ̄at(2.5)] atol=1e-12
        c25 = sqrt(1 + τ²(2.5)); η25 = β[1] .+ β[2] .* d.x .+ β[3] * 2.5
        g25 = vec(mean(φ.(η25 ./ c25) ./ c25 .* [ones(nrow(d)) d.x fill(2.5, nrow(d))]; dims=1))
        @test se(sc)[2] ≈ sqrt(g25' * V * g25) atol=1e-12
        @test size(sc.gradients, 2) == length(β) + 1          # w's column of z, coefficient 0
        @test all(iszero, sc.inference.vcov[end, :])
        # second differences of the x effect over w (evaluated at counterfactual w): closed form
        amex(wv) = mean(β[2] / sqrt(1 + τ²(wv)) .* φ.((β[1] .+ β[2] .* d.x .+ β[3] * wv) ./ sqrt(1 + τ²(wv))))
        sd = second_differences(m, d, :x, :w; levels=[0.0, 3.0], random_effects=:marginal, warn=false)
        @test sd.second_diff[1] ≈ (amex(3.0) - amex(0.0)) / 3 atol=1e-12
        # ... including points above every observed τ: the rule is chosen again for them
        wbig = maximum(abs, d.w) + 2
        sdb = second_differences(m, d, :x, :w; levels=[0.0, wbig], random_effects=:marginal, warn=false)
        @test sdb.second_diff[1] ≈ (amex(wbig) - amex(0.0)) / wbig atol=1e-10
        @test nrow(second_differences(m, d, :x, :w; at=:observed, random_effects=:marginal, warn=false)) == 1
        # a row above the τ the rule was checked at is an error, never a silent inaccuracy
        o = Margins.RowSpreadOffset(nothing, 1, [0, 3], [1.0 0.0; 0.0 1.0], 1.0)
        @test_throws MarginsError Margins._row_tau(Margins.RowSpreadOffset(nothing, 1, [0, 3], [1.0 0.0; 0.0 1.0], 1.0),
                                                   [1.0, 0.0, 0.5])
        @test Margins._row_tau(Margins.RowSpreadOffset(nothing, 1, [0, 3], [1.0 0.0; 0.0 1.0], 2.0), [1.0, 0.0, 0.5]) ≈ sqrt(1.25)

        # a categorical random slope: its contrast evaluates τ at each level
        dc = copy(d); dc.t = categorical(ifelse.(dc.w .> 0, "hi", "lo"))
        mc = fit(MixedModel, @formula(yp ~ x + t + (1 + t | rater)), dc, Bernoulli(), ProbitLink();
                 progress=false, fast=true)
        βc = fixef(mc); Sc = Matrix(mc.reterms[1].λ * mc.reterms[1].λ')
        cl(lo) = sqrt(1 + Sc[1, 1] + lo * (2Sc[1, 2] + Sc[2, 2]))
        pt(lo) = Φ.((βc[1] .+ βc[2] .* dc.x .+ βc[3] * lo) ./ cl(lo))
        ct = population_margins(mc, dc; vars=[:t], random_effects=:marginal)
        @test est(ct)[1] ≈ mean(pt(1.0) .- pt(0.0)) atol=1e-12
        gt(lo) = vec(mean(φ.((βc[1] .+ βc[2] .* dc.x .+ βc[3] * lo) ./ cl(lo)) ./ cl(lo) .*
                          [ones(nrow(dc)) dc.x fill(lo, nrow(dc))]; dims=1))
        gct = gt(1.0) .- gt(0.0)
        @test se(ct)[1] ≈ sqrt(gct' * vcov(mc) * gct) atol=1e-12
        # a slope term nonlinear in its variable, w2 in z only: ∂τ/∂w2 from the Jacobian of w2²
        dq = copy(d); dq.w2 = dq.w
        mq = fit(MixedModel, @formula(yp ~ x + w + (1 + w2^2 | rater)), dq, Bernoulli(), ProbitLink();
                 progress=false, fast=true)
        βq = fixef(mq); Sq = Matrix(mq.reterms[1].λ * mq.reterms[1].λ')
        cq = sqrt.(1 .+ Sq[1, 1] .+ 2Sq[1, 2] .* dq.w2 .^ 2 .+ Sq[2, 2] .* dq.w2 .^ 4)
        cq′ = (2Sq[1, 2] .* dq.w2 .+ 2Sq[2, 2] .* dq.w2 .^ 3) ./ cq     # ∂c/∂w2
        ηq = mq.X * βq
        eq = population_margins(mq, dq; vars=[:w2], random_effects=:marginal)
        @test est(eq)[1] ≈ mean(-φ.(ηq ./ cq) .* ηq .* cq′ ./ cq .^ 2) atol=1e-12
        # ... unless the rater factor is not integrated over
        rv = population_margins(m, d; vars=[:w], random_effects=:marginal, re_terms=:village)
        cv = sqrt(1 + σv²)
        @test est(rv)[1] ≈ mean(β[3] / cv .* φ.(η ./ cv)) atol=1e-12
    end

    @testset ":marginal, log link: E[exp(η + τZ)] = exp(η + τ²/2)" begin
        pm = fit(MixedModel, @formula(count ~ x + (1 | village)), re_df, Poisson(); offset=re_df.logexp,
                 progress=false, fast=true)
        τ² = pm.σs.village[1]^2
        η = pm.X * fixef(pm) .+ re_df.logexp
        r = population_margins(pm, re_df; type=:predictions, random_effects=:marginal)
        @test est(r)[1] ≈ mean(exp.(η .+ τ² / 2)) rtol=1e-11
        e = population_margins(pm, re_df; vars=[:x], random_effects=:marginal)
        @test est(e)[1] ≈ fixef(pm)[2] * mean(exp.(η .+ τ² / 2)) rtol=1e-11
    end

    @testset ":marginal, logit: the automatic rule against a 200-node rule" begin
        r = population_margins(glmm, re_df; vars=[:x, :w], random_effects=:marginal)
        r200 = population_margins(glmm, re_df; vars=[:x, :w], random_effects=:marginal, re_nodes=200)
        @test est(r) ≈ est(r200) rtol=1e-9
        @test se(r) ≈ se(r200) rtol=1e-9
        @test r200.metadata[:re_nodes] == 200
        # attenuation toward 0 relative to the typical-group effect
        @test abs(est(r)[1]) < abs(est(population_margins(glmm, re_df; vars=[:x]))[1])
        # the rule itself: exact moments of N(0, 1)
        (z, w) = Margins._gauss_hermite_normal(20)
        @test sum(w) ≈ 1
        @test abs(sum(w .* z)) < 1e-14
        @test sum(w .* z .^ 2) ≈ 1
        @test sum(w .* z .^ 4) ≈ 3
        @test sum(w .* z .^ 38) ≈ prod(1.0:2:37) rtol=1e-10
    end

    @testset ":marginal, identity link: the :zero estimates" begin
        r = population_margins(lmm_int, re_df; type=:predictions, random_effects=:marginal)
        r0 = population_margins(lmm_int, re_df; type=:predictions)
        @test est(r) == est(r0) && se(r) == se(r0)
        @test r.metadata[:random_effects] === :marginal && r.metadata[:re_nodes] == 0
    end

    @testset "per-factor modes: (village = :conditional, rater = :marginal)" begin
        hy = (village=:conditional, rater=:marginal)
        E = Base.get_extension(Margins, :MixedModelsExt)
        # identity link: averaging changes nothing, so = :conditional on villages (estimates and SEs)
        h = population_margins(lmm_int, re_df; type=:predictions, random_effects=hy)
        c = population_margins(lmm_int, re_df; type=:predictions, random_effects=:conditional, re_terms=:village)
        @test est(h) == est(c)
        @test se(h) == se(c)
        @test h.metadata[:random_effects] === :hybrid
        @test h.metadata[:re_terms] == [:village, :rater]
        @test h.metadata[:re_averaged] == [:rater]
        @test occursin("predicted for village; averaged over rater (no change", sprint(show, h))
        # probit, random intercepts: Φ((x'β + b̂_village)/√(1 + σ²_rater)), joint SE over (β, b_village)
        m = fit(MixedModel, @formula(yb ~ x + w + (1 | rater) + (1 | village)), re_df, Bernoulli(), ProbitLink();
                progress=false, fast=true)
        β = fixef(m); cc = sqrt(1 + m.σs.rater[1]^2)
        bv = [blup(m, :village, v) for v in re_df.village]
        t = (m.X * β .+ bv) ./ cc
        r = population_margins(m, re_df; type=:predictions, random_effects=hy)
        @test est(r)[1] ≈ mean(cdf.(Normal(), t)) atol=1e-12
        spec = Margins._validate_random_effects(m, :conditional, :zero, :village, :auto, :auto)
        (wv, _) = Margins._re_wrapper_model(m, Tables.columntable(re_df), spec, GLM.vcov)
        hc = E._hidden_column(wv, 1, Tables.columntable(re_df), nrow(re_df), :zero)
        Zv = Float64.(hc .== permutedims(Margins.CategoricalArrays.levels(hc)))
        g = [vec(mean(pdf.(Normal(), t) ./ cc .* m.X; dims=1)); vec(mean(pdf.(Normal(), t) ./ cc .* Zv; dims=1))]
        @test se(r)[1] ≈ sqrt(g' * wv.C * g) rtol=1e-10
        @test occursin("predicted for village; averaged over rater (population-averaged", sprint(show, r))
        # a profile row: the grid sets the village; the raters are averaged over
        p = profile_margins(m, re_df, DataFrame(x=[0.0], w=[0.0], village=["v3"]); type=:predictions, random_effects=hy)
        @test est(p)[1] ≈ cdf(Normal(), (β[1] + blup(m, :village, "v3")) / cc) atol=1e-12
        # random slopes in the averaged factor: τᵢ from w, the conditioned villages' b̂ in η
        ms = fit(MixedModel, @formula(yb ~ x + w + (1 + w | rater) + (1 | village)), re_df, Bernoulli(), ProbitLink();
                 progress=false, fast=true)
        βs = fixef(ms); Λ = only(r.λ for r in ms.reterms if string(r.trm) == "rater"); S = Matrix(Λ * Λ')
        cs = sqrt.(1 .+ S[1, 1] .+ 2S[1, 2] .* re_df.w .+ S[2, 2] .* re_df.w .^ 2)
        cs′ = (S[1, 2] .+ re_df.w .* S[2, 2]) ./ cs
        ηs = ms.X * βs .+ [blup(ms, :village, v) for v in re_df.village]
        e = population_margins(ms, re_df; vars=[:w], random_effects=hy)
        @test est(e)[1] ≈ mean(pdf.(Normal(), ηs ./ cs) .* (βs[3] ./ cs .- ηs .* cs′ ./ cs .^ 2)) atol=1e-12
        sc = population_margins(ms, re_df; type=:predictions, scenarios=(w=[1.5],), random_effects=hy)
        c15 = sqrt(1 + S[1, 1] + 3S[1, 2] + 2.25S[2, 2])
        η15 = ms.X[:, 1:2] * βs[1:2] .+ 1.5βs[3] .+ [blup(ms, :village, v) for v in re_df.village]
        @test est(sc)[1] ≈ mean(cdf.(Normal(), η15 ./ c15)) atol=1e-12
        # re_se = :fixed: the same estimate, β-only SE
        rf = population_margins(m, re_df; type=:predictions, random_effects=hy, re_se=:fixed)
        @test est(rf) ≈ est(r) atol=1e-14
        gβ = g[1:length(β)]
        @test se(rf)[1] ≈ sqrt(gβ' * vcov(m) * gβ) rtol=1e-10
        # one mode per NamedTuple reduces to that mode
        @test est(population_margins(m, re_df; type=:predictions, random_effects=(village=:conditional,))) ==
              est(population_margins(m, re_df; type=:predictions, random_effects=:conditional, re_terms=:village))
        @test est(population_margins(m, re_df; type=:predictions, random_effects=(rater=:marginal, village=:zero))) ==
              est(population_margins(m, re_df; type=:predictions, random_effects=:marginal, re_terms=:rater))
        @test est(population_margins(m, re_df; type=:predictions, random_effects=(village=:zero,))) ==
              est(population_margins(m, re_df; type=:predictions))
        @test second_differences(m, re_df, :x, :w; levels=[0.0, 1.0], random_effects=hy, warn=false) isa DataFrame
        # errors
        @test_throws ArgumentError population_margins(m, re_df; random_effects=(village=:cond,))
        @test_throws ArgumentError population_margins(m, re_df; random_effects=(school=:conditional,))
        @test_throws ArgumentError population_margins(m, re_df; random_effects=hy, re_terms=:village)
        @test_throws ArgumentError population_margins(m, re_df; random_effects=(village=:conditional,), re_nodes=20)
        @test_throws ArgumentError population_margins(m, re_df; random_effects=(rater=:marginal,), re_se=:fixed)
    end

    @testset "input checks: missing and absent grouping columns, grid mixtures" begin
        E = Base.get_extension(Margins, :MixedModelsExt)
        logistic(η) = 1 / (1 + exp(-η))
        # (1) a missing grouping value: the fit dropped those rows, and every mode rejects them as
        # :zero does (the wrapper's formula names no grouping columns, so this is checked first)
        dm = allowmissing(copy(re_df), :rater); dm.rater[1:20] .= missing
        mm = fit(MixedModel, @formula(yb ~ x + w + (1 + w | rater) + (1 | village)), dm, Bernoulli();
                 progress=false, fast=true)
        for kw in ((;), (random_effects=:conditional,), (random_effects=:conditional, re_se=:fixed),
                   (random_effects=:marginal,), (random_effects=(village=:conditional, rater=:marginal),))
            @test_throws MarginsError population_margins(mm, dm; type=:predictions, kw...)
        end
        # (2) a population request needs the conditioned factors' grouping columns
        d2 = select(re_df, Not(:village))
        for s in (:joint, :fixed)
            @test_throws ArgumentError population_margins(glmm, d2; type=:predictions,
                                                          random_effects=:conditional, re_se=s)
        end
        @test_throws ArgumentError population_margins(glmm, d2; type=:predictions,
                                                      random_effects=(village=:conditional, rater=:marginal))
        # a profile request may leave the column out: that factor's random effects are 0
        pr = profile_margins(glmm, d2, DataFrame(x=[0.0], w=[0.0], rater=["c"]); type=:predictions,
                             random_effects=:conditional)
        @test est(pr)[1] ≈ logistic(fixef(glmm)[1] + blup(glmm, :rater, "c")) atol=1e-10
        @test pr.metadata[:re_terms] == [:rater]
        # (3) a mixture in a grouping column: the share-weighted average of the levels' b̂ (the
        # dummy columns at the shares, as for a fixed-effects categorical), with joint SEs
        β = fixef(glmm)
        gmix = DataFrame(x=[0.3], w=[0.0], village=[mix("v1" => 0.25, "v2" => 0.75)])
        bv = 0.25 * blup(glmm, :village, "v1") + 0.75 * blup(glmm, :village, "v2")
        η = β[1] + 0.3β[2] + bv
        for s in (:joint, :fixed)
            p = profile_margins(glmm, re_df, gmix; type=:predictions, random_effects=:conditional, re_se=s)
            @test est(p)[1] ≈ logistic(η) atol=1e-10
            @test p.metadata[:re_terms] == [:village]
            μ′ = logistic(η) * (1 - logistic(η))
            gβ = μ′ .* [1.0, 0.3, 0.0]
            if s === :fixed
                @test se(p)[1] ≈ sqrt(gβ' * vcov(glmm) * gβ) rtol=1e-8
            else
                (Cbb, Cbβ, Cββ, offs) = E._joint_covariance(glmm)
                k = findfirst(r -> string(r.trm) == "village", glmm.LMM.reterms)
                lv = glmm.LMM.reterms[k].levels
                gb = zeros(size(Cbb, 1))
                gb[offs[k] + findfirst(==("v1"), lv)] = 0.25μ′; gb[offs[k] + findfirst(==("v2"), lv)] = 0.75μ′
                @test se(p)[1] ≈ sqrt(gβ' * Cββ * gβ + 2 * gb' * Cbβ * gβ + gb' * Cbb * gb) rtol=1e-8
            end
        end
        # on the link scale the mixture is the share-weighted average of the two levels' η
        pl = profile_margins(glmm, re_df, gmix; type=:predictions, scale=:link, random_effects=:conditional)
        p1 = profile_margins(glmm, re_df, DataFrame(x=[0.3, 0.3], w=[0.0, 0.0], village=["v1", "v2"]);
                             type=:predictions, scale=:link, random_effects=:conditional)
        @test est(pl)[1] ≈ 0.25 * est(p1)[1] + 0.75 * est(p1)[2] atol=1e-12
        # means_grid: every grouping column at its observed shares
        mg = means_grid(re_df)
        pg = profile_margins(glmm, re_df, mg; type=:predictions, random_effects=:conditional)
        sh(f) = (c = mg[1, f]; sum(w * blup(glmm, f, l) for (l, w) in zip(c.levels, c.weights)))
        @test est(pg)[1] ≈ logistic(β[1] + mean(re_df.x)β[2] + mean(re_df.w)β[3] + sh(:rater) + sh(:village)) atol=1e-10
        @test Set(pg.metadata[:re_terms]) == Set([:rater, :village])
        # a new level inside a mixture: share 0 of b̂ under re_new = :zero, an error under :error
        gnew = DataFrame(x=[0.3], w=[0.0], village=[mix("v1" => 0.25, "v99" => 0.75)])
        @test est(profile_margins(glmm, re_df, gnew; type=:predictions, random_effects=:conditional))[1] ≈
              logistic(β[1] + 0.3β[2] + 0.25 * blup(glmm, :village, "v1")) atol=1e-10
        @test_throws ArgumentError profile_margins(glmm, re_df, gnew; type=:predictions,
                                                   random_effects=:conditional, re_new=:error)
        # a nested factor's grouping columns cannot hold mixtures
        mn = fit(MixedModel, @formula(y ~ x + (1 | village / rater)), re_df; progress=false)
        @test_throws ArgumentError profile_margins(mn, re_df, means_grid(re_df); type=:predictions,
                                                   random_effects=:conditional)
        # (4) a categorical random-slope variable left out of a grid holds its frequency mixture: z
        # at the shares, for z'b̂ (:conditional) and for τ² = z'Σz (:marginal), probit closed forms
        dc = copy(re_df); dc.t = categorical(ifelse.(dc.w .> 0, "hi", "lo"))
        gp = fit(MixedModel, @formula(yb ~ x + t + (1 + t | rater) + (1 | village)), dc, Bernoulli(),
                 ProbitLink(); progress=false, fast=true)
        Φ(x) = cdf(Normal(), x)
        βp = fixef(gp); slo = mean(dc.t .== "lo")
        Σr = E._factor_Σ(gp, :rater); σv² = gp.σs.village[1]^2
        z̄ = [1.0, slo]
        τ̄² = z̄' * Σr * z̄ + σv²
        pmg = profile_margins(gp, dc, DataFrame(x=[0.3]); type=:predictions, random_effects=:marginal)
        @test est(pmg)[1] ≈ Φ((βp[1] + 0.3βp[2] + slo * βp[3]) / sqrt(1 + τ̄²)) atol=1e-10
        # the contrast lo − hi at that row evaluates t at each level (τ at z = [1, 1] and [1, 0])
        emg = profile_margins(gp, dc, DataFrame(x=[0.3]); vars=[:t], random_effects=:marginal)
        τlo² = [1.0, 1.0]' * Σr * [1.0, 1.0] + σv²; τhi² = Σr[1, 1] + σv²
        @test est(emg)[1] ≈ Φ((βp[1] + 0.3βp[2] + βp[3]) / sqrt(1 + τlo²)) - Φ((βp[1] + 0.3βp[2]) / sqrt(1 + τhi²)) atol=1e-10
        pcg = profile_margins(gp, dc, DataFrame(x=[0.3], rater=["c"]); type=:predictions, random_effects=:conditional)
        @test est(pcg)[1] ≈ Φ(βp[1] + 0.3βp[2] + slo * βp[3] + blup(gp, :rater, "c") + slo * blup(gp, :rater, "c", "t: lo")) atol=1e-10
        # (5) :marginal on the link scale averages nothing, and the header says so
        rl = population_margins(glmm, re_df; type=:predictions, scale=:link, random_effects=:marginal)
        @test occursin("no change on the link scale", sprint(show, MIME("text/plain"), rl))
    end

    @testset "validation" begin
        ols = lm(@formula(y ~ x), re_df)
        @test_throws ArgumentError population_margins(ols, re_df; random_effects=:conditional)
        @test population_margins(ols, re_df; random_effects=:zero) isa EffectsResult
        # an identity-link :marginal is :zero, so a random-slope variable's effect is allowed
        @test est(population_margins(lmm_slope, re_df; vars=[:w], random_effects=:marginal)) ≈
              [fixef(lmm_slope)[3]] atol=1e-12
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:marginal, re_new=:error)
        @test_throws ArgumentError population_margins(glmm, re_df; re_nodes=20)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_nodes=20)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:marginal, re_nodes=1)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:blups)
        @test_throws ArgumentError population_margins(glmm, re_df; re_new=:error)
        @test_throws ArgumentError population_margins(glmm, re_df; re_terms=:village)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_terms=:school)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_new=:drop)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_terms=Symbol[])
        # a per-factor NamedTuple must name grouping factors, :zero entries too, and needs a mixed model
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=(school=:zero,))
        @test_throws ArgumentError population_margins(ols, re_df; random_effects=(rater=:zero,))
    end
end
