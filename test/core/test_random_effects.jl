# test_random_effects.jl
# `random_effects = :zero | :conditional | :marginal` for MixedModels fits.
# References are closed forms from MixedModels' own quantities: `fitted(m)` (which includes
# Z b̂), `fixef`, `ranef` / `raneftables` and `vcov`.

using Test, DataFrames, MixedModels, Random, Statistics, Distributions, LinearAlgebra
using CategoricalArrays
using Margins
using Margins: MarginsError
using GLM: lm

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

# b̂ of `level` for factor `f`, coefficient `coef`
blup(m, f, level, coef="(Intercept)") = (t = getproperty(raneftables(m), f);
    getproperty(t[findfirst(==(level), getproperty(t, f))], Symbol(coef)))

est(r) = DataFrame(r).estimate
se(r) = DataFrame(r).se

@testset "random_effects (MixedModels)" begin
    @testset "LMM: conditional predictions are the fitted values" begin
        for m in (lmm_int, lmm_slope)
            r = population_margins(m, re_df; type=:predictions, random_effects=:conditional)
            @test est(r)[1] ≈ mean(fitted(m)) atol=1e-10
            r0 = population_margins(m, re_df; type=:predictions)
            @test est(r0)[1] ≈ mean(m.X * fixef(m)) atol=1e-10
            # the delta-method SE is on β only, so the same gradient (x̄) as at zero
            @test se(r)[1] ≈ se(r0)[1] atol=1e-12
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
        r = population_margins(glmm, re_df; vars=[:x], random_effects=:conditional)
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
        rw = population_margins(lmm_slope, re_df; vars=[:w], random_effects=:conditional)
        @test est(rw)[1] ≈ fixef(lmm_slope)[3] + mean(bw) atol=1e-8
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
        e = population_margins(gs, re_df; vars=[:w], random_effects=:conditional)
        @test est(e)[1] ≈ mean(d1 .* sl) atol=1e-8
        g = vec(mean(d2 .* sl .* X; dims=1)); g[3] += mean(d1)
        @test se(e)[1] ≈ sqrt(g' * V * g) atol=1e-8
        # the finite-difference backend agrees
        @test est(population_margins(gs, re_df; vars=[:w], random_effects=:conditional, backend=:fd))[1] ≈
              est(e)[1] rtol=1e-6
        # a categorical variable with a random slope: its contrasts are not supported yet
        dc = copy(re_df); dc.t = categorical(ifelse.(dc.w .> 0, "hi", "lo"))
        mc = fit(MixedModel, @formula(y ~ x + t + (1 + t | rater)), dc; progress=false)
        @test_throws MarginsError population_margins(mc, dc; vars=[:t], random_effects=:conditional)
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
        # setting w counterfactually is blocked in population requests
        @test_throws MarginsError population_margins(lmm_slope, re_df; type=:predictions,
                                                     scenarios=(w=[0.0, 1.0],), random_effects=:conditional)
        @test_throws MarginsError second_differences(lmm_slope, re_df, :x, :w; random_effects=:conditional, warn=false)
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
        @test est(ew)[1] ≈ mean(dw) atol=1e-8
        gw = vec(mean((-t .* φ.(t)) ./ c .* (β[3] ./ c .- η .* cp ./ c .^ 2) .* X .-
                      φ.(t) .* cp ./ c .^ 2 .* X; dims=1))
        gw[3] += mean(φ.(t) ./ c)
        @test se(ew)[1] ≈ sqrt(gw' * V * gw) atol=1e-8
        # at profile rows
        pw = profile_margins(m, d, gr; vars=[:w], random_effects=:marginal)
        ηp = [β[1] + β[2]x + β[3]w for (x, w) in zip(gr.x, gr.w)]
        cg = sqrt.(1 .+ τ².(gr.w)); cgp = (Σr[1, 2] .+ gr.w .* Σr[2, 2]) ./ cg
        @test est(pw) ≈ φ.(ηp ./ cg) .* (β[3] ./ cg .- ηp .* cgp ./ cg .^ 2) atol=1e-8
        # population scenarios on w are not supported yet
        @test_throws MarginsError population_margins(m, d; type=:predictions, scenarios=(w=0.0,),
                                                     random_effects=:marginal)
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
    end
end
