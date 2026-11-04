# test_random_effects.jl
# `random_effects = :zero | :conditional` for MixedModels fits (Margins 3.1, phase 1).
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
        # w has a rater slope: its conditional effect is not supported yet, named or detected
        @test_throws MarginsError population_margins(lmm_slope, re_df; vars=[:w], random_effects=:conditional)
        err = try population_margins(lmm_slope, re_df; random_effects=:conditional) catch e; e end
        @test err isa MarginsError && occursin("vars = [:x]", err.msg)
        # ... unless the rater is not conditioned on
        rw = population_margins(lmm_slope, re_df; vars=[:w], random_effects=:conditional, re_terms=[:village])
        @test est(rw)[1] ≈ fixef(lmm_slope)[3] atol=1e-10
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

    @testset "validation" begin
        ols = lm(@formula(y ~ x), re_df)
        @test_throws ArgumentError population_margins(ols, re_df; random_effects=:conditional)
        @test population_margins(ols, re_df; random_effects=:zero) isa EffectsResult
        @test_throws MarginsError population_margins(glmm, re_df; random_effects=:marginal)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:blups)
        @test_throws ArgumentError population_margins(glmm, re_df; re_new=:error)
        @test_throws ArgumentError population_margins(glmm, re_df; re_terms=:village)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_terms=:school)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_new=:drop)
        @test_throws ArgumentError population_margins(glmm, re_df; random_effects=:conditional, re_terms=Symbol[])
    end
end
