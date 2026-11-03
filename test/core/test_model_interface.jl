# test_model_interface.jl
#
# The model-class interface of Margins 3.0 (step 5; src/core/model_interface.jl):
#
# - the df rule (decision D1): df is a property of the model class, whatever `vcov` is passed,
#   and `df=` overrides it;
# - offsets (B3): GLM and GLMM offsets enter η in every kernel; under `scenarios` they stay at
#   their observed values; at profile rows they must be given (decision D3);
# - FixedEffectModels through the shared request path: the absorbed fixed effects as a per-row
#   offset (B4), `esample` (B5), inference metadata (B6), `weights=:model` (decision D2, B11),
#   scenarios, groups and second differences.

using Test
using Margins
using GLM, DataFrames, CategoricalArrays, StatsModels, Statistics, Random
using Base.CoreLogging: Warn, NullLogger, with_logger   # (Logging is not a test dependency)
using MixedModels, FixedEffectModels
using Distributions: TDist, Normal, quantile, ccdf
using CovarianceMatrices: HC1, CR1

# Warnings logged by `f()` (a fresh TestLogger, so maxlog counts start at zero)
function _warnings(f)
    logs, _ = Test.collect_test_logs(f; min_level=Warn)
    return [l for l in logs if l.level == Warn]
end

@testset "Model-class interface" begin

    Random.seed!(20261026)
    n = 400
    d = DataFrame(x = randn(n), z = randn(n), cl = rand(1:12, n),
                  g = categorical(rand(["a", "b", "c"], n)),
                  e = 0.5 .+ 2 .* rand(n))
    d.y = 1 .+ 0.8 .* d.x .- 0.4 .* d.z .+ 0.3 .* (d.g .== "b") .+ randn(n)
    d.cnt = Float64.(rand.(Poisson.(d.e .* exp.(-0.2 .+ 0.4 .* d.x .+ 0.3 .* (d.g .== "c")))))
    d.yb = Float64.(rand(n) .< 1 ./ (1 .+ exp.(-(0.2 .+ 0.7 .* d.x))))

    @testset "df rule (D1): a property of the model class" begin
        m = lm(@formula(y ~ x + z + g), d)
        dfr = dof_residual(m)
        @test Margins._inference_df(m) == dfr
        # the same df whatever vcov is passed: robust, clustered, a function, a matrix
        for v in (GLM.vcov, HC1(), CR1(d.cl), mm -> GLM.vcov(mm), 2 .* vcov(m))
            r = DataFrame(population_margins(m, d; vars=[:x], vcov=v))
            @test r.df[1] == dfr
            @test r.p_value[1] ≈ 2 * ccdf(TDist(dfr), abs(r.statistic[1]))
            @test r.ci_upper[1] - r.ci_lower[1] ≈ 2 * quantile(TDist(dfr), 0.975) * r.se[1]
        end
        # robust lm: t(n−k), as Stata's `regress, vce(robust)` + margins and R's coeftest
        rh = DataFrame(profile_margins(m, d, means_grid(d); type=:predictions, vcov=HC1()))
        @test rh.df[1] == dfr
        # Gaussian identity GLM: t; other GLMs and mixed models: z
        mg = glm(@formula(y ~ x + z), d, Normal(), IdentityLink())
        @test Margins._inference_df(mg) == dof_residual(mg)
        @test DataFrame(population_margins(mg, d; vars=[:x], vcov=HC1())).df[1] == dof_residual(mg)
        mb = glm(@formula(yb ~ x), d, Binomial())
        @test Margins._inference_df(mb) == Inf
        mm = fit(MixedModel, @formula(y ~ x + (1 | cl)), d; progress=false)
        @test Margins._inference_df(mm) == Inf
        @test DataFrame(population_margins(mm, d; vars=[:x])).df[1] == Inf

        # second differences of a robust lm: t, at the model level and from a result
        mi = lm(@formula(y ~ x * z), d)
        sd = second_differences(mi, d, :x, :z; vcov=HC1(), at=:mean, warn=false)
        @test all(sd.df .== dof_residual(mi))
        ames = population_margins(mi, d; vars=[:x], scenarios=(z=[-1.0, 1.0],), vcov=HC1())
        @test all(second_differences(ames, :x, :z; warn=false).df .== dof_residual(mi))
        # ... also when the result was computed with another covariance matrix
        a3 = population_margins(mi, d; vars=[:x], scenarios=(z=[-1.0, 1.0],), vcov=3 .* vcov(mi))
        @test all(second_differences(a3, :x, :z; warn=false).df .== dof_residual(mi))
    end

    @testset "df= keyword overrides the rule" begin
        m = lm(@formula(y ~ x + z), d)
        G = length(unique(d.cl))
        r = DataFrame(population_margins(m, d; vars=[:x], vcov=CR1(d.cl), df=G - 1))
        @test r.df[1] == G - 1
        @test r.ci_upper[1] - r.ci_lower[1] ≈ 2 * quantile(TDist(G - 1), 0.975) * r.se[1]
        @test DataFrame(population_margins(m, d; vars=[:x], df=Inf)).df[1] == Inf
        @test DataFrame(population_margins(m, d; type=:predictions, groups=:g, df=7)).df == fill(7.0, 3)
        @test all(DataFrame(profile_margins(m, d, means_grid(d); vars=[:x], df=5)).df .== 5)
        mb = glm(@formula(yb ~ x), d, Binomial())
        @test DataFrame(population_margins(mb, d; vars=[:x], df=30)).df[1] == 30
        mi = lm(@formula(y ~ x * z), d)
        @test all(second_differences(mi, d, :x, :z; at=:mean, df=12, warn=false).df .== 12)
        @test all(second_differences(mi, d, :x, :z; levels=[-1.0, 1.0], df=Inf, warn=false).df .== Inf)
        # the override is stored, and result-level second differences inherit it
        ames = population_margins(mi, d; vars=[:x], scenarios=(z=[-1.0, 1.0],), df=9)
        @test ames.inference.df == 9
        @test all(second_differences(ames, :x, :z; warn=false).df .== 9)
        for bad in (0, -1, NaN, "10")
            @test_throws ArgumentError population_margins(m, d; vars=[:x], df=bad)
        end
        @test_throws ArgumentError profile_margins(m, d, means_grid(d); vars=[:x], df=0)
    end

    # ---------------------------------------------------------------------------------------
    # Offsets
    # ---------------------------------------------------------------------------------------

    off = log.(d.e)
    mp = glm(@formula(cnt ~ x + g), d, Poisson(); offset=off)
    β = coef(mp)
    X = modelmatrix(mp)
    μ = exp.(X * β .+ off)

    @testset "GLM offset: population (B3)" begin
        @test Margins._has_offset(mp) && Margins._offset(mp, (x=d.x,)) == off
        @test !Margins._has_offset(lm(@formula(y ~ x), d))           # lm stores zeros
        @test !Margins._has_offset(glm(@formula(cnt ~ x), d, Poisson()))
        # Oracle: with an intercept and the canonical log link, the score equations give
        # mean(μ̂) = mean(y), so the AAP is mean(y)
        aap = DataFrame(population_margins(mp, d; type=:predictions))
        @test aap.estimate[1] ≈ mean(d.cnt) rtol = 1e-8
        @test aap.estimate[1] ≈ mean(predict(mp)) rtol = 1e-12
        # its SE: ∇β = mean(μᵢ xᵢ)
        gβ = vec(mean(μ .* X; dims=1))
        @test aap.se[1] ≈ sqrt(gβ' * vcov(mp) * gβ) rtol = 1e-10
        # AME of x = β_x mean(μᵢ); categorical contrast at η + offset
        for backend in (:ad, :fd)
            ame = DataFrame(population_margins(mp, d; vars=[:x, :g], backend))
            @test ame.estimate[1] ≈ β[2] * mean(μ) rtol = (backend === :ad ? 1e-10 : 1e-6)
            Xa = copy(X); Xa[:, 3:4] .= 0; Xc = copy(Xa); Xc[:, 4] .= 1
            @test ame.estimate[end] ≈ mean(exp.(Xc * β .+ off) .- exp.(Xa * β .+ off)) rtol = 1e-10
        end
        # link scale: the offset is part of η for predictions, and cancels in effects
        aapl = DataFrame(population_margins(mp, d; type=:predictions, scale=:link))
        @test aapl.estimate[1] ≈ mean(X * β .+ off) rtol = 1e-12
        @test DataFrame(population_margins(mp, d; vars=[:x], scale=:link)).estimate[1] ≈ β[2] rtol = 1e-12
        # elasticity divides by ŷ at η + offset: here ∂μ/∂x · x/μ = β_x x
        el = DataFrame(population_margins(mp, d; vars=[:x], measure=:elasticity))
        @test el.estimate[1] ≈ β[2] * mean(d.x) rtol = 1e-10
        # weights and groups use each row's own offset
        w = 0.5 .+ rand(n)
        @test DataFrame(population_margins(mp, d; type=:predictions, weights=w)).estimate[1] ≈ sum(w .* μ) / sum(w) rtol = 1e-12
        grp = DataFrame(population_margins(mp, d; type=:predictions, groups=:g))
        for lev in levels(d.g)
            k = findfirst(==(lev), string.(grp.over_g))
            @test grp.estimate[k] ≈ mean(μ[d.g .== lev]) rtol = 1e-12
        end
        # an offset that cannot be matched to the data's rows is an error
        @test_throws ArgumentError population_margins(mp, d[1:end-1, :]; type=:predictions)
        # population margins take the model's own offset; there is no keyword
        @test_throws MethodError population_margins(mp, d; type=:predictions, offset=0.0)
    end

    @testset "offsets under scenarios: held at observed values" begin
        for x0 in (-1.0, 0.5)
            Xs = copy(X); Xs[:, 2] .= x0
            s = DataFrame(population_margins(mp, d; type=:predictions, scenarios=(x=x0,)))
            @test s.estimate[1] ≈ mean(exp.(Xs * β .+ off)) rtol = 1e-12
        end
        # an effect under a scenario of another variable
        Xb = copy(X); Xb[:, 3] .= 1; Xb[:, 4] .= 0
        s = DataFrame(population_margins(mp, d; vars=[:x], scenarios=(g="b",)))
        @test s.estimate[1] ≈ β[2] * mean(exp.(Xb * β .+ off)) rtol = 1e-10
        # second differences (scenarios built from the request) keep the offset too
        sd = second_differences(mp, d, :x, :g; warn=false)
        Xa = copy(X); Xa[:, 3:4] .= 0
        ame_a = β[2] * mean(exp.(Xa * β .+ off))
        @test sd.ame_at_level1[1] ≈ ame_a rtol = 1e-10
    end

    @testset "offsets at profile rows (D3)" begin
        grid = cartesian_grid(x=[0.0, 1.0], g=["a"])
        # required for predictions and response-scale effects: never a silent 0
        @test_throws ArgumentError profile_margins(mp, d, grid; type=:predictions)
        @test_throws ArgumentError profile_margins(mp, d, grid; vars=[:x])
        msg = try profile_margins(mp, d, grid; type=:predictions); "" catch e; sprint(showerror, e) end
        @test occursin("offset=", msg) && occursin("`offset` column", msg)
        # the keyword: one value, or one per grid row
        p1 = DataFrame(profile_margins(mp, d, grid; type=:predictions, offset=log(2.0)))
        @test p1.estimate ≈ exp.([β[1], β[1] + β[2]] .+ log(2.0)) rtol = 1e-12
        @test p1.se[2] ≈ sqrt([2exp(β[1] + β[2]), 2exp(β[1] + β[2]), 0, 0]' * vcov(mp) *
                              [2exp(β[1] + β[2]), 2exp(β[1] + β[2]), 0, 0]) rtol = 1e-10
        pv = DataFrame(profile_margins(mp, d, grid; type=:predictions, offset=[0.0, log(3.0)]))
        @test pv.estimate ≈ exp.([β[1], β[1] + β[2] + log(3.0)]) rtol = 1e-12
        # an `offset` grid column does the same, and is not shown as a grid variable
        gcol = copy(grid); gcol.offset = [0.0, log(3.0)]
        pc = DataFrame(profile_margins(mp, d, gcol; type=:predictions))
        @test pc.estimate == pv.estimate
        @test !("offset" in names(pc))
        # effects at η + offset: ∂μ/∂x = β_x exp(η + offset)
        ef = DataFrame(profile_margins(mp, d, grid; vars=[:x], offset=log(2.0)))
        @test ef.estimate ≈ β[2] .* exp.([β[1], β[1] + β[2]] .+ log(2.0)) rtol = 1e-10
        # not needed (and accepted, unused) where it cannot matter: link-scale effects
        el = DataFrame(profile_margins(mp, d, grid; vars=[:x], scale=:link))
        @test el.estimate ≈ [β[2], β[2]] rtol = 1e-12
        @test DataFrame(profile_margins(mp, d, grid; vars=[:x], scale=:link, offset=1.0)).estimate == el.estimate
        # errors: given twice, wrong length, non-finite, for a model without an offset
        @test_throws ArgumentError profile_margins(mp, d, gcol; type=:predictions, offset=0.0)
        @test_throws ArgumentError profile_margins(mp, d, grid; type=:predictions, offset=[0.0, 0.0, 0.0])
        @test_throws ArgumentError profile_margins(mp, d, grid; type=:predictions, offset=NaN)
        m0 = glm(@formula(cnt ~ x + g), d, Poisson())
        @test_throws ArgumentError profile_margins(m0, d, grid; type=:predictions, offset=0.0)
        # mixed-level grids are evaluated in blocks: each block gets its rows' offsets
        hg = DataFrame(x=[0.0, 1.0, 2.0], g=[mix("a" => 0.5, "b" => 0.5), "a", "a"])
        ph = DataFrame(profile_margins(mp, d, hg; type=:predictions, offset=[0.1, 0.2, 0.3]))
        @test ph.estimate[2] ≈ exp(β[1] + β[2] + 0.2) rtol = 1e-12
        @test ph.estimate[3] ≈ exp(β[1] + 2β[2] + 0.3) rtol = 1e-12
    end

    @testset "GLMM offset" begin
        gm = fit(MixedModel, @formula(cnt ~ x + (1 | cl)), d, Poisson(); offset=off, progress=false)
        @test Margins._has_offset(gm)
        βg = coef(gm)
        μg = exp.(βg[1] .+ βg[2] .* d.x .+ off)        # population level: random effects at 0
        @test DataFrame(population_margins(gm, d; type=:predictions)).estimate[1] ≈ mean(μg) rtol = 1e-12
        @test DataFrame(population_margins(gm, d; vars=[:x])).estimate[1] ≈ βg[2] * mean(μg) rtol = 1e-10
        @test_throws ArgumentError profile_margins(gm, d, means_grid(d); type=:predictions)
        pg = DataFrame(profile_margins(gm, d, means_grid(d); type=:predictions, offset=0.0))
        @test pg.estimate[1] ≈ exp(βg[1] + βg[2] * mean(d.x)) rtol = 1e-12
        gm0 = fit(MixedModel, @formula(cnt ~ x + (1 | cl)), d, Poisson(); progress=false)
        @test !Margins._has_offset(gm0)
    end

    # ---------------------------------------------------------------------------------------
    # FixedEffectModels on the shared request path
    # ---------------------------------------------------------------------------------------

    df = DataFrame(x1 = randn(n), x2 = randn(n), st = rand(["A", "B", "C", "D", "E"], n),
                   yr = rand(2001:2006, n), w = 0.5 .+ rand(n))
    fe_st = Dict("A" => 1.0, "B" => -1.0, "C" => 0.5, "D" => -0.5, "E" => 0.0)
    df.y = 2 .* df.x1 .- df.x2 .+ [fe_st[s] for s in df.st] .+ 0.1 .* (df.yr .- 2003) .+ 0.3 .* randn(n)
    df.stc = categorical(df.st)
    # rows the fit drops: a singleton state, and a missing regressor
    df = allowmissing(df, :x2)
    push!(df, (x1=0.3, x2=0.1, st="Z", yr=2003, w=1.0, y=1.0, stc="A"); promote=true)
    df.x2[5] = missing
    fm = reg(df, @formula(y ~ x1 + x2 + fe(st) + fe(yr)), Vcov.cluster(:yr); save=:fe)
    es = fm.esample
    pf = predict(fm, df)

    @testset "FEM: inference (B6) and df" begin
        r = population_margins(fm, df; vars=[:x1, :x2], weights=:none)
        out = DataFrame(r)
        @test r.inference.df == dof_residual(fm) == length(unique(df.yr)) - 1   # cluster-adjusted
        @test r.inference.vcov == Matrix(vcov(fm))
        # additive model: the AME is the coefficient, and its interval is reg's own
        ci = confint(fm)
        @test out.estimate ≈ coef(fm) rtol = 1e-10
        @test out.ci_lower ≈ ci[:, 1] rtol = 1e-8
        @test out.ci_upper ≈ ci[:, 2] rtol = 1e-8
        # result-level second differences work on a FEM result
        fi = reg(df, @formula(y ~ x1 * x2 + fe(st)); save=:fe)
        ames = population_margins(fi, df; vars=[:x1], scenarios=(x2=[-1.0, 1.0],), weights=:none)
        sd = second_differences(ames, :x1, :x2; warn=false)
        @test sd.second_diff[1] ≈ 2 * coef(fi)[3] rtol = 1e-8
        @test sd.df[1] == dof_residual(fi)
        # and at the model level (a slope between the two levels: the difference over 2)
        sdm = second_differences(fi, df, :x1, :x2; levels=[-1.0, 1.0], weights=:none, warn=false)
        @test sdm.second_diff[1] ≈ sd.second_diff[1] / 2 rtol = 1e-10
    end

    @testset "FEM: esample (B5) and predictions with the FE offset (B4)" begin
        @test sum(es) == nrow(df) - 2
        @test Margins._estimation_rows(fm, nrow(df)) == findall(es)
        aap = population_margins(fm, df; type=:predictions, weights=:none)
        @test aap.metadata[:n_obs] == sum(es)
        @test DataFrame(aap).estimate[1] ≈ mean(pf[es]) rtol = 1e-12
        # groups: each row's own fixed effects (B4: used to add one average offset to every group)
        gp = DataFrame(population_margins(fm, df; type=:predictions, groups=:stc, weights=:none))
        for lev in ("A", "B", "C", "D", "E")
            k = findfirst(==(lev), string.(gp.over_stc))
            @test gp.estimate[k] ≈ mean(pf[es .& (df.st .== lev)]) rtol = 1e-12
        end
        # scenarios: the fixed effects stay at each row's observed values
        sc = DataFrame(population_margins(fm, df; type=:predictions, scenarios=(x1=[0.0, 1.0],), weights=:none))
        for (k, v) in enumerate((0.0, 1.0))
            dv = copy(df); dv.x1 .= v
            @test sc.estimate[k] ≈ mean(predict(fm, dv)[es]) rtol = 1e-12
        end
        # weights: explicit, a column or a vector over all rows of the data (subset to esample)
        wp = DataFrame(population_margins(fm, df; type=:predictions, weights=:w))
        @test wp.estimate[1] ≈ sum(df.w[es] .* pf[es]) / sum(df.w[es]) rtol = 1e-12
        @test DataFrame(population_margins(fm, df; type=:predictions, weights=df.w)).estimate == wp.estimate
        # the data must be the table given to reg(): one row per entry of esample, subset here.
        # A table with nobs rows (pre-subset by hand, or another table of that length) is an error,
        # not taken to be the estimation sample, which would misalign rows silently.
        err = try population_margins(fm, df[es, :]; type=:predictions, weights=:none); nothing catch e; e end
        @test err isa ArgumentError && occursin("Pass the table given to reg()", err.msg)
        @test_throws ArgumentError population_margins(fm, df[findall(!, es) ∪ findall(es)[3:end], :]; type=:predictions, weights=:none)
        @test_throws ArgumentError population_margins(fm, df[1:10, :]; type=:predictions, weights=:none)
        @test_throws ArgumentError profile_margins(fm, df[es, :], DataFrame(x1=[0.0]); vars=[:x1])
        @test_throws ArgumentError second_differences(fm, df[es, :], :x1, :x2; levels=[0.0, 1.0], warn=false)
        # the model-level second differences take the full table too, and use the estimation rows
        sdf = second_differences(fm, df, :x1, :x2; levels=[0.0, 1.0], weights=:none, warn=false)
        ref = population_margins(fm, df; vars=[:x1], scenarios=(x2=[0.0, 1.0],), weights=:none)
        @test sdf.ame_at_level1[1] == ref.estimates[1] && sdf.ame_at_level2[1] == ref.estimates[2]
        # predictions need the FE estimates (save=:fe)
        fm0 = reg(df, @formula(y ~ x1 + x2 + fe(st)))
        @test_throws Margins.MarginsError population_margins(fm0, df; type=:predictions, weights=:none)
        # effects that do not involve ŷ do not need them
        @test DataFrame(population_margins(fm0, df; vars=[:x1], weights=:none)).estimate[1] ≈ coef(fm0)[1] rtol = 1e-10
    end

    @testset "FEM: profile predictions" begin
        grid = DataFrame(x1=[0.0, 1.0], x2=[0.0, 0.0], st=["A", "C"], yr=[2002, 2005])
        pp = DataFrame(profile_margins(fm, df, grid; type=:predictions))
        @test pp.estimate ≈ predict(fm, grid) rtol = 1e-12
        # without the fixed effects' levels (or an offset) it is an error, not the average FE
        @test_throws ArgumentError profile_margins(fm, df, DataFrame(x1=[0.0], x2=[0.0]); type=:predictions)
        @test_throws ArgumentError profile_margins(fm, df, DataFrame(x1=[0.0], x2=[0.0], st=["A"]); type=:predictions)
        @test_throws ArgumentError profile_margins(fm, df, DataFrame(x1=[0.0], x2=[0.0], st=["Q"], yr=[2002]); type=:predictions)
        # an explicit offset (the sum of the fixed effects) works too
        po = DataFrame(profile_margins(fm, df, DataFrame(x1=[1.0], x2=[0.0]); type=:predictions, offset=0.25))
        @test po.estimate[1] ≈ coef(fm)[1] + 0.25 rtol = 1e-12
        # effects need no offset
        @test DataFrame(profile_margins(fm, df, DataFrame(x1=[0.0]); vars=[:x1])).estimate[1] ≈ coef(fm)[1] rtol = 1e-10
    end

    @testset "FEM weights=:model (D2): one warning per session, unweighted" begin
        fw = reg(df, @formula(y ~ x1 + x2 + fe(st)); weights=:w, save=:fe)
        @test Margins._model_weights(fw) isa Margins.UnknownWeights
        warns = _warnings() do
            population_margins(fw, df; type=:predictions)
            population_margins(fw, df; vars=[:x1])
            second_differences(fw, df, :x1, :x2; levels=[-1.0, 1.0], warn=false)
        end
        @test length(warns) == 1
        @test occursin("weights=:model", warns[1].message) && occursin("weights=:none", warns[1].message)
        r = with_logger(NullLogger()) do
            population_margins(fw, df; type=:predictions)
        end
        @test r.metadata[:weights_source] === :none
        @test DataFrame(r).estimate[1] ≈ mean(predict(fw, df)[fw.esample]) rtol = 1e-12
        # explicit weights, or :none, do not warn
        @test isempty(_warnings(() -> population_margins(fw, df; type=:predictions, weights=:w)))
        @test isempty(_warnings(() -> population_margins(fw, df; type=:predictions, weights=:none)))
        @test isempty(_warnings(() -> population_margins(fw, df; vars=[:x1], weights=nothing)))
    end
end
