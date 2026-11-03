# test_second_differences_model_interface.jl
# second_differences(model, data, variables, modifier; ...)

using Test
using Margins
using DataFrames
using GLM
using CategoricalArrays
using LinearAlgebra: Symmetric
using Random
using CovarianceMatrices
using CSV
using Distributions: TDist, Normal, quantile

@testset "second_differences(model, data, …)" begin
    Random.seed!(7)
    n = 800
    df = DataFrame(
        x = randn(n), w = randn(n), z = randn(n),
        t = rand([0, 1], n), g = rand(["a", "b", "c"], n),
    )
    df.y = 0.2 .+ 0.4df.x .+ 0.3df.w .+ 0.5df.t .+ 0.7df.x .* df.t .- 0.6df.x .* df.w .+
           0.3 .* (df.g .== "b") .+ 0.4 .* df.x .* (df.g .== "c") .+ 0.5df.z .+ randn(n)
    lmod = lm(@formula(y ~ x * t + x * w + x * g + z), df)
    β = coef(lmod); names = coefnames(lmod)
    b(nm) = β[findfirst(==(nm), names)]

    @testset "binary modifier: equals the interaction coefficient in a linear model" begin
        sd = second_differences(lmod, df, :x, :t)
        @test nrow(sd) == 1
        @test sd.second_diff[1] ≈ b("x & t") atol=1e-8
        @test sd.modifier_type[1] == :binary
        @test (sd.modifier_level1[1], sd.modifier_level2[1]) == (0, 1)
        @test sd.df[1] == dof_residual(lmod)
        # same answer as building it by hand
        ames = population_margins(lmod, df; vars=:x, scenarios=(t=[0, 1],), type=:effects)
        by_hand = second_differences(ames, :x, :t)
        @test sd.second_diff[1] ≈ by_hand.second_diff[1]
        @test sd.se[1] ≈ by_hand.se[1]
    end

    @testset "continuous modifier: local derivative and slope between levels" begin
        loc = second_differences(lmod, df, :x, :w)
        @test loc.second_diff[1] ≈ b("x & w") atol=1e-6
        @test isnan(loc.eval_point[1])                                   # default at = :observed: no single point
        @test isequal(loc, second_differences(lmod, df, :x, :w; at = :observed))
        @test second_differences(lmod, df, :x, :w; at = :mean).eval_point[1] ≈ sum(df.w) / n
        @test loc.modifier_type[1] == :continuous
        sl = second_differences(lmod, df, :x, :w; levels=[-1.0, 0.0, 1.0])
        @test nrow(sl) == 3
        @test all(isapprox.(sl.second_diff, b("x & w"); atol=1e-8))
        two = second_differences(lmod, df, :x, :w; levels=[0.0, 10.0])   # slope, not raw difference
        @test two.second_diff[1] ≈ b("x & w") atol=1e-8
        @test two.modifier_type[1] == :continuous
        @test second_differences(lmod, df, :x, :w; at=[-1.0, 1.0]).eval_point == [-1.0, 1.0]
    end

    @testset "categorical modifier, several focal variables" begin
        sd = second_differences(lmod, df, [:x, :z], :g; warn=false)
        @test nrow(sd) == 6                       # 2 variables × 3 level pairs
        @test Set(sd.variable) == Set([:x, :z])
        xr = sd[(sd.variable .== :x) .&& (sd.modifier_level1 .== "a") .&& (sd.modifier_level2 .== "c"), :]
        @test xr.second_diff[1] ≈ b("x & g: c") atol=1e-8
        zr = sd[sd.variable .== :z, :]
        @test all(abs.(zr.second_diff) .< 1e-8)    # no z × g interaction
    end

    @testset "categorical focal variable and categorical modifier array" begin
        d2 = copy(df); d2.gc = categorical(d2.g; levels=["c", "b", "a"])
        m2 = lm(@formula(y ~ x + t * gc + w + z), d2)
        sd = second_differences(m2, d2, :gc, :t)
        @test nrow(sd) == 2                       # baseline contrasts
        @test all(sd.modifier_level1 .== 0) && all(sd.modifier_level2 .== 1)
        sdm = second_differences(m2, d2, :x, :gc; warn=false)    # no x:gc term in m2
        @test (sdm.modifier_level1[1], sdm.modifier_level2[1]) == ("c", "b")   # level order, not alphabetical
    end

    @testset "profile strata and options" begin
        sd = second_differences(lmod, df, :x, :t; profile=(g=["a", "b"],))
        @test nrow(sd) == 2 && sd.g == ["a", "b"]
        # integer-coded categorical modifier
        d3 = copy(df); d3.k = rand(1:3, n)
        m3 = lm(@formula(y ~ x * k + t + w + z), d3)     # k enters the model, with an interaction
        auto = @test_logs (:warn, r"takes only 3 integer values \(1, 2, 3\)") second_differences(m3, d3, :x, :k)
        @test all(auto.modifier_type .== :continuous)
        # no warning when the user states what they mean
        @test_logs second_differences(m3, d3, :x, :k; at=2)
        @test_logs second_differences(m3, d3, :x, :k; levels=[1, 3])
        cat = @test_logs second_differences(m3, d3, :x, :k; modifier_type=:categorical)
        @test all(cat.modifier_type .== :categorical)
        # many distinct integers (an age-like variable) is continuous without a warning
        d3.age = rand(20:70, n)
        m3b = lm(@formula(y ~ x * age + t + z), d3)
        @test_logs second_differences(m3b, d3, :x, :age)
        # non-integer numeric modifier: no warning
        @test_logs second_differences(lmod, df, :x, :w)
        # covariance given as matrix / Symmetric / function: same numbers, t for model-based, z otherwise
        V = vcov(lmod)
        base = second_differences(lmod, df, :x, :t)
        for v in (V, Symmetric(V))
            r = second_differences(lmod, df, :x, :t; vcov=v)
            @test r.se[1] ≈ base.se[1] && r.df[1] == dof_residual(lmod)
        end
        # any other covariance: same t df (3.0 step 5, D1: df is the model class's)
        rf = second_differences(lmod, df, :x, :t; vcov=m -> vcov(m))
        @test rf.se[1] ≈ base.se[1] && rf.df[1] == dof_residual(lmod)
        rob = second_differences(lmod, df, :x, :t; vcov=1.5 .* V)
        @test rob.se[1] ≈ sqrt(1.5) * base.se[1] rtol=1e-6
        @test rob.df[1] == dof_residual(lmod)
        @test second_differences(lmod, df, :x, :w; vcov=1.5 .* V).df[1] == dof_residual(lmod)
        @test second_differences(lmod, df, :x, :w; vcov=1.5 .* V, df=Inf).df[1] == Inf
    end

    @testset "CovarianceMatrices estimator as vcov (HC1)" begin
        V = vcov(lmod); Vr = CovarianceMatrices.vcov(HC1(), lmod)
        @test Vr != V
        # discrete comparison: estimator == same estimator as an explicit matrix == hand-built result
        r_est = second_differences(lmod, df, :x, :t; vcov=HC1())
        r_mat = second_differences(lmod, df, :x, :t; vcov=Vr)
        ames = population_margins(lmod, df; vars=:x, scenarios=(t=[0, 1],), type=:effects, vcov=HC1())
        r_hand = second_differences(ames, :x, :t)
        @test r_est.second_diff[1] ≈ second_differences(lmod, df, :x, :t).second_diff[1]
        @test r_est.se[1] ≈ r_mat.se[1] ≈ r_hand.se[1]
        @test r_est.se[1] != second_differences(lmod, df, :x, :t).se[1]
        dfr = dof_residual(lmod)   # robust lm: t(n − k) (3.0 step 5, D1)
        @test r_est.df[1] == dfr && r_mat.df[1] == dfr && r_hand.df[1] == dfr
        # the result carries the robust Σ it was computed with
        @test ames.inference.vcov ≈ CovarianceMatrices.vcov(HC1(), lmod)
        # local derivative
        l_est = second_differences(lmod, df, :x, :w; vcov=HC1())
        l_mat = second_differences(lmod, df, :x, :w; vcov=Vr)
        @test l_est.se[1] ≈ l_mat.se[1]
        @test l_est.df[1] == dfr
        @test l_est.second_diff[1] ≈ second_differences(lmod, df, :x, :w).second_diff[1]
        # categorical modifier, several focal variables
        c_est = second_differences(lmod, df, [:x, :z], :g; vcov=HC1(), warn=false)
        @test all(c_est.df .== dfr) && nrow(c_est) == 6
    end

    @testset "confidence intervals and alpha" begin
        sd = second_differences(lmod, df, :x, :t)
        crit = quantile(TDist(dof_residual(lmod)), 0.975)
        @test sd.ci_lower[1] ≈ sd.second_diff[1] - crit * sd.se[1]
        @test sd.ci_upper[1] ≈ sd.second_diff[1] + crit * sd.se[1]
        # columns sit between p_value and significant
        nm = propertynames(sd)
        @test nm[findfirst(==(:p_value), nm) + 1:findfirst(==(:p_value), nm) + 2] == [:ci_lower, :ci_upper]
        wide = second_differences(lmod, df, :x, :t; alpha=0.01)
        narrow = second_differences(lmod, df, :x, :t; alpha=0.2)
        @test (wide.ci_upper[1] - wide.ci_lower[1]) > (sd.ci_upper[1] - sd.ci_lower[1]) > (narrow.ci_upper[1] - narrow.ci_lower[1])
        @test narrow.ci_lower[1] ≈ narrow.second_diff[1] - quantile(TDist(dof_residual(lmod)), 0.9) * narrow.se[1]
        # significance follows alpha, interval excludes zero exactly when significant
        for a in (0.001, 0.05, 0.5)
            r = second_differences(lmod, df, :x, :g; alpha=a)
            @test all(r.significant .== (r.p_value .< a))
            @test all(r.significant .== ((r.ci_lower .> 0) .| (r.ci_upper .< 0)))
        end
        # z interval for a non-linear model
        yb = df.y .> 0; d4 = copy(df); d4.yb = yb
        gm = glm(@formula(yb ~ x * t + w), d4, Binomial(), LogitLink())
        r = second_differences(gm, d4, :x, :t)
        @test r.ci_upper[1] ≈ r.second_diff[1] + quantile(Normal(), 0.975) * r.se[1]
        # local derivative and result-level functions
        l = second_differences(lmod, df, :x, :w; alpha=0.2)
        @test l.ci_lower[1] ≈ l.second_diff[1] - quantile(TDist(dof_residual(lmod)), 0.9) * l.se[1]
        a1 = population_margins(lmod, df; vars=:x, scenarios=(t=[0, 1],), type=:effects, alpha=0.2)
        @test second_differences(a1, :x, :t).ci_lower[1] ≈ narrow.ci_lower[1]            # inherits the result's alpha
        @test second_differences(a1, :x, :t; alpha=0.05).ci_lower[1] ≈ sd.ci_lower[1]   # explicit alpha wins
        @test Margins._second_difference(a1, :x, :t).ci_lower ≈ narrow.ci_lower[1]
        @test_throws ErrorException second_differences(lmod, df, :x, :t; alpha=0.0)
        @test_throws ErrorException second_differences(lmod, df, :x, :t; alpha=1.5)
        @test_throws ErrorException second_differences(lmod, df, :x, :w; alpha=2)
    end

    @testset "multiple-testing adjustment" begin
        p = [0.01, 0.04, 0.03, 0.005]
        @test Margins._adjust_pvalues(p, :none) == p
        @test Margins._adjust_pvalues(p, :bonferroni) ≈ [0.04, 0.16, 0.12, 0.02]
        @test Margins._adjust_pvalues(p, :holm) ≈ [0.03, 0.06, 0.06, 0.02]
        @test Margins._adjust_pvalues(p, :bh) ≈ [0.02, 0.04, 0.04, 0.02]
        @test Margins._adjust_pvalues([0.5, 0.9], :bonferroni) == [1.0, 1.0]
        @test Margins._adjust_pvalues(Float64[], :holm) == Float64[]
        @test_throws ErrorException Margins._adjust_pvalues(p, :nope)

        raw = second_differences(lmod, df, [:x, :z], :g; warn=false)
        @test :p_adjusted ∉ propertynames(raw)
        tested = .!isnan.(raw.p_value)
        @test count(tested) == 3 && count(.!tested) == 3        # z × g: no interaction, structural zeros
        for (meth, f) in ((:bonferroni, p -> min.(1, length(p) .* p)),
                          (:holm, p -> Margins._adjust_pvalues(p, :holm)),
                          (:bh, p -> Margins._adjust_pvalues(p, :bh)))
            adj = second_differences(lmod, df, [:x, :z], :g; p_adjust=meth, alpha=0.05, warn=false)
            @test adj.p_adjusted[tested] ≈ f(raw.p_value[tested])   # family = the rows that are tests
            @test all(isnan, adj.p_adjusted[.!tested])
            @test all(adj.p_adjusted[tested] .>= adj.p_value[tested] .- 1e-15)
            @test adj.significant == (tested .& (adj.p_adjusted .< 0.05))
            nm = propertynames(adj)
            @test nm[findfirst(==(:p_value), nm) + 1] == :p_adjusted
            @test isequal(adj.p_value, raw.p_value) && adj.second_diff == raw.second_diff
        end
        # family spans strata and evaluation points
        st = second_differences(lmod, df, :x, :t; profile=(g=["a", "b", "c"],), p_adjust=:bonferroni)
        @test st.p_adjusted ≈ min.(1, 3 .* st.p_value)
        ev = second_differences(lmod, df, :x, :w; at=[-1.0, 0.0, 1.0], p_adjust=:holm)
        @test nrow(ev) == 3 && :p_adjusted in propertynames(ev)
        # result-level functions
        ames = population_margins(lmod, df; vars=[:x, :z], scenarios=(g=["a", "b", "c"],), type=:effects)
        rl = second_differences(ames, [:x, :z], :g; p_adjust=:bonferroni, warn=false)
        rt = .!isnan.(rl.p_value)
        @test rl.p_adjusted[rt] ≈ min.(1, count(rt) .* rl.p_value[rt])   # family excludes structural zeros
        @test second_differences(ames, [:x, :z], :g; p_adjust=:holm, warn=false).p_adjusted[rt] ≈
              Margins._adjust_pvalues(rl.p_value[rt], :holm)
        pw = Margins._second_differences_pairwise(ames, :x, :g; p_adjust=:bonferroni)
        @test pw.p_adjusted ≈ min.(1, 3 .* pw.p_value)
        @test_throws ErrorException second_differences(lmod, df, :x, :t; p_adjust=:nope)
        @test_throws ErrorException Margins._second_differences_local(lmod, df, :x, :w, vcov(lmod); p_adjust=:nope)
    end

    @testset "gradients are opt-in, as in DataFrame(result)" begin
        V = vcov(lmod)
        calls = (
            () -> second_differences(lmod, df, :x, :t; include_gradients=true),
            () -> second_differences(lmod, df, [:x, :z], :g; include_gradients=true, warn=false),
            () -> second_differences(lmod, df, :x, :w; include_gradients=true),
            () -> second_differences(lmod, df, :x, :w; levels=[-1.0, 1.0], include_gradients=true),
        )
        defaults = (
            second_differences(lmod, df, :x, :t),
            second_differences(lmod, df, [:x, :z], :g; warn=false),
            second_differences(lmod, df, :x, :w),
            second_differences(lmod, df, :x, :w; levels=[-1.0, 1.0]),
        )
        for (with, without) in zip(calls, defaults)
            w = with()
            @test :gradient ∉ propertynames(without)
            @test !any(eltype(c) <: AbstractVector for c in eachcol(without))   # only scalar columns
            @test :gradient in propertynames(w)
            @test all(g -> g isa AbstractVector && length(g) == length(coef(lmod)), w.gradient)
            @test isequal(select(w, Not(:gradient)), without)                    # nothing else changes
            # SE is the delta-method form of the gradient
            @test all(isapprox.(w.se, [sqrt(g' * V * g) for g in w.gradient]; atol=1e-8))
        end
        # the gradient sits right after the schema columns and before profile / evaluation columns
        w = second_differences(lmod, df, :x, :w; include_gradients=true)
        nm = propertynames(w)
        @test nm[1:length(Margins.SECOND_DIFFERENCE_COLUMNS)] == collect(Margins.SECOND_DIFFERENCE_COLUMNS)
        @test nm[length(Margins.SECOND_DIFFERENCE_COLUMNS) + 1] == :gradient
        # result-level functions and p_adjust interplay
        ames = population_margins(lmod, df; vars=[:x, :z], scenarios=(g=["a", "b", "c"],), type=:effects)
        @test :gradient ∉ propertynames(second_differences(ames, [:x, :z], :g; warn=false))
        @test :gradient in propertynames(second_differences(ames, [:x, :z], :g; include_gradients=true, warn=false))
        @test :gradient ∉ propertynames(Margins._second_differences_pairwise(ames, :x, :g))
        @test :gradient in propertynames(Margins._second_differences_pairwise(ames, :x, :g; include_gradients=true))
        @test :gradient ∉ propertynames(Margins._second_differences_all_contrasts(ames, :x, :g))
        @test :gradient in propertynames(Margins._second_differences_all_contrasts(ames, :x, :g; include_gradients=true))
        adj = second_differences(ames, [:x, :z], :g; p_adjust=:holm, include_gradients=true, warn=false)
        @test :gradient in propertynames(adj) && :p_adjusted in propertynames(adj)
        @test :gradient ∉ propertynames(Margins._second_differences_local(lmod, df, :x, :w, V))
        @test :gradient in propertynames(Margins._second_differences_local(lmod, df, :x, :w, V; include_gradients=true))
        # the single-row API keeps it
        row = Margins._second_difference(population_margins(lmod, df; vars=:x, scenarios=(t=[0, 1],), type=:effects), :x, :t)
        @test row.gradient isa AbstractVector
        # a table without gradients round-trips through CSV
        io = IOBuffer(); CSV.write(io, second_differences(lmod, df, [:x, :z], :g; warn=false)); @test position(io) > 0
    end

    @testset "profile_margins results and predictions results" begin
        V = vcov(lmod)
        grid = cartesian_grid(t=[0, 1])
        p = profile_margins(lmod, df, grid; type=:effects, vars=[:x])
        @test p.inference.vcov == V
        sd = second_differences(p, :x, :t)                       # the result's own covariance
        @test sd.second_diff[1] ≈ b("x & t") atol=1e-8           # difference of MEMs = interaction coefficient
        @test sd.df[1] == dof_residual(lmod)
        g = p.gradients[2, :] .- p.gradients[1, :]
        @test sd.se[1] ≈ sqrt(g' * V * g)
        @test Margins._second_differences_pairwise(p, :x, :t).second_diff ≈ sd.second_diff
        @test Margins._second_difference(p, :x, :t).second_diff ≈ sd.second_diff[1]
        # robust covariance recorded and used
        pr = profile_margins(lmod, df, grid; type=:effects, vars=[:x], vcov=HC1())
        @test pr.inference.vcov != V
        gr = pr.gradients[2, :] .- pr.gradients[1, :]
        Vhc = CovarianceMatrices.vcov(HC1(), lmod)
        @test second_differences(pr, :x, :t).se[1] ≈ sqrt(gr' * Vhc * gr)
        @test second_differences(pr, :x, :t).df[1] == dof_residual(lmod)   # D1: t with a robust vcov
        # a grid that varies another variable too is rejected, not mixed
        p2 = profile_margins(lmod, df, cartesian_grid(t=[0, 1], g=["a", "b"]); type=:effects, vars=[:x])
        @test_throws ErrorException second_differences(p2, :x, :t)
        # predictions results get a pointer, not a MethodError
        pp = profile_margins(lmod, df, grid; type=:predictions)
        ap = population_margins(lmod, df; scenarios=(t=[0, 1],), type=:predictions)
        for r in (pp, ap)
            err = try second_differences(r, :x, :t); nothing catch e; e end
            @test err isa ErrorException && occursin("type=:effects", err.msg)
        end
        err = try second_differences(ap, [:x], :t); nothing catch e; e end
        @test err isa ErrorException && occursin("PredictionsResult", err.msg)
    end

    @testset "logit: model-level equals the hand-built result" begin
        yb = df.y .> 0
        d4 = copy(df); d4.yb = yb
        gm = glm(@formula(yb ~ x * t + w), d4, Binomial(), LogitLink())
        a = second_differences(gm, d4, :x, :t)
        ames = population_margins(gm, d4; vars=:x, scenarios=(t=[0, 1],), type=:effects)
        h = second_differences(ames, :x, :t)
        @test a.second_diff[1] ≈ h.second_diff[1] && a.se[1] ≈ h.se[1]
        @test a.df[1] == Inf
    end

    @testset "errors" begin
        @test_throws ErrorException second_differences(lmod, df, :x, :nope)
        @test_throws ErrorException second_differences(lmod, df, :x, :w; at=0.0, levels=[0.0, 1.0])
        @test_throws ErrorException second_differences(lmod, df, :x, :t; at=0.5)
        @test_throws ErrorException second_differences(lmod, df, :x, :t; modifier_type=:nonsense)
        @test_throws ErrorException second_differences(lmod, df, :x, :t; levels=[1])
        @test_throws Exception second_differences(lmod, df, :t, :t)
    end
end
