# sd_inference.jl
# Inference for second differences: the covariance used (matrix, Symmetric, function, a
# CovarianceMatrices estimator, a matrix changed in place, the result's own), the df rule,
# confidence intervals and alpha, multiple-testing adjustment, opt-in gradients, and the
# calculations the model cannot support (structural zeros, extrapolation warnings)

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

@testset "second differences: covariance, df, intervals, p_adjust, gradients" begin
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

    @testset "covariance as matrix / Symmetric / function / robust matrix, and the df rule" begin
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
    end

    @testset "bad alpha is an ArgumentError on both forms" begin
        for a in (0.0, 1.0, 1.5, -0.1, 2, "0.05")
            @test_throws ArgumentError second_differences(lmod, df, :x, :t; alpha = a)
            @test_throws ArgumentError second_differences(lmod, df, :x, :w; alpha = a)
        end
        r = population_margins(lmod, df; vars = [:x], scenarios = (t = [0, 1],))
        @test_throws ArgumentError second_differences(r, :x, :t; alpha = 2)
        @test second_differences(lmod, df, :x, :t; alpha = 1 // 10).ci_lower[1] ≈
              second_differences(lmod, df, :x, :t; alpha = 0.1).ci_lower[1]
    end

    @testset "multiple-testing adjustment" begin
        p = [0.01, 0.04, 0.03, 0.005]
        @test Margins._adjust_pvalues(p, :none) == p
        @test Margins._adjust_pvalues(p, :bonferroni) ≈ [0.04, 0.16, 0.12, 0.02]
        @test Margins._adjust_pvalues(p, :holm) ≈ [0.03, 0.06, 0.06, 0.02]
        @test Margins._adjust_pvalues(p, :bh) ≈ [0.02, 0.04, 0.04, 0.02]
        @test Margins._adjust_pvalues([0.5, 0.9], :bonferroni) == [1.0, 1.0]
        @test Margins._adjust_pvalues(Float64[], :holm) == Float64[]
        @test_throws ArgumentError Margins._adjust_pvalues(p, :nope)

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
        @test_throws ArgumentError second_differences(lmod, df, :x, :t; p_adjust=:nope)
        @test_throws ArgumentError second_differences(lmod, df, :x, :w; vcov=vcov(lmod), modifier_type=:continuous, p_adjust=:nope)
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
        @test :gradient ∉ propertynames(second_differences(lmod, df, :x, :w; vcov=V, modifier_type=:continuous))
        @test :gradient in propertynames(second_differences(lmod, df, :x, :w; vcov=V, modifier_type=:continuous, include_gradients=true))
        # the single-row API keeps it
        row = Margins._second_difference(population_margins(lmod, df; vars=:x, scenarios=(t=[0, 1],), type=:effects), :x, :t)
        @test row.gradient isa AbstractVector
        # a table without gradients round-trips through CSV
        io = IOBuffer(); CSV.write(io, second_differences(lmod, df, [:x, :z], :g; warn=false)); @test position(io) > 0
    end
end

@testset "second differences of a result: its df and its own covariance" begin
    Random.seed!(1)
    n = 1500
    df = DataFrame(
        x = randn(n), w = randn(n),
        m = rand([0.0, 1.0], n), z = rand([0.0, 1.0], n),
        g = rand(["a", "b", "c"], n),
    )
    df.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.3 .+ 0.5df.x .+ 0.4df.m .+ 0.6df.x .* df.m .+ 0.2df.z)))
    model = glm(@formula(y ~ x * m + z), df, Binomial(), LogitLink())
    Σ = vcov(model)

    @testset "inference df and covariance are the result's (3.0 step 6; D1: df is the model class's, whatever covariance is used)" begin
        Random.seed!(2)
        m = 300
        d2 = DataFrame(x = randn(m), m = rand([0.0, 1.0], m), z = randn(m))
        d2.y = 0.5 .+ 0.3d2.x .+ 0.4d2.m .+ 0.2d2.x .* d2.m .+ randn(m)
        lmod = lm(@formula(y ~ x * m + z), d2)
        V = vcov(lmod)
        ames = population_margins(lmod, d2; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        # model-based covariance -> t with residual df
        @test Margins._second_differences_pairwise(ames, :x, :m).df[1] == dof_residual(lmod)
        @test Margins._second_difference(ames, :x, :m).df == dof_residual(lmod)
        # a result computed with another covariance: still the linear model's t, and that covariance
        Vr = 1.2 .* V
        ames_r = population_margins(lmod, d2; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects, vcov=_ -> Vr)
        @test ames_r.inference.vcov == Vr
        @test Margins._second_differences_pairwise(ames_r, :x, :m).df[1] == dof_residual(lmod)
        @test Margins._second_differences_pairwise(ames_r, :x, :m).se[1] ≈
              sqrt(1.2) * Margins._second_differences_pairwise(ames, :x, :m).se[1]
        # the result's df= override is what second differences use
        ames_z = population_margins(lmod, d2; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects, df=Inf)
        @test Margins._second_differences_pairwise(ames_z, :x, :m).df[1] == Inf
        # non-linear model -> z
        @test Margins._second_differences_pairwise(
            population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects),
            :x, :m).df[1] == Inf
    end

    @testset "local derivative accepts Symmetric and function vcov" begin
        r1 = second_differences(model, df, :x, :m; vcov=Σ, modifier_type=:continuous, at=0.5)
        r2 = second_differences(model, df, :x, :m; vcov=Symmetric(Σ), modifier_type=:continuous, at=0.5)
        r3 = second_differences(model, df, :x, :m; vcov=m -> Symmetric(vcov(m)), modifier_type=:continuous, at=0.5)
        @test r1.second_diff ≈ r2.second_diff ≈ r3.second_diff
        @test r1.se ≈ r2.se ≈ r3.se
    end

    @testset "the covariance is the result's own; a matrix argument is an error" begin
        a = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        @test a.inference.vcov == Σ
        p = profile_margins(model, df, DataFrame(m=[0.0, 1.0]); type=:effects, vars=[:x])
        @test p.inference.vcov == Σ                      # profile results carry it too
        @test Margins._second_differences_pairwise(p, :x, :m).se[1] > 0
        err = try second_differences(a, :x, :m, Σ); nothing catch e; e end
        @test err isa ArgumentError && occursin("second_differences(result, variables, modifier)", err.msg)
        @test_throws ArgumentError second_differences(a, [:x], :m, Σ)
        err = try second_differences(a, :x, :m; vcov=Σ); nothing catch e; e end
        @test err isa ArgumentError && occursin("vcov", err.msg)
        @test_throws MethodError Margins._second_differences_pairwise(a, :x, :m, Σ)
    end
end

@testset "second differences: covariance changed in place, df rule, alpha" begin
    Random.seed!(21)
    n = 600
    df = DataFrame(x = randn(n), w = randn(n), z = rand([false, true], n), t = rand([0, 1], n),
                   g = rand(["a", "b", "c"], n))
    df.y = 0.2 .+ 0.4df.x .+ 0.3df.w .+ 0.5df.z .+ 0.6df.x .* df.z .- 0.4df.x .* df.w .+ randn(n)
    df.yb = (0.3 .+ 0.6df.x .+ 0.5df.z .+ 0.4df.x .* df.z .+ randn(n)) .> 0
    lmod = lm(@formula(y ~ x * z + x * w + t + g), df)
    lg = glm(@formula(yb ~ x * z + w), df, Binomial(), LogitLink())

    @testset "a vcov matrix changed in place is respected (discrete and local paths)" begin
        M = Matrix(vcov(lg))
        s1 = second_differences(lg, df, :x, :z; vcov = M).se
        M .*= 4
        s2 = second_differences(lg, df, :x, :z; vcov = M).se
        @test s2 ≈ 2 .* s1
        M2 = Matrix(vcov(lg))
        l1 = second_differences(lg, df, :x, :w; vcov = M2).se
        M2 .*= 4
        @test second_differences(lg, df, :x, :w; vcov = M2).se ≈ 2 .* l1
    end

    @testset "df follows the same rule as population_margins" begin
        # any covariance function: the model class's df (3.0 step 5, D1), as population_margins
        sd_fun = second_differences(lmod, df, :x, :z; vcov = m -> GLM.vcov(m))
        pm = population_margins(lmod, df; vars = :x, type = :effects, vcov = m -> GLM.vcov(m))
        @test sd_fun.df[1] == dof_residual(lmod) == pm.inference.df
        @test second_differences(lmod, df, :x, :w; vcov = m -> GLM.vcov(m)).df[1] == dof_residual(lmod)
    end

    @testset "result-level form: a different vcov is passed to the entry point" begin
        ames = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects)
        V = 4 .* Matrix(vcov(lg))
        ames4 = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects, vcov = V)
        @test second_differences(ames4, :x, :z).se[1] ≈ 2 * second_differences(ames, :x, :z).se[1]
    end
end

@testset "second differences: calculations the model cannot support" begin
    Random.seed!(11)
    n = 1200
    df = DataFrame(x = randn(n), z = randn(n), t = rand([0, 1], n), w = randn(n), u = randn(n),
                   g = rand(["a", "b", "c"], n))
    df.y = 0.5 .+ 0.4df.x .+ 0.3df.z .+ 0.5df.t .+ 0.2df.w .+ 0.6df.x .* df.w .+ randn(n)
    df.yb = (0.4 .+ 0.8df.x .+ 0.5df.t .+ randn(n)) .> 0
    # x*w is the only interaction; t, z, g and u have none with x; u is not in the model
    lmod = lm(@formula(y ~ x * w + z + t + g), df)
    structural(sd) = isnan.(sd.p_value)

    @testset "no interaction in an identity-link model: NaN inference and one warning" begin
        sd = @test_logs (:warn, r"1 second difference\(s\) \(x × t\) are zero by construction") second_differences(lmod, df, :x, :t)
        @test sd.second_diff[1] == 0.0 && sd.se[1] == 0.0
        @test all(isnan, (sd.statistic[1], sd.p_value[1], sd.ci_lower[1], sd.ci_upper[1]))
        @test sd.significant == [false]
        # same for a local derivative
        ld = @test_logs (:warn, r"1 second difference\(s\) \(z × w\) are zero by construction") second_differences(lmod, df, :z, :w)
        @test isnan(ld.p_value[1]) && ld.second_diff[1] == 0.0 && ld.significant == [false]
        # warn=false silences it without changing the numbers
        quiet = @test_logs second_differences(lmod, df, :x, :t; warn=false)
        @test isequal(quiet, sd)
    end

    @testset "the warning is aggregated and says why" begin
        # three rows (z × g pairs) -> one warning that names the pair and counts the rows
        sd = @test_logs (:warn, r"3 second difference\(s\) \(z × g\) are zero by construction") second_differences(lmod, df, :z, :g)
        @test all(structural(sd))
        # several focal variables: one warning listing every affected pair
        sd2 = @test_logs (:warn, r"\(z × t, x × t\)|\(x × t, z × t\)") second_differences(lmod, df, [:x, :z], :t)
        @test count(structural(sd2)) == 2
        # a mix: the interaction row is normal, the structural row is flagged
        mixed = @test_logs (:warn, r"1 second difference\(s\) \(z × w\)") second_differences(lmod, df, [:x, :z], :w)
        @test mixed.variable == [:x, :z]
        @test !isnan(mixed.p_value[1]) && isnan(mixed.p_value[2])
        # a modifier outside the model gets that explanation
        out = @test_logs (:warn, r"modifier :u does not enter the model") second_differences(lmod, df, :x, :u)
        @test isnan(out.p_value[1])
        inm = @test_logs (:warn, r"does not let the effect of the focal variable depend on the modifier") second_differences(lmod, df, :x, :t)
        @test isnan(inm.p_value[1])
    end

    @testset "a real interaction is never flagged, even when its coefficient is ~0" begin
        sd = @test_logs second_differences(lmod, df, :x, :w)
        @test !isnan(sd.p_value[1]) && sd.se[1] > 0
        # true x:t effect is 0. The noise has its own RNG: a bare randn(n) here replays the
        # testset's seeded stream, which gave df.x again, so y0 was exactly linear in the
        # regressors and the interaction was 0 ± 1e-17 (it passed or failed on rounding)
        d0 = copy(df); d0.y0 = 0.5 .+ 0.4d0.x .+ 0.3d0.z .+ 0.5d0.t .+ randn(Random.MersenneTwister(12), n)
        m0 = lm(@formula(y0 ~ x * t + z + w + g), d0)
        s0 = @test_logs second_differences(m0, d0, :x, :t)                          # estimated ≈ 0, but estimable
        @test !isnan(s0.p_value[1]) && s0.se[1] > 0 && abs(s0.second_diff[1]) < 5 * s0.se[1]
    end

    @testset "a nonlinear model without an interaction term is reported normally" begin
        gm = glm(@formula(yb ~ x + t + w), df, Binomial(), LogitLink())
        sd = @test_logs second_differences(gm, df, :x, :t)           # varies through the link: no warning
        @test !isnan(sd.p_value[1]) && sd.se[1] > 0 && sd.second_diff[1] != 0
        @test sd.significant[1] == (sd.p_value[1] < 0.05)
        # ... but a modifier that is not in the model cannot matter, even in a logit
        out = @test_logs (:warn, r"modifier :u does not enter the model") second_differences(gm, df, :x, :u)
        @test isnan(out.p_value[1]) && out.second_diff[1] == 0.0
    end

    @testset "p_adjust leaves structural zeros out of the family" begin
        raw = second_differences(lmod, df, [:x, :z], :w; warn=false)          # x × w tested, z × w structural
        adj = second_differences(lmod, df, [:x, :z], :w; p_adjust=:bonferroni, warn=false)
        @test adj.p_adjusted[1] ≈ adj.p_value[1]                              # family of 1: no inflation
        @test isnan(adj.p_adjusted[2]) && adj.significant == [raw.p_value[1] < 0.05, false]
    end

    @testset "result-level function behaves the same" begin
        ames = population_margins(lmod, df; vars=[:x, :z], scenarios=(t=[0, 1],), type=:effects)
        sd = @test_logs (:warn, r"2 second difference\(s\)") second_differences(ames, [:x, :z], :t)
        @test all(structural(sd)) && all(sd.significant .== false)
        @test_logs second_differences(ames, [:x, :z], :t; warn=false)
        # the generic wording (no model to inspect)
        @test_logs (:warn, r"does not let the effect") second_differences(ames, :x, :t)
        # profile results too
        p = profile_margins(lmod, df, cartesian_grid(t=[0, 1]); type=:effects, vars=[:x])
        sp = @test_logs (:warn, r"zero by construction") second_differences(p, :x, :t)
        @test isnan(sp.p_value[1])
        # with include_gradients the gradient is exactly zero for a structural row
        sg = second_differences(ames, :x, :t; include_gradients=true, warn=false)
        @test all(iszero, sg.gradient[1]) || maximum(abs, sg.gradient[1]) < 1e-12
    end

    @testset "values outside the observed range warn about extrapolation" begin
        lo, hi = extrema(df.w)
        # levels
        sd = @test_logs (:warn, r"Value\(s\) 10\.0 of :w lie outside the observed range") second_differences(lmod, df, :x, :w; levels=[0.0, 10.0])
        @test !isnan(sd.p_value[1])
        # at: a number, a vector, a NamedTuple
        @test_logs (:warn, r"Value\(s\) 50\.0 of :w") second_differences(lmod, df, :x, :w; at=50.0)
        @test_logs (:warn, r"Value\(s\) -40\.0, 40\.0 of :w") second_differences(lmod, df, :x, :w; at=[-40.0, 0.0, 40.0])
        @test_logs (:warn, r"Value\(s\) 99\.0 of :z") second_differences(lmod, df, :x, :w; at=(w=0.0, z=99.0))
        # profile values
        @test_logs (:warn, r"Value\(s\) 30\.0 of :z") second_differences(lmod, df, :x, :w; profile=(z=30.0,))
        # inside the range, categorical values, the default at=:mean: no warning
        @test_logs second_differences(lmod, df, :x, :w; levels=[lo, hi])
        @test_logs second_differences(lmod, df, :x, :w; at=0.0)
        @test_logs second_differences(lmod, df, :x, :w)
        @test_logs (:warn, r"zero by construction") second_differences(lmod, df, :x, :g; levels=["a", "c"])
        # warn=false silences extrapolation warnings too
        @test_logs second_differences(lmod, df, :x, :w; levels=[0.0, 10.0], warn=false)
    end
end
