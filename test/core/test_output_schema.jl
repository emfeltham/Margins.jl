# test_output_schema.jl - Margins 3.0 output schema:
#   * `statistic` + `df` (t with residual df for linear models, z otherwise)
#   * `over_` prefix for group columns
#   * one second-differences schema for every modifier type and for second_differences_at

using Test, Random, DataFrames, CategoricalArrays, GLM, Statistics, Distributions
using CovarianceMatrices
using MixedModels
using Margins

@testset "Margins 3.0 output schema" begin
    Random.seed!(20261016)
    n = 80
    d = DataFrame(
        x = randn(n),
        z = randn(n),
        t = rand(0:1, n),
        g = categorical(rand(["a", "b", "c"], n)),
        h = categorical(rand(["p", "q"], n)),
        cl = rand(1:8, n),
    )
    d.y = 1 .+ 0.8 .* d.x .- 0.4 .* d.z .+ 0.5 .* d.x .* d.t .+ randn(n)
    d.yb = Float64.(rand(n) .< 1 ./ (1 .+ exp.(-(0.3 .+ 0.9 .* d.x))))

    @testset "linear model: t with residual df" begin
        m = lm(@formula(y ~ x + z), d)
        dfr = dof_residual(m)
        @test Margins._inference_df(m) == dfr

        # The AME of x in an additive lm is its coefficient: inference must equal coeftable
        res = population_margins(m, d; vars=[:x, :z])
        out = DataFrame(res)
        @test res.inference.df == dfr
        @test all(out.df .== dfr)
        ct = coeftable(m)
        for (row, name) in enumerate(["x", "z"])
            j = findfirst(==(name), ct.rownms)
            @test out.estimate[row] ≈ ct.cols[1][j] rtol = 1e-10
            @test out.se[row] ≈ ct.cols[2][j] rtol = 1e-10
            @test out.statistic[row] ≈ ct.cols[3][j] rtol = 1e-8
            @test out.p_value[row] ≈ ct.cols[4][j] rtol = 1e-8
            @test out.ci_lower[row] ≈ ct.cols[5][j] rtol = 1e-8
            @test out.ci_upper[row] ≈ ct.cols[6][j] rtol = 1e-8
        end
        # Closed form, and strictly wider than the z interval
        tcrit = quantile(TDist(dfr), 0.975)
        @test out.ci_upper[1] - out.ci_lower[1] ≈ 2 * tcrit * out.se[1]
        @test tcrit > quantile(Normal(), 0.975)

        # Predictions and profile results use the same df
        aap = DataFrame(population_margins(m, d; type=:predictions))
        @test aap.df[1] == dfr
        @test aap.p_value[1] ≈ 2 * ccdf(TDist(dfr), abs(aap.estimate[1] / aap.se[1]))
        prof = DataFrame(profile_margins(m, d, cartesian_grid(x=[-1.0, 1.0]); type=:effects, vars=[:x]))
        @test all(prof.df .== dfr)
        @test hasproperty(prof, :statistic) && hasproperty(prof, :p_value)
        @test prof.ci_upper[1] - prof.ci_lower[1] ≈ 2 * tcrit * prof.se[1]
        pp = DataFrame(profile_margins(m, d, cartesian_grid(x=[-1.0, 1.0]); type=:predictions))
        @test all(pp.df .== dfr)
        @test pp.ci_upper[1] - pp.ci_lower[1] ≈ 2 * tcrit * pp.se[1]

        # Link scale coincides with response scale (identity link)
        out_link = DataFrame(population_margins(m, d; vars=[:x], scale=:link))
        @test out_link.df[1] == dfr

        # Gaussian identity GLM follows the same rule
        mg = glm(@formula(y ~ x + z), d, Normal(), IdentityLink())
        @test Margins._inference_df(mg) == dof_residual(mg)
        @test DataFrame(population_margins(mg, d; vars=[:x])).df[1] == dof_residual(mg)

        # Contrasts inherit the df
        pr = profile_margins(m, d, cartesian_grid(x=[-1.0, 1.0]); type=:predictions)
        cr = contrast(pr, 2, 1)
        @test cr.df == dfr
        @test cr.statistic ≈ cr.contrast / cr.se
        @test cr.p_value ≈ 2 * ccdf(TDist(dfr), abs(cr.statistic))
        @test cr.ci_upper - cr.ci_lower ≈ 2 * tcrit * cr.se
        crd = contrast(pr, (x=1.0,), (x=-1.0,))   # the same rows, by their grid values
        @test crd.df == dfr && crd.row1 == 2 && crd.row2 == 1
        @test crd.p_value == cr.p_value
        @test occursin("t (df = $(Int(dfr)))", sprint(show, cr))
        @test names(DataFrame(cr))[1:7] == ["contrast", "se", "statistic", "df", "p_value", "ci_lower", "ci_upper"]

        # :stata format keeps Stata's labels
        st = DataFrame(res; format=:stata)
        @test hasproperty(st, :t) && hasproperty(st, :P_t)
    end

    @testset "nonlinear / other models: z (df = Inf)" begin
        mb = glm(@formula(yb ~ x + z), d, Binomial(), LogitLink())
        @test Margins._inference_df(mb) == Inf
        out = DataFrame(population_margins(mb, d; vars=[:x]))
        @test out.df[1] == Inf
        @test out.statistic[1] ≈ out.estimate[1] / out.se[1]
        @test out.p_value[1] ≈ 2 * ccdf(Normal(), abs(out.statistic[1]))
        @test out.ci_upper[1] - out.ci_lower[1] ≈ 2 * quantile(Normal(), 0.975) * out.se[1]
        # link scale of a logit is still z
        @test DataFrame(population_margins(mb, d; vars=[:x], scale=:link)).df[1] == Inf
        st = DataFrame(population_margins(mb, d; vars=[:x]); format=:stata)
        @test hasproperty(st, :z) && hasproperty(st, :P_z)
        # no p-value underflow for a huge statistic
        @test Margins.compute_p_value(30.0, Inf) > 0
        @test Margins.compute_p_value(30.0, 30.0) > 0

        mm = fit(MixedModel, @formula(y ~ x + z + (1 | cl)), d; progress=false)
        @test Margins._inference_df(mm) == Inf
        @test DataFrame(population_margins(mm, d; vars=[:x])).df[1] == Inf
    end

    # 3.0 step 5 (decision D1): df is a property of the model class, so a linear model keeps
    # t(dof_residual) with robust, clustered and custom vcov (Stata `regress, vce(robust)` +
    # `margins`, R `coeftest(m, vcovHC(m))`); before, any vcov other than GLM.vcov gave z.
    @testset "robust / clustered / custom vcov: t for a linear model (D1)" begin
        m = lm(@formula(y ~ x + z), d)
        dfr = dof_residual(m)
        for v in (HC1(), CR1(d.cl))
            r = population_margins(m, d; vars=[:x], vcov=v)
            out = DataFrame(r)
            @test out.df[1] == dfr
            @test out.p_value[1] ≈ 2 * ccdf(TDist(dfr), abs(out.statistic[1]))
        end
        @test DataFrame(population_margins(m, d; vars=[:x], vcov = mm -> GLM.vcov(mm))).df[1] == dfr
        @test DataFrame(population_margins(m, d; vars=[:x], vcov = 2 .* vcov(m))).df[1] == dfr
        # z on request
        @test DataFrame(population_margins(m, d; vars=[:x], vcov=HC1(), df=Inf)).df[1] == Inf
    end

    @testset "group columns: over_ prefix" begin
        m = lm(@formula(y ~ x + z + g + h), d)
        plain = DataFrame(population_margins(m, d; vars=[:x], groups=:g))
        @test :over_g in propertynames(plain) && :g ∉ propertynames(plain)
        @test names(plain)[1] == "over_g"

        binned = DataFrame(population_margins(m, d; vars=[:x], groups=(:z, 3)))
        @test :over_z in propertynames(binned) && :z ∉ propertynames(binned)

        nested = DataFrame(population_margins(m, d; vars=[:x], groups=:g => :h))
        @test :over_g in propertynames(nested) && :over_h in propertynames(nested)

        both = population_margins(m, d; type=:predictions, groups=:g, scenarios=(x=[0.0, 1.0],))
        bdf = DataFrame(both)
        @test names(bdf)[1:3] == ["over_g", "at_x", "type"]
        shown = sprint(show, both)
        @test occursin("over_g", shown) && occursin("at_x", shown)
        @test occursin("over_g", sprint(show, population_margins(m, d; vars=[:x], groups=:g)))

        # contrast() addresses rows by the prefixed names
        r = population_margins(m, d; vars=[:x], groups=:g)
        cr = contrast(r, (over_g="b",), (over_g="a",))
        @test cr.df == dof_residual(m)

        # profile results keep bare grid names
        prof = DataFrame(profile_margins(m, d, cartesian_grid(g=["a", "b"]); type=:predictions))
        @test :g in propertynames(prof)
    end

    @testset "second differences: one schema for every modifier type" begin
        m = lm(@formula(y ~ x * t + x * z + g), d)
        cols = collect(Margins.SECOND_DIFFERENCE_COLUMNS)

        ame_bin = population_margins(m, d; vars=[:x, :g], scenarios=(t=[0, 1],))
        sd_bin = second_differences(ame_bin, [:x, :g], :t)       # binary path
        ame_3 = population_margins(m, d; vars=[:x], scenarios=(z=[-1.0, 0.0, 1.0],))
        sd_cont = second_differences(ame_3, :x, :z)                # continuous
        sd_cat = second_differences(ame_3, :x, :z; modifier_type=:categorical)
        sd_pw = Margins._second_differences_pairwise(ame_bin, :x, :t)      # binary via pairwise
        sd_all = Margins._second_differences_all_contrasts(ame_bin, :g, :t)
        sd_at = Margins._second_differences_local(m, d, :x, :z, vcov(m); at = :mean)
        sd_at_f = Margins._second_differences_local(m, d, :x, :z, GLM.vcov)

        for sd in (sd_bin, sd_cont, sd_cat, sd_pw, sd_all)
            @test propertynames(sd) == cols
        end
        @test propertynames(sd_at)[1:length(cols)] == cols
        @test propertynames(sd_at)[length(cols)+1:end] == [:eval_point, :delta_used]

        @test all(sd_bin.modifier_type .== :binary)
        @test all(sd_pw.modifier_type .== :binary)
        @test all(sd_cont.modifier_type .== :continuous)
        @test all(sd_cat.modifier_type .== :categorical)
        @test all(sd_at.modifier_type .== :continuous)

        dfr = dof_residual(m)
        for sd in (sd_bin, sd_cont, sd_cat, sd_pw, sd_all, sd_at, sd_at_f)
            @test all(sd.df .== dfr)
            ok = sd.se .> 1e-8  # contrasts of g do not vary with t (no g:t term): se = 0
            @test all(sd.statistic[ok] .≈ sd.second_diff[ok] ./ sd.se[ok])
            @test all(sd.p_value[ok] .≈ 2 .* ccdf.(TDist(dfr), abs.(sd.statistic[ok])))
            @test all(sd.significant .== (sd.p_value .< 0.05))
        end
        # binary path and pairwise path agree
        @test sd_bin.second_diff[1] ≈ sd_pw.second_diff[1]
        @test sd_at.modifier_level2[1] - sd_at.modifier_level1[1] ≈ 2 * sd_at.delta_used[1]
        # at = :observed (the default) has no single evaluation point: same columns, NaN point and levels
        sd_obs = Margins._second_differences_local(m, d, :x, :z, vcov(m))
        @test propertynames(sd_obs) == propertynames(sd_at)
        @test isnan(sd_obs.eval_point[1]) && isnan(sd_obs.modifier_level1[1]) && sd_obs.delta_used[1] > 0

        # single-row API
        row = Margins._second_difference(ame_bin, :x, :t)
        @test keys(row) == (Tuple(cols)..., :gradient)   # the single-row API keeps the gradient
        @test row.modifier_type == :binary && row.df == dfr

        # logit: z
        mb = glm(@formula(yb ~ x * t), d, Binomial(), LogitLink())
        sdb = second_differences(population_margins(mb, d; vars=[:x], scenarios=(t=[0, 1],)), :x, :t)
        @test sdb.df[1] == Inf
        @test sdb.p_value[1] ≈ 2 * ccdf(Normal(), abs(sdb.statistic[1]))
    end
end
