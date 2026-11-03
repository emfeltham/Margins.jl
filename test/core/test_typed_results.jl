# test_typed_results.jl
# Typed results (Margins 3.0, step 6): `result.inference` (Σ, df, alpha) is what everything
# computed from a result reads, so `contrast` cannot pair the result's df with another Σ (B9),
# and `result.context` keeps the context columns in grid / user order, never Dict hash order (B10).

using Test
using Margins
using DataFrames, CategoricalArrays, GLM
using Distributions: TDist, Normal, ccdf, quantile
using LinearAlgebra: Symmetric
using Random
import CovarianceMatrices
using CovarianceMatrices: HC1

@testset "typed results: inference and context" begin
    Random.seed!(606)
    n = 300
    d = DataFrame(x = randn(n), z = randn(n), w = randn(n), b = rand(n) .< 0.5,
                  g = categorical(rand(["a", "b", "c"], n)), h = categorical(rand(["p", "q"], n)),
                  k = categorical(rand(["u", "v"], n)))
    d.y = 0.5 .+ 0.4 .* d.x .- 0.2 .* d.z .+ 0.3 .* d.x .* d.z .+ 0.2 .* d.b .+ randn(n)
    d.yb = rand(n) .< 1 ./ (1 .+ exp.(-(0.2 .+ 0.6 .* d.x .- 0.3 .* d.z)))
    lmod = lm(@formula(y ~ x * z + w + b + g + h + k), d)
    lg = glm(@formula(yb ~ x * z + w + b + g), d, Binomial(), LogitLink())

    @testset "Inference is populated from the call" begin
        # lm: t with the residual df, the model's covariance, alpha
        r = population_margins(lmod, d; vars = [:x, :z])
        @test r.inference isa Inference
        @test r.inference.df == dof_residual(lmod)
        @test r.inference.vcov == vcov(lmod)
        @test r.inference.alpha == 0.05
        @test DataFrame(r).df == fill(dof_residual(lmod), 2)
        # Gaussian-identity glm: also t
        mg = glm(@formula(y ~ x * z + w), d, Normal(), IdentityLink())
        @test population_margins(mg, d; vars = [:x]).inference.df == dof_residual(mg)
        # logit glm: z
        rl = population_margins(lg, d; vars = [:x], alpha = 0.1)
        @test rl.inference.df == Inf && rl.inference.vcov == vcov(lg) && rl.inference.alpha == 0.1
        @test DataFrame(rl).df == [Inf]
        # HC1 on lm: the robust Σ, and still t(n − k) (decision D1)
        rh = population_margins(lmod, d; vars = [:x], vcov = HC1())
        @test rh.inference.vcov ≈ CovarianceMatrices.vcov(HC1(), lmod)
        @test rh.inference.df == dof_residual(lmod)
        @test rh.standard_errors[1] ≈ sqrt(rh.gradients[1, :]' * rh.inference.vcov * rh.gradients[1, :])
        # df= override, including Inf (z) on an lm
        rz = population_margins(lmod, d; vars = [:x], df = Inf)
        @test rz.inference.df == Inf && DataFrame(rz).df == [Inf]
        @test population_margins(lmod, d; vars = [:x], df = 12).inference.df == 12.0
        # profile results and predictions carry it too
        p = profile_margins(lmod, d, means_grid(d); vars = [:x], vcov = HC1(), alpha = 0.2, df = 40)
        @test p.inference.vcov ≈ CovarianceMatrices.vcov(HC1(), lmod)
        @test p.inference.df == 40 && p.inference.alpha == 0.2
        pp = population_margins(lg, d; type = :predictions)
        @test pp.inference.df == Inf && pp.inference.vcov == vcov(lg)
        # a result owns its matrix: changing one result's Σ does not change another's
        r2 = population_margins(lmod, d; vars = [:x, :z])
        r2.inference.vcov[1, 1] += 1
        @test r.inference.vcov == vcov(lmod)
    end

    @testset "B9: contrast reads Σ, df and alpha from the result" begin
        # a robust-covariance lm result at alpha = 0.1
        r = profile_margins(lmod, d, cartesian_grid(x = [-1.0, 1.0]); type = :predictions,
                            vcov = HC1(), alpha = 0.1)
        V, dfr = r.inference.vcov, r.inference.df
        @test dfr == dof_residual(lmod)
        cr = contrast(r, 2, 1)
        # hand computation from `inference`
        g = r.gradients[2, :] .- r.gradients[1, :]
        se = sqrt(g' * V * g)
        est = r.estimates[2] - r.estimates[1]
        crit = quantile(TDist(dfr), 1 - 0.1 / 2)
        @test cr.contrast == est
        @test cr.se ≈ se rtol = 1e-14
        @test cr.df == dfr
        @test cr.statistic ≈ est / se
        @test cr.p_value ≈ 2 * ccdf(TDist(dfr), abs(est / se))
        @test cr.ci_lower ≈ est - crit * se && cr.ci_upper ≈ est + crit * se   # the result's alpha
        @test cr.metadata[:alpha] == 0.1
        # the robust Σ is used, not the model-based one
        @test !(cr.se ≈ sqrt(g' * vcov(lmod) * g))
        # an explicit alpha overrides the result's
        c95 = contrast(r, 2, 1; alpha = 0.05)
        @test c95.ci_upper - c95.ci_lower ≈ 2 * quantile(TDist(dfr), 0.975) * se
        # the same contrast by column values
        cs = contrast(r, (x = 1.0,), (x = -1.0,))
        @test (cs.contrast, cs.se, cs.df, cs.ci_lower) == (cr.contrast, cr.se, cr.df, cr.ci_lower)
        @test cs.row1 == 2 && cs.row2 == 1 && cs.metadata[:source] === :result_namedtuple
        # z inference for a df = Inf result
        rz = profile_margins(lg, d, cartesian_grid(x = [-1.0, 1.0]); type = :predictions)
        cz = contrast(rz, 2, 1)
        gz = rz.gradients[2, :] .- rz.gradients[1, :]
        @test cz.df == Inf && cz.se ≈ sqrt(gz' * vcov(lg) * gz)
        @test cz.p_value ≈ 2 * ccdf(Normal(), abs(cz.statistic))
        @test occursin("z:", sprint(show, cz)) && occursin("90", sprint(show, cr))
        # effects by variable and scenario columns
        e = population_margins(lmod, d; vars = [:x], scenarios = (z = [0.0, 1.0],))
        ce = contrast(e, (variable = "x", at_z = 1.0), (variable = "x", at_z = 0.0))
        @test ce.contrast ≈ coef(lmod)[findfirst(==("x & z"), coefnames(lmod))] rtol = 1e-10
        # positional Σ (incl. Symmetric, the B9 rejection) is gone, with a pointer
        @test_throws ArgumentError contrast(r, 2, 1, Matrix(vcov(lmod)))
        @test_throws ArgumentError contrast(r, 2, 1, Symmetric(vcov(lmod)))
    end

    @testset "B10: context columns in grid order and user order" begin
        # Column names chosen so that Dict hash order differs from the requested order
        hash_order(names_) = collect(keys(NamedTuple(Dict(nm => 1 for nm in names_))))
        @test hash_order([:z, :x, :b]) != [:z, :x, :b] || hash_order([:k, :h, :g]) != [:k, :h, :g]

        # profile: the grid's column order
        grid = DataFrame(z = [0.0, 1.0], x = [0.5, 0.5], b = [false, true])
        p = profile_margins(lmod, d, grid; type = :predictions)
        @test collect(keys(p.context)) == [:z, :x, :b]
        @test p.context.kinds == [:grid, :grid, :grid]
        @test names(DataFrame(p))[1:4] == ["z", "x", "b", "type"]
        pe = profile_margins(lmod, d, grid; vars = [:w])
        @test collect(keys(pe.context)) == [:z, :x, :b]
        @test names(DataFrame(pe))[1:3] == ["z", "x", "b"]
        @test names(DataFrame(pe; format = :standard))[1:3] == ["z", "x", "b"]
        @test p.context[:z] == [0.0, 1.0] && p.context[:b] == [false, true]
        # the response and filled-in typical values are not context columns
        @test collect(keys(profile_margins(lmod, d, DataFrame(x = [0.0], y = [1.0]); vars = [:w]).context)) == [:x]

        # population: groups, then scenarios, each in the order given
        r = population_margins(lmod, d; vars = [:w], groups = [:k, :h, :g], scenarios = (z = [0.0, 1.0], x = [0.0], b = [false, true]))
        @test collect(keys(r.context)) == [:k, :h, :g, :z, :x, :b]
        @test r.context.kinds == [:group, :group, :group, :scenario, :scenario, :scenario]
        @test Margins.context_columns(r) == ([:k, :h, :g], [:z, :x, :b])
        @test names(DataFrame(r))[1:7] == ["over_k", "over_h", "over_g", "at_z", "at_x", "at_b", "type"]
        r2 = population_margins(lmod, d; vars = [:w], groups = [:g, :k, :h])
        @test names(DataFrame(r2))[1:3] == ["over_g", "over_k", "over_h"]
        # rows carry their own context values
        df_r = DataFrame(r)
        @test all(df_r.over_k .== r.context[:k]) && all(df_r.at_b .== r.context[:b])
        # `show` lists them in the same order as the DataFrame
        shown = sprint(show, r)
        header = only(filter(l -> occursin("Variable", l), split(shown, '\n')))
        pos = [findfirst(c, header).start for c in ("over_k", "over_h", "over_g", "at_z", "at_x", "at_b")]
        @test issorted(pos)
        shownp = sprint(show, p)
        hp = split(shownp, '\n')[3]
        @test findfirst("z", hp).start < findfirst("x", hp).start < findfirst("b", hp).start
        # nested groups keep the order written
        rn = population_margins(lmod, d; type = :predictions, groups = :k => :h)
        @test collect(keys(rn.context)) == [:k, :h]
        # a plain AME has no context
        @test isnothing(population_margins(lmod, d; vars = [:x]).context)
    end

    @testset "second differences use the result's inference" begin
        a = population_margins(lmod, d; vars = [:x], scenarios = (z = [0.0, 1.0],), vcov = HC1(), df = 25, alpha = 0.2)
        sd = second_differences(a, :x, :z)
        g = a.gradients[2, :] .- a.gradients[1, :]
        @test sd.se[1] ≈ sqrt(g' * a.inference.vcov * g)
        @test sd.df[1] == 25
        @test sd.ci_upper[1] - sd.ci_lower[1] ≈ 2 * quantile(TDist(25), 0.9) * sd.se[1]
    end
end
