# test_second_differences_review.jl
# Regression tests from the second review: covariance handling, df rule, `at` forms, modifier checks, warnings

using Test
using Margins
using DataFrames
using GLM
using CategoricalArrays
using Random

@testset "second differences: review fixes" begin
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
        sd_default = second_differences(lmod, df, :x, :z)
        @test sd_default.df[1] == dof_residual(lmod)
        # any covariance function: the model class's df (3.0 step 5, D1), as population_margins
        sd_fun = second_differences(lmod, df, :x, :z; vcov = m -> GLM.vcov(m))
        pm = population_margins(lmod, df; vars = :x, type = :effects, vcov = m -> GLM.vcov(m))
        @test sd_fun.df[1] == dof_residual(lmod) == pm.inference.df
        @test second_differences(lmod, df, :x, :w; vcov = m -> GLM.vcov(m)).df[1] == dof_residual(lmod)
        # the matrix equal to vcov(model) is model-based
        @test second_differences(lmod, df, :x, :z; vcov = Matrix(GLM.vcov(lmod))).df[1] == dof_residual(lmod)
    end

    @testset "`at` forms" begin
        a1 = second_differences(lmod, df, :x, :w; at = [0.0, 0.5, 1.0])
        a2 = second_differences(lmod, df, :x, :w; at = 0.0:0.5:1.0)
        @test a1.second_diff ≈ a2.second_diff
        t = second_differences(lmod, df, :x, :w; at = (w = [0.0, 1.0], t = 1))
        @test nrow(t) == 2 && t.eval_point == [0.0, 1.0]
    end

    @testset "profile must not contain the modifier" begin
        @test_throws ErrorException second_differences(lmod, df, :x, :w; profile = (w = 1.0,))
        @test_throws ErrorException second_differences(lmod, df, :x, :z; profile = (z = true,))
        err = try second_differences(lmod, df, :x, :w; profile = (w = 1.0,)) catch e; sprint(showerror, e) end
        @test occursin("contains the modifier", err)
    end

    @testset "a Bool modifier cannot be forced continuous" begin
        err = try second_differences(lg, df, :x, :z; modifier_type = :continuous) catch e; sprint(showerror, e) end
        @test occursin("Bool", err)
        @test second_differences(lg, df, :x, :z).modifier_type == [:binary]
    end

    @testset "integer codes stored as floats are flagged; warn=false silences it" begin
        d = copy(df); d.k = Float64.(rand(1:3, n))
        m = lm(@formula(y ~ x * k + w), d)
        @test_logs (:warn, r"takes only 3 integer values") second_differences(m, d, :x, :k)
        @test_logs second_differences(m, d, :x, :k; warn = false)
        @test_logs second_differences(m, d, :x, :k; at = :mean)
    end

    @testset "warn=false silences every warning" begin
        d = copy(df); d.k = rand([10.0, 20.0], n)
        m = lm(@formula(y ~ x * k + w), d)
        ames = population_margins(m, d; vars = :x, scenarios = (k = [10.0, 20.0],), type = :effects)
        @test_logs (:warn, r"treated as :binary") second_differences(ames, :x, :k)
        @test_logs second_differences(ames, :x, :k; warn = false)
        @test_logs second_differences(m, d, :x, :k; levels = [10.0, 20.0], delta = 0.1, warn = false)
    end

    @testset "binary route warns like the pairwise route" begin
        d = copy(df); d.k = rand([10.0, 20.0], n)
        d.cg = categorical(d.g)
        m = lm(@formula(y ~ x * k + cg * k + w), d)
        ames = population_margins(m, d; vars = :cg, scenarios = (k = [10.0, 20.0],), type = :effects)
        @test_logs (:warn, r"treated as :binary") second_differences(ames, :cg, :k)
    end

    @testset "result-level form: a different vcov is passed to the entry point" begin
        ames = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects)
        V = 4 .* Matrix(vcov(lg))
        ames4 = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects, vcov = V)
        @test second_differences(ames4, :x, :z).se[1] ≈ 2 * second_differences(ames, :x, :z).se[1]
        @test_throws ArgumentError second_differences(ames, :x, :z; vcov = V)
        @test_throws ArgumentError second_differences(ames, :x, :z, V)
    end

    @testset "weights" begin
        w = 1.0 .+ 2.0 .* (df.x .> 0)          # weight 3 for x > 0, else 1
        dup = vcat(df, df[df.x .> 0, :], df[df.x .> 0, :])  # same weighting as 2 extra copies
        un = second_differences(lg, df, :x, :z)
        wt = second_differences(lg, df, :x, :z; weights = w)
        @test !isapprox(wt.second_diff[1], un.second_diff[1]; rtol = 1e-6)
        # equals the unweighted calculation on the expanded data, and a weights column works too
        ex = second_differences(lg, dup, :x, :z)
        @test wt.second_diff ≈ ex.second_diff && wt.se ≈ ex.se
        dw = copy(df); dw.sw = w
        @test second_differences(lg, dw, :x, :z; weights = :sw).second_diff ≈ wt.second_diff
        @test second_differences(lg, df, :x, :z; weights = ones(n)).second_diff ≈ un.second_diff
        # same as the result-level form on a weighted result
        ames = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects, weights = w)
        @test second_differences(ames, :x, :z).second_diff ≈ wt.second_diff
        # local derivative: weighted mean as the point for at = :mean, weighted effects
        lw = second_differences(lg, df, :x, :w; weights = w, delta = 0.01, at = :mean)
        @test lw.eval_point[1] ≈ sum(w .* df.w) / sum(w)
        lex = second_differences(lg, dup, :x, :w; at = lw.eval_point[1], delta = 0.01)
        @test lw.second_diff ≈ lex.second_diff
        lmed = second_differences(lg, df, :x, :w; weights = ones(n), at = :median, delta = 0.01)
        @test lmed.eval_point[1] ≈ sort(df.w)[n ÷ 2] / 2 + sort(df.w)[n ÷ 2 + 1] / 2
        # invalid weights are rejected
        @test_throws ArgumentError second_differences(lg, df, :x, :z; weights = ones(n - 1))
        @test_throws ArgumentError second_differences(lg, df, :x, :z; weights = :nope)
    end

    @testset "contrast= selects one contrast (both forms); all_contrasts is gone" begin
        d = copy(df); d.cg = categorical(d.g)
        m = lm(@formula(y ~ x * z + cg * z + w), d)
        all_rows = second_differences(m, d, :cg, :z; warn = false)
        @test sort(unique(all_rows.contrast)) == ["b - a", "c - a"]
        one = second_differences(m, d, :cg, :z; contrast = "b - a", warn = false)
        @test one.contrast == ["b - a"]
        @test one.second_diff == all_rows[all_rows.contrast .== "b - a", :].second_diff
        # p_adjust family is the selected rows only
        adj = second_differences(m, d, :cg, :z; contrast = "b - a", p_adjust = :bonferroni, warn = false)
        @test adj.p_adjusted == adj.p_value
        # a continuous focal variable: "dy/dx" and its alias
        @test second_differences(m, d, :x, :z; contrast = "dy/dx").contrast == ["dy/dx"]
        @test second_differences(m, d, :x, :z; contrast = "derivative").contrast == ["dy/dx"]
        @test second_differences(m, d, :x, :w; contrast = "dy/dx", at = 0.0).contrast == ["dy/dx"]
        @test_throws ErrorException second_differences(m, d, :cg, :z; contrast = "c - b", warn = false)  # not built under :baseline
        @test_throws ErrorException second_differences(m, d, :x, :w; contrast = "b - a")
        @test second_differences(m, d, :cg, :z; contrast = "c - b", contrasts = :pairwise, warn = false).contrast == ["c - b"]
        # result-level form
        ames = population_margins(m, d; vars = :cg, scenarios = (z = [false, true],), type = :effects)
        r = second_differences(ames, :cg, :z; warn = false)
        @test nrow(r) == 2
        @test second_differences(ames, :cg, :z; contrast = "c - a", warn = false).contrast == ["c - a"]
        @test_throws ErrorException second_differences(ames, :cg, :z; contrast = "dy/dx", warn = false)
        err = try second_differences(ames, :cg, :z; all_contrasts = false) catch e; sprint(showerror, e) end
        @test occursin("all_contrasts", err) && occursin("contrast=", err)
    end

    @testset "alpha is the one name for the significance level" begin
        r = population_margins(lg, df; vars = :x, alpha = 0.10)
        @test r.inference.alpha == 0.10
        @test all(DataFrame(r).ci_upper .- DataFrame(r).estimate .≈ Margins._critical_value(0.10) .* r.standard_errors)
        @test profile_margins(lg, df, DataFrame(z = [0.0]); vars = :x, alpha = 0.2).inference.alpha == 0.2
        # the 2.x keyword names the new one
        err = try population_margins(lg, df; vars = :x, ci_alpha = 0.1) catch e; sprint(showerror, e) end
        @test occursin("renamed `alpha`", err)
        @test_throws ArgumentError profile_margins(lg, df, DataFrame(z = [0.0]); vars = :x, ci_alpha = 0.1)
        # contrast takes `alpha`, not α
        res = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects)
        c90 = Margins.contrast(res, 1, 2; alpha = 0.10)
        c95 = Margins.contrast(res, 1, 2)
        @test c90.metadata[:alpha] == 0.10 && (c90.ci_upper - c90.ci_lower) < (c95.ci_upper - c95.ci_lower)
        # second_differences inherits the result's alpha, and an explicit one wins
        @test second_differences(res, :x, :z).ci_lower[1] ≈ second_differences(res, :x, :z; alpha = 0.05).ci_lower[1]
        r10 = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects, alpha = 0.10)
        @test second_differences(r10, :x, :z).ci_lower[1] ≈ second_differences(r10, :x, :z; alpha = 0.10).ci_lower[1]
        @test_throws MethodError second_differences(lg, df, :x, :z; ci_alpha = 0.1)
    end
end
