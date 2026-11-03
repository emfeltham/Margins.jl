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
        # a function that merely returns the model covariance is a user-supplied estimator: z
        sd_fun = second_differences(lmod, df, :x, :z; vcov = m -> GLM.vcov(m))
        pm = population_margins(lmod, df; vars = :x, type = :effects, vcov = m -> GLM.vcov(m))
        @test sd_fun.df[1] == Inf == pm.metadata[:df]
        @test second_differences(lmod, df, :x, :w; vcov = m -> GLM.vcov(m)).df[1] == Inf
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

    @testset "result-level form accepts vcov as a keyword" begin
        ames = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects)
        V = 4 .* Matrix(vcov(lg))
        a = second_differences(ames, :x, :z; vcov = V)
        b = second_differences(ames, :x, :z, V)
        @test a.se == b.se
        @test a.se[1] ≈ 2 * second_differences(ames, :x, :z).se[1]
    end
end
