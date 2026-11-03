# test_second_differences_structural.jl
# Second differences the model cannot support: structural zeros (NaN inference + one warning per call)
# and extrapolation warnings

using Test
using Margins
using DataFrames
using GLM
using Random

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
        d0 = copy(df); d0.y0 = 0.5 .+ 0.4d0.x .+ 0.3d0.z .+ 0.5d0.t .+ randn(n)    # true x:t effect is 0
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
        t3 = second_differences(lmod, df, [:x, :z], :g; p_adjust=:bonferroni, warn=false)   # 3 tests + 3 structural
        ok = .!isnan.(t3.p_value)
        @test t3.p_adjusted[ok] ≈ min.(1, 3 .* t3.p_value[ok])
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
