# test_second_differences_regressions.jl
# Regression tests for the second-difference review (row selection, ordering, auto-typing, delta check)

using Test
using Margins
using DataFrames
using GLM
using LinearAlgebra: Symmetric
using Random

@testset "Second Differences - Row selection, ordering, delta" begin
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

    @testset "several scenario variables are rejected, not mixed" begin
        a = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0], z=[0.0, 1.0]), type=:effects)
        @test_throws ErrorException Margins._second_differences_pairwise(a, :x, :m)
        @test_throws ErrorException Margins._second_difference(a, :x, :m)
        @test_throws ErrorException second_differences(a, :x, :m)
    end

    @testset "grouped results are rejected, not mixed" begin
        a = population_margins(model, df; vars=:x, groups=:g, scenarios=(m=[0.0, 1.0],), type=:effects)
        @test_throws ErrorException Margins._second_differences_pairwise(a, :x, :m)
    end

    @testset "numeric levels are ordered ascending in every function" begin
        up = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        down = population_margins(model, df; vars=:x, scenarios=(m=[1.0, 0.0],), type=:effects)
        pw_up = Margins._second_differences_pairwise(up, :x, :m)
        pw_down = Margins._second_differences_pairwise(down, :x, :m)
        sd_down = Margins._second_difference(down, :x, :m)
        @test pw_down.modifier_level1[1] == 0.0 && pw_down.modifier_level2[1] == 1.0
        @test pw_down.second_diff[1] ≈ pw_up.second_diff[1]
        @test pw_down.second_diff[1] ≈ sd_down.second_diff
        @test second_differences(down, :x, :m).second_diff[1] ≈ sd_down.second_diff
    end

    @testset "two non-0/1 numeric levels under :auto warn; explicit type does not" begin
        b = population_margins(model, df; vars=:x, scenarios=(z=[0.0, 10.0],), type=:effects)
        @test_logs (:warn, r"treated as :binary") Margins._second_differences_pairwise(b, :x, :z)
        @test_logs Margins._second_differences_pairwise(b, :x, :z; modifier_type=:continuous)
        c = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        @test_logs Margins._second_differences_pairwise(c, :x, :m)
    end

    @testset "missing contrast / variable give specific errors" begin
        c = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        err = try Margins._second_differences_pairwise(c, :x, :m; contrast="nope"); nothing catch e; e end
        @test err isa ErrorException && occursin("Contrasts available: dy/dx", err.msg)
        err = try Margins._second_differences_pairwise(c, :z, :m); nothing catch e; e end
        @test err isa ErrorException && occursin("not found in the effects result", err.msg)
    end

    @testset "small delta near |at| ≈ 0.5 no longer raises a spurious error" begin
        r = Margins._second_differences_local(model, df, :x, :m, Σ; at=0.5, delta=1e-9)
        @test all(isfinite, r.second_diff)
        r2 = Margins._second_differences_local(model, df, :x, :m, Σ; at=1.0, delta=1e-9)
        @test all(isfinite, r2.second_diff)
    end

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
        r1 = Margins._second_differences_local(model, df, :x, :m, Σ; at=0.5)
        r2 = Margins._second_differences_local(model, df, :x, :m, Symmetric(Σ); at=0.5)
        r3 = Margins._second_differences_local(model, df, :x, :m, m -> Symmetric(vcov(m)); at=0.5)
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

    @testset "EffectIndex: one typed view of the result's labels" begin
        a = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        idx = Margins._effect_index(a)
        @test idx.n == 2 && idx.variables == [:x, :x] && idx.contrasts == ["dy/dx", "dy/dx"]
        @test keys(idx.cells) == (:at_m,) || keys(idx.cells) == (:m,)
        @test Margins._variable_contrasts(idx, :x) == ["dy/dx"]
        @test isempty(Margins._variable_rows(idx, :nope))
        err = try Margins._require_variable(idx, :nope); nothing catch e; e end
        @test err isa ErrorException && occursin("variables: x", err.msg)
        err = try Margins._second_differences_pairwise(a, :x, :zzz); nothing catch e; e end
        @test err isa ErrorException && occursin("does not have modifier :zzz", err.msg)

        # the old "derivative" spelling in a result is matched by the canonical "dy/dx" query
        old = Margins.EffectsResult(a.estimates, a.standard_errors, a.variables,
                                    replace(a.terms, "dy/dx" => "derivative"),
                                    a.context, a.gradients, a.inference, a.metadata)
        @test Margins._effect_index(old).contrasts == ["dy/dx", "dy/dx"]
        @test Margins._second_differences_pairwise(old, :x, :m).second_diff ≈ Margins._second_differences_pairwise(a, :x, :m).second_diff
        @test Margins._second_differences_pairwise(old, :x, :m; contrast="derivative").second_diff ≈
              Margins._second_differences_pairwise(a, :x, :m).second_diff
        @test second_differences(old, :x, :m).second_diff ≈ second_differences(a, :x, :m).second_diff

        # inconsistent result shapes fail loudly instead of indexing out of range
        bad = Margins.EffectsResult(a.estimates, a.standard_errors, a.variables[1:1], a.terms,
                                    a.context, a.gradients, a.inference, a.metadata)
        err = try Margins._effect_index(bad); nothing catch e; e end
        @test err isa ErrorException && occursin("Malformed EffectsResult", err.msg)
        @test_throws ErrorException Margins._second_differences_pairwise(bad, :x, :m)

        # group columns are indexed too, and named as groups in the error
        g = population_margins(model, df; vars=:x, groups=:g, scenarios=(m=[0.0, 1.0],), type=:effects)
        gidx = Margins._effect_index(g)
        @test any(==("group"), values(gidx.sources))
        err = try Margins._second_differences_pairwise(g, :x, :m); nothing catch e; e end
        @test err isa ErrorException && occursin("group variable", err.msg)
    end
end
