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
        @test_throws ErrorException Margins._second_differences_pairwise(a, :x, :m, Σ)
        @test_throws ErrorException Margins._second_difference(a, :x, :m, Σ)
        @test_throws ErrorException second_differences(a, :x, :m, Σ)
    end

    @testset "grouped results are rejected, not mixed" begin
        a = population_margins(model, df; vars=:x, groups=:g, scenarios=(m=[0.0, 1.0],), type=:effects)
        @test_throws ErrorException Margins._second_differences_pairwise(a, :x, :m, Σ)
    end

    @testset "numeric levels are ordered ascending in every function" begin
        up = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        down = population_margins(model, df; vars=:x, scenarios=(m=[1.0, 0.0],), type=:effects)
        pw_up = Margins._second_differences_pairwise(up, :x, :m, Σ)
        pw_down = Margins._second_differences_pairwise(down, :x, :m, Σ)
        sd_down = Margins._second_difference(down, :x, :m, Σ)
        @test pw_down.modifier_level1[1] == 0.0 && pw_down.modifier_level2[1] == 1.0
        @test pw_down.second_diff[1] ≈ pw_up.second_diff[1]
        @test pw_down.second_diff[1] ≈ sd_down.second_diff
        @test second_differences(down, :x, :m, Σ).second_diff[1] ≈ sd_down.second_diff
    end

    @testset "two non-0/1 numeric levels under :auto warn; explicit type does not" begin
        b = population_margins(model, df; vars=:x, scenarios=(w=[0.0, 10.0],), type=:effects)
        @test_logs (:warn, r"treated as :binary") Margins._second_differences_pairwise(b, :x, :w, Σ)
        @test_logs Margins._second_differences_pairwise(b, :x, :w, Σ; modifier_type=:continuous)
        c = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        @test_logs Margins._second_differences_pairwise(c, :x, :m, Σ)
    end

    @testset "missing contrast / variable give specific errors" begin
        c = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        err = try Margins._second_differences_pairwise(c, :x, :m, Σ; contrast="nope"); nothing catch e; e end
        @test err isa ErrorException && occursin("Contrasts available: dy/dx", err.msg)
        err = try Margins._second_differences_pairwise(c, :z, :m, Σ); nothing catch e; e end
        @test err isa ErrorException && occursin("not found in the effects result", err.msg)
    end

    @testset "small delta near |at| ≈ 0.5 no longer raises a spurious error" begin
        r = Margins._second_differences_local(model, df, :x, :m, Σ; at=0.5, delta=1e-9)
        @test all(isfinite, r.second_diff)
        r2 = Margins._second_differences_local(model, df, :x, :m, Σ; at=1.0, delta=1e-9)
        @test all(isfinite, r2.second_diff)
    end

    @testset "inference df follows the covariance that is passed" begin
        Random.seed!(2)
        m = 300
        d2 = DataFrame(x = randn(m), m = rand([0.0, 1.0], m), z = randn(m))
        d2.y = 0.5 .+ 0.3d2.x .+ 0.4d2.m .+ 0.2d2.x .* d2.m .+ randn(m)
        lmod = lm(@formula(y ~ x * m + z), d2)
        V = vcov(lmod)
        ames = population_margins(lmod, d2; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        # model-based covariance (also as Symmetric) -> t with residual df
        @test Margins._second_differences_pairwise(ames, :x, :m, V).df[1] == dof_residual(lmod)
        @test Margins._second_differences_pairwise(ames, :x, :m, Symmetric(V)).df[1] == dof_residual(lmod)
        @test Margins._second_difference(ames, :x, :m, V).df == dof_residual(lmod)
        # any other covariance -> z, even though the result itself used the model-based one
        @test Margins._second_differences_pairwise(ames, :x, :m, 1.2 .* V).df[1] == Inf
        # a robust-covariance result analysed with the model-based Σ -> t
        Vr = 1.2 .* V
        ames_r = population_margins(lmod, d2; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects, vcov=_ -> Vr)
        @test Margins._second_differences_pairwise(ames_r, :x, :m, V).df[1] == dof_residual(lmod)
        @test Margins._second_differences_pairwise(ames_r, :x, :m, Vr).df[1] == Inf
        # non-linear model -> z
        @test Margins._second_differences_pairwise(
            population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects),
            :x, :m, Σ).df[1] == Inf
    end

    @testset "local derivative accepts Symmetric and function vcov" begin
        r1 = Margins._second_differences_local(model, df, :x, :m, Σ; at=0.5)
        r2 = Margins._second_differences_local(model, df, :x, :m, Symmetric(Σ); at=0.5)
        r3 = Margins._second_differences_local(model, df, :x, :m, m -> Symmetric(vcov(m)); at=0.5)
        @test r1.second_diff ≈ r2.second_diff ≈ r3.second_diff
        @test r1.se ≈ r2.se ≈ r3.se
    end

    @testset "vcov is optional and defaults to the result's own covariance" begin
        a = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        @test a.metadata[:vcov] == Σ
        for f in (second_differences, Margins._second_differences_pairwise, Margins._second_differences_all_contrasts)
            @test f(a, :x, :m).second_diff ≈ f(a, :x, :m, Σ).second_diff
            @test f(a, :x, :m).se ≈ f(a, :x, :m, Σ).se
        end
        @test Margins._second_difference(a, :x, :m).se ≈ Margins._second_difference(a, :x, :m, Σ).se
        @test second_differences(a, [:x], :m).se ≈ second_differences(a, [:x], :m, Σ).se
        # the default tracks a robust covariance used for the result
        Vr = 1.5 .* Σ
        ar = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects, vcov=_ -> Vr)
        @test Margins._second_differences_pairwise(ar, :x, :m).se[1] ≈ Margins._second_differences_pairwise(ar, :x, :m, Vr).se[1]
        # a result that records no covariance asks for one
        p = profile_margins(model, df, DataFrame(m=[0.0, 1.0]); type=:effects, vars=[:x])
        @test p.metadata[:vcov] == Σ                      # profile results record it too
        delete!(p.metadata, :vcov)
        @test_throws ErrorException Margins._second_differences_pairwise(p, :x, :m)
        @test Margins._second_differences_pairwise(p, :x, :m, Σ).se[1] > 0
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
        err = try Margins._second_differences_pairwise(a, :x, :zzz, Σ); nothing catch e; e end
        @test err isa ErrorException && occursin("does not have modifier :zzz", err.msg)

        # the old "derivative" spelling in a result is matched by the canonical "dy/dx" query
        old = Margins.EffectsResult(a.estimates, a.standard_errors, a.variables,
                                    replace(a.terms, "dy/dx" => "derivative"),
                                    a.profile_values, a.group_values, a.gradients, a.metadata)
        @test Margins._effect_index(old).contrasts == ["dy/dx", "dy/dx"]
        @test Margins._second_differences_pairwise(old, :x, :m, Σ).second_diff ≈ Margins._second_differences_pairwise(a, :x, :m, Σ).second_diff
        @test Margins._second_differences_pairwise(old, :x, :m, Σ; contrast="derivative").second_diff ≈
              Margins._second_differences_pairwise(a, :x, :m, Σ).second_diff
        @test second_differences(old, :x, :m, Σ).second_diff ≈ second_differences(a, :x, :m, Σ).second_diff

        # inconsistent result shapes fail loudly instead of indexing out of range
        bad = Margins.EffectsResult(a.estimates, a.standard_errors, a.variables[1:1], a.terms,
                                    a.profile_values, a.group_values, a.gradients, a.metadata)
        err = try Margins._effect_index(bad); nothing catch e; e end
        @test err isa ErrorException && occursin("Malformed EffectsResult", err.msg)
        @test_throws ErrorException Margins._second_differences_pairwise(bad, :x, :m, Σ)

        # group columns are indexed too, and named as groups in the error
        g = population_margins(model, df; vars=:x, groups=:g, scenarios=(m=[0.0, 1.0],), type=:effects)
        gidx = Margins._effect_index(g)
        @test any(==("group"), values(gidx.sources))
        err = try Margins._second_differences_pairwise(g, :x, :m, Σ); nothing catch e; e end
        @test err isa ErrorException && occursin("group variable", err.msg)
    end
end
