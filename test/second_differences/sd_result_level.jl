# sd_result_level.jl
# second_differences(result, variables, modifier) on an EffectsResult (population_margins or
# profile_margins) and its engines (_second_difference, _second_differences_pairwise,
# _second_differences_all_contrasts): modifier types, pairwise rows, continuous slopes, several
# focal variables, row selection and ordering, the EffectIndex, and the public API surface

using Test
using Margins
using DataFrames
using GLM
using CategoricalArrays
using CovarianceMatrices
using Random

@testset "second differences: public API surface" begin
    exported = Set(names(Margins))
    @test :second_differences in exported
    removed = (:second_differences_at, :second_differences_pairwise, :second_differences_all_contrasts,
               :second_differences_table, :second_difference)
    for f in removed
        @test f ∉ exported                       # no longer exported ...
        @test !isdefined(Margins, f)             # ... and no longer defined
    end
    @test Set(keys(Margins.REMOVED_SECOND_DIFFERENCE_FUNCTIONS)) == Set(removed)

    # the two methods of the one public function
    ms = methods(second_differences)
    @test any(m -> Base.unwrap_unionall(m.sig).parameters[2] <: Margins.EffectsResult, ms)
    @test any(m -> Base.unwrap_unionall(m.sig).parameters[2] <: Margins.RegressionModel, ms)
end

@testset "second differences: removed functions explain themselves" begin
    for (f, mention) in ((:second_differences_at, "at="), (:second_differences_pairwise, "contrast="),
                         (:second_differences_all_contrasts, "second_differences(result"),
                         (:second_differences_table, "second_differences(result"),
                         (:second_difference, "include_gradients=true"))
        err = try
            getfield(Margins, f); nothing
        catch e
            e
        end
        @test err isa UndefVarError
        msg = sprint(showerror, err)
        @test occursin("removed in Margins 3.0", msg)
        @test occursin(mention, msg)
    end
    # unrelated undefined names get no hint
    err = try getfield(Margins, :definitely_not_a_function); nothing catch e; e end
    @test !occursin("removed in Margins 3.0", sprint(showerror, err))
end

@testset "Second Differences - Basic Functionality" begin
    # Generate test data
    Random.seed!(123)
    n = 200
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        treated = rand([0, 1], n),
        education = rand(["hs", "college", "grad"], n),
        age = rand(25:65, n)
    )

    # Add interaction term implicitly through data structure
    df.y = 0.5 .+ 0.3 .* df.x .+ 0.2 .* df.z .+
           0.4 .* df.treated .+ 0.15 .* df.x .* df.treated .+ randn(n) .* 0.5

    # Model WITH interaction term to get non-zero second differences
    model = lm(@formula(y ~ x * treated + z + education + age), df)
    Σ = vcov(model)

    @testset "Binary Modifier (2 levels)" begin
        # Compute AME at two treatment levels
        ames = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)

        # Test with original function (backward compatibility)
        sd_orig = Margins._second_difference(ames, :x, :treated)
        @test sd_orig.variable == :x
        @test sd_orig.modifier == :treated
        @test sd_orig.modifier_level1 == 0
        @test sd_orig.modifier_level2 == 1
        @test isfinite(sd_orig.second_diff)
        @test isfinite(sd_orig.se)
        @test sd_orig.se > 0
        @test isfinite(sd_orig.statistic)
        @test 0 <= sd_orig.p_value <= 1

        # Test with new pairwise function
        sd_df = Margins._second_differences_pairwise(ames, :x, :treated)
        @test nrow(sd_df) == 1  # Binary has only one pairwise contrast
        @test sd_df.variable[1] == :x
        @test sd_df.modifier[1] == :treated
        @test sd_df.modifier_type[1] == :binary
        @test isfinite(sd_df.second_diff[1])
        @test sd_df.se[1] > 0

        # Results should match between old and new functions
        @test sd_df.second_diff[1] ≈ sd_orig.second_diff
        @test sd_df.se[1] ≈ sd_orig.se
        @test sd_df.p_value[1] ≈ sd_orig.p_value
    end

    @testset "Categorical Modifier (3 levels)" begin
        # Compute AME at three education levels
        ames = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)

        # Use pairwise function
        sd_df = Margins._second_differences_pairwise(ames, :x, :education)

        # Should have 3 pairwise contrasts: (3 choose 2) = 3
        @test nrow(sd_df) == 3
        @test sd_df.modifier_type[1] == :categorical

        # Check that all pairs are present
        pairs = Set([(row.modifier_level1, row.modifier_level2) for row in eachrow(sd_df)])
        @test length(pairs) == 3

        # `education` enters the model but has no interaction with `x`, so each difference is zero by
        # construction: zero estimate and SE, and NaN inference (there is nothing to test)
        @test all(abs.(sd_df.second_diff) .< 1e-12) && all(sd_df.se .< 1e-12)
        @test all(isnan, sd_df.statistic) && all(isnan, sd_df.p_value)
        @test all(isnan, sd_df.ci_lower) && all(isnan, sd_df.ci_upper)
        @test all(sd_df.significant .== false)
    end

    @testset "Auto-detection of modifier type" begin
        # Binary should auto-detect as :binary
        ames_binary = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)
        sd_auto = Margins._second_differences_pairwise(ames_binary, :x, :treated; modifier_type=:auto)
        @test sd_auto.modifier_type[1] == :binary

        # Categorical should auto-detect as :categorical
        ames_cat = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)
        sd_auto = Margins._second_differences_pairwise(ames_cat, :x, :education; modifier_type=:auto)
        @test sd_auto.modifier_type[1] == :categorical

        # Numeric should auto-detect as :continuous
        ames_cont = population_margins(model, df; scenarios=(age=[30, 45, 60],), type=:effects)
        sd_auto = Margins._second_differences_pairwise(ames_cont, :x, :age; modifier_type=:auto)
        @test sd_auto.modifier_type[1] == :continuous
    end
end

@testset "Second Differences - Edge Cases" begin
    Random.seed!(456)
    n = 100
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        group = rand([1, 2], n)
    )
    df.y = 0.5 .+ 0.3 .* df.x .+ 0.2 .* df.z .+ randn(n) .* 0.3

    model = lm(@formula(y ~ x + z + group), df)
    Σ = vcov(model)

    @testset "Error: insufficient modifier levels" begin
        # Only one level - should error
        ames_one = population_margins(model, df; scenarios=(group=[1],), type=:effects)
        @test_throws ArgumentError Margins._second_differences_pairwise(ames_one, :x, :group)
    end

    @testset "Error: missing modifier in the result's context" begin
        ames = population_margins(model, df; type=:effects)
        @test_throws ArgumentError Margins._second_differences_pairwise(ames, :x, :nonexistent)
    end

    @testset "Error: dimension mismatch" begin
        ames = population_margins(model, df; scenarios=(group=[1, 2],), type=:effects)
        Σ_wrong = vcov(model)[1:2, 1:2]  # Wrong dimensions (a hand-built, malformed result)
        bad = EffectsResult(ames.estimates, ames.standard_errors, ames.variables, ames.terms, ames.context,
                            ames.gradients, Inference(Σ_wrong, ames.inference.df, ames.inference.alpha), ames.metadata)
        @test_throws ErrorException Margins._second_differences_pairwise(bad, :x, :group)
    end
end

@testset "Second Differences - Statistical Properties" begin
    Random.seed!(789)
    n = 500

    # Model WITH interaction: y = β₀ + β₁x + β₂m + β₃(x*m) + ε
    df = DataFrame(
        x = randn(n),
        m_binary = rand([0, 1], n),
        m_cat = rand(["A", "B", "C"], n)
    )

    # True interaction coefficient
    β_interaction = 0.5
    df.y = 1.0 .+ 0.8 .* df.x .+ 0.3 .* df.m_binary .+
           β_interaction .* df.x .* df.m_binary .+ randn(n) .* 0.5

    # Model WITH interaction term to detect the effect
    model = lm(@formula(y ~ x * m_binary), df)
    Σ = vcov(model)

    @testset "Second difference should detect interaction" begin
        ames = population_margins(model, df; scenarios=(m_binary=[0, 1],), type=:effects)
        sd = Margins._second_differences_pairwise(ames, :x, :m_binary)

        # The second difference should be non-zero and statistically significant
        @test nrow(sd) == 1
        @test isfinite(sd.second_diff[1])
        @test sd.se[1] > 0

        # z-statistic and p-value should be computed correctly
        @test abs(sd.statistic[1]) ≈ abs(sd.second_diff[1]) / sd.se[1]

        # Should be statistically significant (large sample with true interaction)
        @test sd.p_value[1] < 0.05
    end
end

@testset "Second Differences - Continuous Scaling" begin
    Random.seed!(321)
    n = 200
    df = DataFrame(
        x = randn(n),
        age = rand(20:70, n)
    )
    # Add interaction for non-zero second differences
    df.y = 1.0 .+ 0.5 .* df.x .+ 0.01 .* df.age .+ 0.01 .* df.x .* df.age .+ randn(n) .* 0.3

    # Model WITH interaction
    model = lm(@formula(y ~ x * age), df)
    Σ = vcov(model)

    @testset "Slope scaling is correct" begin
        ages = [25.0, 35.0, 55.0]
        ames = population_margins(model, df; scenarios=(age=ages,), type=:effects)
        sd = Margins._second_differences_pairwise(ames, :x, :age; modifier_type=:continuous)

        # Check each pairwise slope
        for row in eachrow(sd)
            age_diff = abs(row.modifier_level2 - row.modifier_level1)
            ame_diff = row.ame_at_level2 - row.ame_at_level1
            expected_slope = ame_diff / age_diff

            @test row.second_diff ≈ expected_slope
            @test row.se >= 0  # Can be zero but unlikely with interaction
            @test isfinite(row.statistic)
        end
    end

    @testset "SE scaling is correct for continuous" begin
        ages = [30.0, 50.0]
        ames = population_margins(model, df; scenarios=(age=ages,), type=:effects)

        # Get slope
        sd_cont = Margins._second_differences_pairwise(ames, :x, :age; modifier_type=:continuous)

        # Also compute as categorical (no scaling)
        sd_cat = Margins._second_differences_pairwise(ames, :x, :age; modifier_type=:categorical)

        # SE relationship: SE_slope = SE_diff / |age_diff|
        age_diff = 20.0
        @test sd_cont.se[1] ≈ sd_cat.se[1] / age_diff rtol=1e-10
        @test sd_cont.second_diff[1] ≈ sd_cat.second_diff[1] / age_diff rtol=1e-10
    end
end

@testset "Second Differences - Categorical × Categorical" begin
    Random.seed!(555)
    n = 300
    df = DataFrame(
        x = randn(n),
        education = rand(["hs", "college", "grad"], n),
        region = rand(["north", "south", "west"], n)
    )

    # Interaction between education and region
    edu_codes = Dict("hs" => 0, "college" => 1, "grad" => 2)
    region_codes = Dict("north" => 0, "south" => 1, "west" => 2)
    df.y = 1.0 .+ 0.5 .* df.x .+
           0.3 .* [edu_codes[e] for e in df.education] .+
           0.2 .* [region_codes[r] for r in df.region] .+
           0.1 .* df.x .* [region_codes[r] for r in df.region] .+
           randn(n) .* 0.4

    # Model with interaction
    model = lm(@formula(y ~ x * region), df)
    Σ = vcov(model)

    @testset "All contrasts across categorical modifier" begin
        # Compute AME for continuous variable across regions
        ames = population_margins(model, df; scenarios=(region=["north", "south", "west"],), type=:effects)

        # Should return 3 pairwise region comparisons (south-north, west-north, west-south)
        sd = Margins._second_differences_all_contrasts(ames, :x, :region)

        @test nrow(sd) == 3  # 3 pairwise region comparisons
        @test sd.modifier_type[1] == :categorical
        @test all(isfinite.(sd.second_diff))
        @test all(sd.se .>= 0)
    end

    @testset "Categorical focal × categorical modifier (full matrix)" begin
        # This is the complex case: categorical variable with pairwise contrasts × categorical modifier
        # Need to compute AME for categorical variable (with contrasts) across regions

        # For now, test with a simpler categorical variable
        df2 = copy(df)
        df2.treated = rand([0, 1], n)
        df2.y2 = 1.0 .+ 0.5 .* df2.x .+ 0.4 .* df2.treated .+
                 0.2 .* [region_codes[r] for r in df2.region] .+
                 0.15 .* df2.treated .* [region_codes[r] for r in df2.region] .+
                 randn(n) .* 0.4

        model2 = lm(@formula(y2 ~ x + treated * region), df2)

        # Compute AME for binary treated variable across regions
        ames2 = population_margins(model2, df2; scenarios=(region=["north", "south", "west"],), type=:effects)

        # Should return 3 pairwise region comparisons for the treated effect
        sd2 = Margins._second_differences_all_contrasts(ames2, :treated, :region)

        @test nrow(sd2) == 3  # 3 pairwise region comparisons
        @test all(isfinite.(sd2.second_diff))
    end
end

@testset "Second Differences - Unified Wrapper" begin
    Random.seed!(999)
    n = 200
    df = DataFrame(
        x = randn(n),
        treated = rand([0, 1], n),
        education = rand(["hs", "college", "grad"], n),
        age = rand(25:65, n)
    )

    df.y = 1.0 .+ 0.5 .* df.x .+ 0.3 .* df.treated .+
           0.2 .* df.x .* df.treated .+ randn(n) .* 0.4

    model = lm(@formula(y ~ x * treated + education + age), df)
    Σ = vcov(model)

    @testset "Wrapper with a single contrast" begin
        ames = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)

        # Single contrast only
        sd_single = second_differences(ames, :x, :education; contrast="dy/dx")

        # Should return 3 rows (pairwise education levels for single contrast)
        @test nrow(sd_single) == 3
        @test all(sd_single.contrast .== "dy/dx")
    end

    @testset "Multiple Variables Support" begin
        Random.seed!(456)
        n = 200
        df = DataFrame(
            x1 = randn(n),
            x2 = randn(n),
            x3 = randn(n),
            treated = rand([0, 1], n),
            education = rand(["hs", "college", "grad"], n),
            age = rand(25:65, n)
        )

        # Model with interactions for all variables
        df.y = 0.5 .+ 0.3 .* df.x1 .+ 0.2 .* df.x2 .+ 0.1 .* df.x3 .+
               0.4 .* df.treated .+
               0.15 .* df.x1 .* df.treated .+
               0.10 .* df.x2 .* df.treated .+
               0.05 .* df.x3 .* df.treated .+
               randn(n) .* 0.5

        model = lm(@formula(y ~ (x1 + x2 + x3) * treated + education + age), df)
        Σ = vcov(model)

        @testset "Multiple Variables - Binary Modifier" begin
            ames = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)

            # Test with vector of variables
            sd = second_differences(ames, [:x1, :x2, :x3], :treated)
            @test nrow(sd) == 3  # One row per variable
            @test Set(sd.variable) == Set([:x1, :x2, :x3])
            @test all(sd.modifier .== :treated)
            @test all(sd.se .> 0)
            @test all(isfinite.(sd.second_diff))
            @test all(0 .<= sd.p_value .<= 1)

            # Compare to individual calls
            sd1 = second_differences(ames, :x1, :treated)
            sd2 = second_differences(ames, :x2, :treated)
            sd3 = second_differences(ames, :x3, :treated)

            @test sd[sd.variable .== :x1, :second_diff][1] ≈ sd1.second_diff[1]
            @test sd[sd.variable .== :x2, :second_diff][1] ≈ sd2.second_diff[1]
            @test sd[sd.variable .== :x3, :second_diff][1] ≈ sd3.second_diff[1]
            # one focal variable gives one row
            @test nrow(sd1) == 1 && sd1.variable[1] == :x1
            # the significance column follows the p-values
            @test all(sd.significant .== (sd.p_value .< 0.05))
        end

        @testset "Multiple Variables - Categorical Modifier" begin
            ames = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)

            # Test with multiple variables and categorical modifier
            sd = second_differences(ames, [:x1, :x2], :education)
            @test nrow(sd) == 6  # 2 variables × 3 education pairs
            @test count(sd.variable .== :x1) == 3  # 3 education pairs
            @test count(sd.variable .== :x2) == 3  # 3 education pairs
            @test all(sd.modifier .== :education)
        end

        @testset "Multiple Variables - Continuous Modifier" begin
            ames = population_margins(model, df; scenarios=(age=[30, 45, 60],), type=:effects)

            # Test with multiple variables and continuous modifier
            sd = second_differences(ames, [:x1, :x2, :x3], :age; modifier_type=:continuous)
            @test nrow(sd) == 9  # 3 variables × 3 age pairs
            @test count(sd.variable .== :x1) == 3
            @test count(sd.variable .== :x2) == 3
            @test count(sd.variable .== :x3) == 3
            @test all(sd.modifier_type .== :continuous)
        end
    end
end

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
        @test_throws ArgumentError Margins._second_differences_pairwise(a, :x, :m)
        @test_throws ArgumentError Margins._second_difference(a, :x, :m)
        @test_throws ArgumentError second_differences(a, :x, :m)
    end

    @testset "grouped results are rejected, not mixed" begin
        a = population_margins(model, df; vars=:x, groups=:g, scenarios=(m=[0.0, 1.0],), type=:effects)
        @test_throws ArgumentError Margins._second_differences_pairwise(a, :x, :m)
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
        @test err isa ArgumentError && occursin("Contrasts available: dy/dx", err.msg)
        err = try Margins._second_differences_pairwise(c, :z, :m); nothing catch e; e end
        @test err isa ArgumentError && occursin("not found in the effects result", err.msg)
    end

    @testset "small delta near |at| ≈ 0.5 no longer raises a spurious error" begin
        r = second_differences(model, df, :x, :m; vcov=Σ, modifier_type=:continuous, at=0.5, delta=1e-9)
        @test all(isfinite, r.second_diff)
        r2 = second_differences(model, df, :x, :m; vcov=Σ, modifier_type=:continuous, at=1.0, delta=1e-9)
        @test all(isfinite, r2.second_diff)
    end

    @testset "EffectIndex: one typed view of the result's labels" begin
        a = population_margins(model, df; vars=:x, scenarios=(m=[0.0, 1.0],), type=:effects)
        idx = Margins._effect_index(a)
        @test idx.n == 2 && idx.variables == [:x, :x] && idx.contrasts == ["dy/dx", "dy/dx"]
        @test keys(idx.cells) == (:at_m,) || keys(idx.cells) == (:m,)
        @test Margins._variable_contrasts(idx, :x) == ["dy/dx"]
        @test isempty(Margins._variable_rows(idx, :nope))
        err = try Margins._require_variable(idx, :nope); nothing catch e; e end
        @test err isa ArgumentError && occursin("variables: x", err.msg)
        err = try Margins._second_differences_pairwise(a, :x, :zzz); nothing catch e; e end
        @test err isa ArgumentError && occursin("does not have modifier :zzz", err.msg)

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
        @test err isa ArgumentError && occursin("group variable", err.msg)
    end
end

@testset "second differences of profile_margins and predictions results" begin
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
        @test_throws ArgumentError second_differences(p2, :x, :t)
        # predictions results get a pointer, not a MethodError
        pp = profile_margins(lmod, df, grid; type=:predictions)
        ap = population_margins(lmod, df; scenarios=(t=[0, 1],), type=:predictions)
        for r in (pp, ap)
            err = try second_differences(r, :x, :t); nothing catch e; e end
            @test err isa ArgumentError && occursin("type=:effects", err.msg)
        end
        err = try second_differences(ap, [:x], :t); nothing catch e; e end
        @test err isa ArgumentError && occursin("PredictionsResult", err.msg)
    end
end
