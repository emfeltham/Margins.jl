# test_second_differences.jl
# Tests for second differences (interaction effects)

using Test
using Margins
using DataFrames
using GLM
using Random

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
    model = lm(@formula(y ~ x * treated + z), df)
    Σ = vcov(model)

    @testset "Binary Modifier (2 levels)" begin
        # Compute AME at two treatment levels
        ames = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)

        # Test with original function (backward compatibility)
        sd_orig = second_difference(ames, :x, :treated, Σ)
        @test sd_orig.variable == :x
        @test sd_orig.modifier == :treated
        @test sd_orig.modifier_level1 == 0
        @test sd_orig.modifier_level2 == 1
        @test isfinite(sd_orig.second_diff)
        @test isfinite(sd_orig.se)
        @test sd_orig.se > 0
        @test isfinite(sd_orig.z_stat)
        @test 0 <= sd_orig.p_value <= 1

        # Test with new pairwise function
        sd_df = second_differences_pairwise(ames, :x, :treated, Σ)
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
        sd_df = second_differences_pairwise(ames, :x, :education, Σ)

        # Should have 3 pairwise contrasts: (3 choose 2) = 3
        @test nrow(sd_df) == 3
        @test sd_df.modifier_type[1] == :categorical

        # Check that all pairs are present
        pairs = Set([(row.modifier_level1, row.modifier_level2) for row in eachrow(sd_df)])
        @test length(pairs) == 3

        # All should have valid statistics
        @test all(isfinite.(sd_df.second_diff))
        @test all(sd_df.se .>= 0)  # Can be zero if no interaction with modifier
        @test all(isfinite.(sd_df.z_stat))
        @test all(0 .<= sd_df.p_value .<= 1)
        @test all(in.(sd_df.significant, Ref([true, false])))
    end

    @testset "Continuous Modifier (numeric levels)" begin
        # Compute AME at three age values
        ages = [30, 45, 60]
        ames = population_margins(model, df; scenarios=(age=ages,), type=:effects)

        # Use pairwise function with continuous modifier
        sd_df = second_differences_pairwise(ames, :x, :age, Σ; modifier_type=:continuous)

        # Should have 3 pairwise contrasts: (3 choose 2) = 3
        @test nrow(sd_df) == 3
        @test sd_df.modifier_type[1] == :continuous

        # For continuous, second_diff is slope (change per unit)
        # Find the 30-45 comparison
        idx = findfirst(r -> r.modifier_level1 == 30 && r.modifier_level2 == 45, eachrow(sd_df))
        @test !isnothing(idx)

        row_30_45 = sd_df[idx, :]
        # Slope should be: (AME_45 - AME_30) / (45 - 30)
        expected_slope = (row_30_45.ame_at_level2 - row_30_45.ame_at_level1) / 15
        @test row_30_45.second_diff ≈ expected_slope

        # All slopes should be valid
        @test all(isfinite.(sd_df.second_diff))
        @test all(sd_df.se .>= 0)  # Can be zero if no interaction with modifier
    end

    @testset "Auto-detection of modifier type" begin
        # Binary should auto-detect as :binary
        ames_binary = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)
        sd_auto = second_differences_pairwise(ames_binary, :x, :treated, Σ; modifier_type=:auto)
        @test sd_auto.modifier_type[1] == :binary

        # Categorical should auto-detect as :categorical
        ames_cat = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)
        sd_auto = second_differences_pairwise(ames_cat, :x, :education, Σ; modifier_type=:auto)
        @test sd_auto.modifier_type[1] == :categorical

        # Numeric should auto-detect as :continuous
        ames_cont = population_margins(model, df; scenarios=(age=[30, 45, 60],), type=:effects)
        sd_auto = second_differences_pairwise(ames_cont, :x, :age, Σ; modifier_type=:auto)
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

    model = lm(@formula(y ~ x + z), df)
    Σ = vcov(model)

    @testset "Error: insufficient modifier levels" begin
        # Only one level - should error
        ames_one = population_margins(model, df; scenarios=(group=[1],), type=:effects)
        @test_throws ErrorException second_differences_pairwise(ames_one, :x, :group, Σ)
    end

    @testset "Error: missing modifier in profile_values" begin
        ames = population_margins(model, df; type=:effects)
        @test_throws ErrorException second_differences_pairwise(ames, :x, :nonexistent, Σ)
    end

    @testset "Error: dimension mismatch" begin
        ames = population_margins(model, df; scenarios=(group=[1, 2],), type=:effects)
        Σ_wrong = vcov(model)[1:2, 1:2]  # Wrong dimensions
        @test_throws ErrorException second_differences_pairwise(ames, :x, :group, Σ_wrong)
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
        sd = second_differences_pairwise(ames, :x, :m_binary, Σ)

        # The second difference should be non-zero and statistically significant
        @test nrow(sd) == 1
        @test isfinite(sd.second_diff[1])
        @test sd.se[1] > 0

        # z-statistic and p-value should be computed correctly
        @test abs(sd.z_stat[1]) ≈ abs(sd.second_diff[1]) / sd.se[1]

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
        sd = second_differences_pairwise(ames, :x, :age, Σ; modifier_type=:continuous)

        # Check each pairwise slope
        for row in eachrow(sd)
            age_diff = abs(row.modifier_level2 - row.modifier_level1)
            ame_diff = row.ame_at_level2 - row.ame_at_level1
            expected_slope = ame_diff / age_diff

            @test row.second_diff ≈ expected_slope
            @test row.se >= 0  # Can be zero but unlikely with interaction
            @test isfinite(row.z_stat)
        end
    end

    @testset "SE scaling is correct for continuous" begin
        ages = [30.0, 50.0]
        ames = population_margins(model, df; scenarios=(age=ages,), type=:effects)

        # Get slope
        sd_cont = second_differences_pairwise(ames, :x, :age, Σ; modifier_type=:continuous)

        # Also compute as categorical (no scaling)
        sd_cat = second_differences_pairwise(ames, :x, :age, Σ; modifier_type=:categorical)

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
        sd = second_differences_all_contrasts(ames, :x, :region, Σ)

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
        Σ2 = vcov(model2)

        # Compute AME for binary treated variable across regions
        ames2 = population_margins(model2, df2; scenarios=(region=["north", "south", "west"],), type=:effects)

        # Should return 3 pairwise region comparisons for the treated effect
        sd2 = second_differences_all_contrasts(ames2, :treated, :region, Σ2)

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

    model = lm(@formula(y ~ x * treated), df)
    Σ = vcov(model)

    @testset "Wrapper routes to pairwise for continuous focal" begin
        ames = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)

        # Using unified wrapper
        sd_wrapper = second_differences(ames, :x, :treated, Σ)

        # Using direct function
        sd_direct = second_differences_pairwise(ames, :x, :treated, Σ)

        # Should produce same results
        @test sd_wrapper.second_diff == sd_direct.second_diff
        @test sd_wrapper.se == sd_direct.se
        @test sd_wrapper.p_value == sd_direct.p_value
    end

    @testset "Wrapper routes to all_contrasts for categorical modifier >2" begin
        ames = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)

        # Using unified wrapper
        sd_wrapper = second_differences(ames, :x, :education, Σ)

        # Using direct function
        sd_direct = second_differences_pairwise(ames, :x, :education, Σ)

        # Should produce same results (3 pairwise contrasts)
        @test nrow(sd_wrapper) == 3
        @test nrow(sd_direct) == 3
        @test sd_wrapper.second_diff == sd_direct.second_diff
    end

    @testset "Wrapper with all_contrasts=false" begin
        ames = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)

        # Single contrast only
        sd_single = second_differences(ames, :x, :education, Σ; all_contrasts=false)

        # Should return 3 rows (pairwise education levels for single contrast)
        @test nrow(sd_single) == 3
        @test all(sd_single.contrast .== "derivative")
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

        model = lm(@formula(y ~ (x1 + x2 + x3) * treated), df)
        Σ = vcov(model)

        @testset "Multiple Variables - Binary Modifier" begin
            ames = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)

            # Test with vector of variables
            sd = second_differences(ames, [:x1, :x2, :x3], :treated, Σ)
            @test nrow(sd) == 3  # One row per variable
            @test Set(sd.variable) == Set([:x1, :x2, :x3])
            @test all(sd.modifier .== :treated)
            @test all(sd.se .> 0)
            @test all(isfinite.(sd.second_diff))
            @test all(0 .<= sd.p_value .<= 1)

            # Compare to individual calls
            sd1 = second_differences(ames, :x1, :treated, Σ)
            sd2 = second_differences(ames, :x2, :treated, Σ)
            sd3 = second_differences(ames, :x3, :treated, Σ)

            @test sd[sd.variable .== :x1, :second_diff][1] ≈ sd1.second_diff[1]
            @test sd[sd.variable .== :x2, :second_diff][1] ≈ sd2.second_diff[1]
            @test sd[sd.variable .== :x3, :second_diff][1] ≈ sd3.second_diff[1]
        end

        @testset "Multiple Variables - Categorical Modifier" begin
            ames = population_margins(model, df; scenarios=(education=["hs", "college", "grad"],), type=:effects)

            # Test with multiple variables and categorical modifier
            sd = second_differences(ames, [:x1, :x2], :education, Σ)
            @test nrow(sd) == 6  # 2 variables × 3 education pairs
            @test count(sd.variable .== :x1) == 3  # 3 education pairs
            @test count(sd.variable .== :x2) == 3  # 3 education pairs
            @test all(sd.modifier .== :education)
            @test all(sd.se .>= 0)
        end

        @testset "Multiple Variables - Continuous Modifier" begin
            ames = population_margins(model, df; scenarios=(age=[30, 45, 60],), type=:effects)

            # Test with multiple variables and continuous modifier
            sd = second_differences(ames, [:x1, :x2, :x3], :age, Σ; modifier_type=:continuous)
            @test nrow(sd) == 9  # 3 variables × 3 age pairs
            @test count(sd.variable .== :x1) == 3
            @test count(sd.variable .== :x2) == 3
            @test count(sd.variable .== :x3) == 3
            @test all(sd.modifier_type .== :continuous)
        end

        @testset "second_differences_table with Multiple Variables" begin
            ames = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)

            # Test table wrapper
            sd = second_differences_table(ames, [:x1, :x2, :x3], :treated, Σ)
            @test nrow(sd) == 3
            @test hasproperty(sd, :significant)
            @test all(sd.significant .== (sd.p_value .< 0.05))
        end

        @testset "Single Variable Still Works (Backward Compatibility)" begin
            ames = population_margins(model, df; scenarios=(treated=[0, 1],), type=:effects)

            # Single Symbol should still work
            sd = second_differences(ames, :x1, :treated, Σ)
            @test nrow(sd) == 1
            @test sd.variable[1] == :x1
        end
    end
end

println("✓ All second differences tests passed")
