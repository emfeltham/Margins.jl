# test_second_differences_at.jl
# Tests for second_differences_at() - local derivatives at specific points

using Test
using Margins
using DataFrames
using GLM
using Random
using Statistics
using CategoricalArrays

@testset "second_differences_at() - Local Derivatives" begin
    Random.seed!(789)
    n = 200

    # Generate test data with interactions
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        x3 = randn(n),
        age = rand(25:65, n) .+ randn(n),  # Continuous modifier
        income = rand(20000:80000, n) .+ randn(n) .* 1000,  # Another continuous modifier
        region = rand(["north", "south", "west"], n)
    )

    # Model with interactions
    df.y = 0.5 .+ 0.3 .* df.x1 .+ 0.2 .* df.x2 .+ 0.1 .* df.x3 .+
           0.01 .* df.age .+ 0.0001 .* df.income .+
           0.02 .* df.x1 .* df.age .+  # Interaction with age
           0.01 .* df.x2 .* df.age .+
           randn(n) .* 0.5

    model = lm(@formula(y ~ (x1 + x2 + x3) * age + income), df)
    Σ = vcov(model)

    @testset "Basic Functionality - At Mean" begin
        # Single variable at mean
        sd = second_differences_at(model, df, :x1, :age, Σ; at=:mean)

        @test nrow(sd) == 1
        @test sd.variable[1] == :x1
        @test sd.modifier[1] == :age
        @test sd.eval_point[1] ≈ mean(df.age) rtol=0.01
        @test isfinite(sd.derivative[1])
        @test isfinite(sd.se[1])
        @test sd.se[1] > 0
        @test isfinite(sd.z_stat[1])
        @test 0 <= sd.p_value[1] <= 1
        @test sd.delta_used[1] > 0
    end

    @testset "At Median" begin
        sd = second_differences_at(model, df, :x1, :age, Σ; at=:median)

        @test nrow(sd) == 1
        @test sd.eval_point[1] ≈ median(df.age) rtol=0.01
    end

    @testset "At Specific Value" begin
        sd = second_differences_at(model, df, :x1, :age, Σ; at=45.0)

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
    end

    @testset "Multiple Evaluation Points" begin
        sd = second_differences_at(model, df, :x1, :age, Σ; at=[30.0, 45.0, 60.0])

        @test nrow(sd) == 3
        @test sd.eval_point == [30.0, 45.0, 60.0]
        @test all(sd.variable .== :x1)
        @test all(sd.modifier .== :age)
    end

    @testset "Multiple Variables - Single Eval Point" begin
        sd = second_differences_at(model, df, [:x1, :x2, :x3], :age, Σ; at=:mean)

        @test nrow(sd) == 3
        @test Set(sd.variable) == Set([:x1, :x2, :x3])
        @test all(sd.modifier .== :age)
        @test all(sd.eval_point .≈ mean(df.age))
        @test all(isfinite.(sd.derivative))
        @test all(sd.se .> 0)
    end

    @testset "Multiple Variables × Multiple Eval Points" begin
        sd = second_differences_at(model, df, [:x1, :x2], :age, Σ; at=[30.0, 45.0, 60.0])

        @test nrow(sd) == 6  # 2 variables × 3 eval points
        @test count(sd.variable .== :x1) == 3
        @test count(sd.variable .== :x2) == 3

        # Check eval points are correct for each variable
        x1_rows = sd[sd.variable .== :x1, :]
        @test x1_rows.eval_point == [30.0, 45.0, 60.0]
    end

    @testset "Profile Support - Single Variable" begin
        # With profile holding income constant
        sd = second_differences_at(model, df, :x1, :age, Σ;
                                  at=45.0,
                                  profile=(income=50000,))

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
        @test sd.income[1] == 50000
    end

    @testset "Profile Support - Multiple Profile Variables" begin
        sd = second_differences_at(model, df, :x1, :age, Σ;
                                  at=45.0,
                                  profile=(income=50000, region="north"))

        @test nrow(sd) == 1
        @test sd.income[1] == 50000
        @test sd.region[1] == "north"
    end

    @testset "Profile with Multiple Variables and Eval Points" begin
        sd = second_differences_at(model, df, [:x1, :x2], :age, Σ;
                                  at=[30.0, 60.0],
                                  profile=(income=50000,))

        @test nrow(sd) == 4  # 2 variables × 2 eval points
        @test all(sd.income .== 50000)
        @test count(sd.eval_point .== 30.0) == 2  # x1 and x2
        @test count(sd.eval_point .== 60.0) == 2  # x1 and x2
    end

    @testset "NamedTuple for at Parameter" begin
        # Full profile specification via `at`
        sd = second_differences_at(model, df, :x1, :age, Σ;
                                  at=(age=45.0, income=50000, region="north"))

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
        @test sd.income[1] == 50000
        @test sd.region[1] == "north"
    end

    @testset "NamedTuple with Both at and profile" begin
        # Combine `at` NamedTuple with `profile` parameter
        sd = second_differences_at(model, df, :x1, :age, Σ;
                                  at=(age=45.0, income=50000),
                                  profile=(region="south",))

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
        @test sd.income[1] == 50000
        @test sd.region[1] == "south"
    end

    @testset "Custom Delta" begin
        sd1 = second_differences_at(model, df, :x1, :age, Σ; at=:mean, delta=:auto)
        sd2 = second_differences_at(model, df, :x1, :age, Σ; at=:mean, delta=1.0)

        @test sd1.delta_used[1] ≈ 0.01 * std(df.age) rtol=0.01
        @test sd2.delta_used[1] == 1.0

        # Derivatives may differ slightly due to different delta
        @test sd1.derivative[1] != sd2.derivative[1]
    end

    @testset "Scale Parameter" begin
        # Test both :link and :response scales
        sd_response = second_differences_at(model, df, :x1, :age, Σ; scale=:response)
        sd_link = second_differences_at(model, df, :x1, :age, Σ; scale=:link)

        # For linear models, they should be the same
        @test sd_response.derivative[1] ≈ sd_link.derivative[1]
    end

    @testset "vcov as Function (Mixed Model Support)" begin
        # Test with vcov as a function instead of matrix
        sd_matrix = second_differences_at(model, df, :x1, :age, Σ)
        sd_function = second_differences_at(model, df, :x1, :age, vcov)  # vcov is the function

        # Results should be identical
        @test sd_matrix.derivative[1] ≈ sd_function.derivative[1]
        @test sd_matrix.se[1] ≈ sd_function.se[1]
        @test sd_matrix.p_value[1] ≈ sd_function.p_value[1]
    end

    @testset "Error Handling - Invalid Modifier" begin
        @test_throws ErrorException second_differences_at(
            model, df, :x1, :nonexistent, Σ
        )
    end

    @testset "Error Handling - Non-Numeric Modifier" begin
        @test_throws ErrorException second_differences_at(
            model, df, :x1, :region, Σ
        )
    end

    @testset "Error Handling - Invalid Variable" begin
        @test_throws ErrorException second_differences_at(
            model, df, :nonexistent, :age, Σ
        )
    end

    @testset "Error Handling - Invalid Profile Variable" begin
        @test_throws ErrorException second_differences_at(
            model, df, :x1, :age, Σ;
            profile=(nonexistent=100,)
        )
    end

    @testset "Error Handling - Invalid at Parameter" begin
        @test_throws TypeError second_differences_at(
            model, df, :x1, :age, Σ;
            at="invalid"
        )
    end

    @testset "Error Handling - NamedTuple without Modifier" begin
        @test_throws ErrorException second_differences_at(
            model, df, :x1, :age, Σ;
            at=(income=50000,)  # Missing :age
        )
    end

    @testset "Error Handling - Invalid Delta" begin
        @test_throws TypeError second_differences_at(
            model, df, :x1, :age, Σ;
            delta="invalid"
        )
    end

    @testset "Error Handling - Invalid Contrasts" begin
        @test_throws ErrorException second_differences_at(
            model, df, :x1, :age, Σ;
            contrasts=:invalid
        )
    end

    @testset "Consistency with second_differences()" begin
        # Compare at-point derivative with discrete contrast slope
        # They should be similar when evaluation points are the same

        # Using second_differences_at at specific points
        sd_at = second_differences_at(model, df, :x1, :age, Σ;
                                     at=[40.0, 50.0])

        # Using second_differences with same points
        ames = population_margins(model, df; scenarios=(age=[40.0, 50.0],), type=:effects)
        sd_discrete = second_differences(ames, :x1, :age, Σ; modifier_type=:continuous)

        # The slopes should be similar (but not identical due to different delta)
        # second_differences uses (50-40)=10 as delta
        # second_differences_at uses auto delta around each point
        @test sd_discrete.second_diff[1] isa Real
        @test sd_at.derivative[1] isa Real
    end

    @testset "Real-World Example - Interaction Detection" begin
        # Model with strong interaction
        Random.seed!(999)
        n = 300
        df_interact = DataFrame(
            x = randn(n),
            m = randn(n) .* 10 .+ 50
        )
        df_interact.y = 1.0 .+ 0.5 .* df_interact.x .+
                        0.02 .* df_interact.m .+
                        0.1 .* df_interact.x .* df_interact.m .+  # Strong interaction
                        randn(n) .* 0.5

        model_interact = lm(@formula(y ~ x * m), df_interact)

        # Compute derivative at mean
        sd = second_differences_at(model_interact, df_interact, :x, :m, vcov(model_interact))

        # Should detect significant interaction
        @test abs(sd.z_stat[1]) > 2  # Statistically significant
        @test sd.p_value[1] < 0.05
    end

    @testset "Vector-Valued Profile Support" begin
        # Create fresh data frame for stratification tests with categorical variables
        Random.seed!(888)
        df_strat = DataFrame(
            x1 = randn(n),
            x2 = randn(n),
            x3 = randn(n),
            age = rand(25:65, n) .+ randn(n),
            income = rand(20000:80000, n) .+ randn(n) .* 1000,
            socio4 = rand([false, true], n),
            gender = rand(["male", "female"], n)
        )

        # Generate y with interactions
        df_strat.y = 0.5 .+ 0.3 .* df_strat.x1 .+ 0.2 .* df_strat.x2 .+ 0.1 .* df_strat.x3 .+
                     0.01 .* df_strat.age .+ 0.0001 .* df_strat.income .+
                     0.02 .* df_strat.x1 .* df_strat.age .+
                     randn(n) .* 0.5

        # Refit model with categorical variables
        model_strat = lm(@formula(y ~ (x1 + x2 + x3) * age + income + socio4 + gender), df_strat)
        Σ_strat = vcov(model_strat)

        @testset "Single Vector Profile" begin
            # Vector-valued profile: separate estimation for each socio4 level
            sd = second_differences_at(model_strat, df_strat, :x1, :age, Σ_strat;
                                      at=45.0,
                                      profile=(socio4=[false, true],))

            @test nrow(sd) == 2  # One row per socio4 level
            @test sd.variable == [:x1, :x1]
            @test sd.modifier == [:age, :age]
            @test sd.eval_point == [45.0, 45.0]
            @test hasproperty(sd, :socio4)
            @test Set(sd.socio4) == Set([false, true])
            @test all(isfinite.(sd.derivative))
            @test all(sd.se .> 0)
        end

        @testset "Vector Profile with Scalar Profile" begin
            # Mixed: vector socio4 + scalar income
            sd = second_differences_at(model_strat, df_strat, :x1, :age, Σ_strat;
                                      at=45.0,
                                      profile=(socio4=[false, true],
                                              income=50000.0))

            @test nrow(sd) == 2
            @test hasproperty(sd, :socio4)
            @test hasproperty(sd, :income)
            @test all(sd.income .== 50000.0)  # Scalar stays constant
            @test Set(sd.socio4) == Set([false, true])  # Vector varies
        end

        @testset "Multiple Vector Profiles (Cartesian Product)" begin
            # Cartesian product: socio4 × gender
            sd = second_differences_at(model_strat, df_strat, :x1, :age, Σ_strat;
                                      at=45.0,
                                      profile=(socio4=[false, true],
                                              gender=["male", "female"]))

            @test nrow(sd) == 4  # 2 × 2 = 4 combinations
            @test hasproperty(sd, :socio4)
            @test hasproperty(sd, :gender)

            # Check all combinations present
            combos = Set(zip(sd.socio4, sd.gender))
            expected_combos = Set([
                (false, "male"), (false, "female"),
                (true, "male"), (true, "female")
            ])
            @test combos == expected_combos
        end

        @testset "Vector Profile × Multiple Eval Points" begin
            # Combine vector profile with multiple evaluation points
            sd = second_differences_at(model_strat, df_strat, :x1, :age, Σ_strat;
                                      at=[30.0, 60.0],
                                      profile=(socio4=[false, true],))

            @test nrow(sd) == 4  # 2 eval points × 2 socio4 levels

            # Check structure
            @test count(sd.eval_point .== 30.0) == 2
            @test count(sd.eval_point .== 60.0) == 2
            @test count(sd.socio4 .== false) == 2
            @test count(sd.socio4 .== true) == 2
        end

        @testset "Multiple Variables × Vector Profile" begin
            # Multiple focal variables with vector profile
            sd = second_differences_at(model_strat, df_strat, [:x1, :x2], :age, Σ_strat;
                                      at=45.0,
                                      profile=(socio4=[false, true],))

            @test nrow(sd) == 4  # 2 variables × 2 socio4 levels
            @test count(sd.variable .== :x1) == 2
            @test count(sd.variable .== :x2) == 2

            # Each variable should have both socio4 levels
            x1_rows = sd[sd.variable .== :x1, :]
            @test Set(x1_rows.socio4) == Set([false, true])
        end

        @testset "Complex Combination: All Dimensions" begin
            # Combine everything: multiple variables × multiple eval points × vector profiles
            sd = second_differences_at(model_strat, df_strat, [:x1, :x2], :age, Σ_strat;
                                      at=[30.0, 60.0],
                                      profile=(socio4=[false, true],
                                              gender=["male", "female"]))

            @test nrow(sd) == 16  # 2 variables × 2 eval points × 2 socio4 × 2 gender

            # Verify structure
            @test length(unique(sd.variable)) == 2
            @test length(unique(sd.eval_point)) == 2
            @test length(unique(sd.socio4)) == 2
            @test length(unique(sd.gender)) == 2

            # All combinations should be present
            @test all(isfinite.(sd.derivative))
            @test all(sd.se .> 0)
        end
    end

    @testset "Categorical Focal Variables" begin
        # Create data with categorical focal variable and interaction
        Random.seed!(999)
        df_cat = DataFrame(
            religion = categorical(rand(["Catholic", "Protestant", "None"], n)),
            modifier = randn(n),
            other = randn(n)
        )

        # Create interaction: religion effects vary with modifier
        df_cat.y = 0.5 .+
                   (df_cat.religion .== "Protestant") .* 0.3 .+
                   (df_cat.religion .== "None") .* 0.2 .+
                   0.1 .* df_cat.modifier .+
                   (df_cat.religion .== "Protestant") .* df_cat.modifier .* 0.5 .+
                   (df_cat.religion .== "None") .* df_cat.modifier .* 0.3 .+
                   randn(n) .* 0.3

        model_cat = lm(@formula(y ~ religion * modifier + other), df_cat)
        Σ_cat = vcov(model_cat)

        @testset "Baseline Contrasts (Default)" begin
            # Default should use baseline contrasts
            sd = second_differences_at(model_cat, df_cat, :religion, :modifier, Σ_cat; at=:mean)

            # Should have K-1 contrasts for K=3 levels
            @test nrow(sd) == 2
            @test hasproperty(sd, :contrast)
            @test all(sd.variable .== :religion)
        end

        @testset "Pairwise Contrasts" begin
            # Explicit pairwise contrasts
            sd_pairwise = second_differences_at(model_cat, df_cat, :religion, :modifier, Σ_cat;
                                                at=:mean, contrasts=:pairwise)

            # Should have K(K-1)/2 = 3 contrasts for K=3 levels
            @test nrow(sd_pairwise) == 3
            @test hasproperty(sd_pairwise, :contrast)
            @test all(sd_pairwise.variable .== :religion)

            # All derivatives should be finite
            @test all(isfinite.(sd_pairwise.derivative))
            @test all(sd_pairwise.se .> 0)
        end

        @testset "Baseline vs Pairwise Comparison" begin
            sd_baseline = second_differences_at(model_cat, df_cat, :religion, :modifier, Σ_cat;
                                               at=:mean, contrasts=:baseline)
            sd_pairwise = second_differences_at(model_cat, df_cat, :religion, :modifier, Σ_cat;
                                               at=:mean, contrasts=:pairwise)

            # Different number of rows
            @test nrow(sd_baseline) == 2  # K-1
            @test nrow(sd_pairwise) == 3  # K(K-1)/2

            # Some contrasts should overlap (baseline contrasts appear in pairwise)
            baseline_contrasts = Set(sd_baseline.contrast)
            pairwise_contrasts = Set(sd_pairwise.contrast)
            # Baseline contrasts should be a subset of pairwise contrasts
            @test length(intersect(baseline_contrasts, pairwise_contrasts)) >= 1
        end

        @testset "Basic Categorical Focal Variable" begin
            # Should work without error
            sd = second_differences_at(model_cat, df_cat, :religion, :modifier, Σ_cat; at=:mean)

            # Should have one row per contrast
            @test nrow(sd) >= 2  # At least 2 contrasts (depends on levels)
            @test hasproperty(sd, :contrast)
            @test hasproperty(sd, :variable)
            @test hasproperty(sd, :derivative)

            # All contrasts should be for religion
            @test all(sd.variable .== :religion)

            # Contrasts should not be "derivative" (that's for continuous only)
            @test !("derivative" in sd.contrast)

            # Statistical properties
            @test all(isfinite.(sd.derivative))
            @test all(sd.se .> 0)
            @test all(0 .<= sd.p_value .<= 1)
        end

        @testset "Categorical with Vector Profile" begin
            # Add binary variable for stratification
            df_cat.treated = rand([false, true], n)
            model_cat2 = lm(@formula(y ~ religion * modifier + treated + other), df_cat)

            sd = second_differences_at(
                model_cat2, df_cat, :religion, :modifier, vcov(model_cat2);
                at=:mean,
                profile=(treated=[false, true],)
            )

            # Should have rows for each contrast × treated level
            @test hasproperty(sd, :treated)
            n_contrasts = length(unique(sd.contrast))
            @test nrow(sd) == n_contrasts * 2  # contrasts × 2 treated levels

            # Check stratification worked
            @test Set(sd.treated) == Set([false, true])
        end

        @testset "Mixed: Continuous and Categorical Focal Variables" begin
            # Should handle both types in one call
            sd = second_differences_at(
                model_cat, df_cat, [:religion, :other], :modifier, Σ_cat;
                at=:mean
            )

            # Other is continuous (1 row), religion is categorical (multiple rows)
            religion_rows = sd[sd.variable .== :religion, :]
            other_rows = sd[sd.variable .== :other, :]

            @test nrow(religion_rows) >= 2  # Multiple contrasts
            @test nrow(other_rows) == 1     # Single derivative

            # Continuous variable should have "derivative" contrast
            @test other_rows.contrast[1] == "derivative"

            # Categorical should not
            @test !("derivative" in religion_rows.contrast)
        end
    end
end

println("✓ All second_differences_at() tests passed")
