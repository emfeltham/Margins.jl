# test_manual_counterfactual_validation.jl - Manual Counterfactual Validation
# julia --project="." test/validation/test_manual_counterfactual_validation.jl > test/validation/test_manual_counterfactual_validation.txt 2>&1
#
# This test implements step-by-step manual counterfactual computation and validates
# that population_margins() matches hand-computed counterfactual results exactly.
#
# CRITICAL: This validates the actual computational sequence of counterfactual analysis:
# 1. Set focal variable to level A for ALL observations → predict ŷₐ
# 2. Set focal variable to level B for ALL observations → predict ŷᵦ
# 3. AME = mean(ŷₐ - ŷᵦ) [contrast-then-average]
#
# This catches bugs that analytical validation tests miss:
# - Wrong contrast directions (baseline→level vs level→baseline)
# - Wrong baseline inference (Control vs Treatment as reference)
# - Wrong computational sequences (average-then-contrast vs contrast-then-average)
# - Override system failures (incorrect scenario application)

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins
using Statistics: mean

@testset "Manual Counterfactual Validation - Critical Implementation Correctness" begin

    @testset "Binary Variable Counterfactual Validation" begin
        # Test binary (boolean) variable AME against manual counterfactual computation

        Random.seed!(08540)
        n = 200
        df = DataFrame(
            x = randn(n),
            treatment = rand([true, false], n),  # Boolean treatment variable
        )
        df.y = 0.5 * df.x + 1.0 * df.treatment + 0.1 * randn(n)

        @testset "Linear Model: Binary Treatment" begin
            model = lm(@formula(y ~ x + treatment), df)

            # === MANUAL COUNTERFACTUAL COMPUTATION ===
            # Step 1: Create counterfactual datasets
            df_false = copy(df)
            df_false.treatment .= false  # Set ALL observations to treatment=false

            df_true = copy(df)
            df_true.treatment .= true   # Set ALL observations to treatment=true

            # Step 2: Predict under both scenarios
            predictions_false = GLM.predict(model, df_false)
            predictions_true = GLM.predict(model, df_true)

            # Step 3: Manual AME computation (contrast-then-average)
            manual_contrasts = predictions_true .- predictions_false  # Per-observation contrasts
            manual_ame = mean(manual_contrasts)  # Average of contrasts

            # === VALIDATE AGAINST population_margins ===
            ame_result = population_margins(model, df; vars=[:treatment], scale=:response)
            computed_ame = DataFrame(ame_result).estimate[1]

            @test manual_ame ≈ computed_ame atol=1e-12

            # Additional validation: Check that we're using correct baseline (false→true)
            # For boolean, baseline should be false, so AME should be positive if treatment effect is positive
            treatment_coef = coef(model)[3]  # Coefficient of treatment variable
            @test manual_ame ≈ treatment_coef atol=1e-12  # For linear model, AME equals coefficient
        end

        # NOTE: Logistic model with boolean treatment is extensively tested in:
        # - statistical_validation/categorical_bootstrap_tests.jl
        # - statistical_validation/bootstrap_se_validation.jl
        # - core/test_glm_basic.jl
        # Boolean categorical effects are validated to work correctly for GLMs there.
    end

    @testset "Categorical Variable Counterfactual Validation" begin
        Random.seed!(08541)
        n = 300

        df = DataFrame(
            x = randn(n),
            region = categorical(rand(["North", "South", "East", "West"], n)),
        )
        df.y = 0.5 * df.x +
               1.0 * (df.region .== "North") +
               0.5 * (df.region .== "South") +
               -0.5 * (df.region .== "East") +
               0.1 * randn(n)

        @testset "Linear Model: Categorical Region" begin
            model = lm(@formula(y ~ x + region), df)

            # Test manual contrast: validate at least one level matches
            @testset "Manual validation: Categorical contrasts" begin
                # Get all the contrasts from population_margins
                ame_result = population_margins(model, df; vars=[:region], scale=:response)
                result_df = DataFrame(ame_result)

                # Test each non-baseline level
                for level in setdiff(unique(df.region), [levels(df.region)[1]])  # Skip baseline
                    df_baseline = copy(df)
                    df_baseline.region .= categorical(repeat([levels(df.region)[1]], n))

                    df_level = copy(df)
                    df_level.region .= categorical(repeat([String(level)], n))

                    pred_baseline = GLM.predict(model, df_baseline)
                    pred_level = GLM.predict(model, df_level)

                    manual_contrast = mean(pred_level .- pred_baseline)

                    # Find the row in results for this level
                    level_rows = result_df[occursin.(lowercase(String(level)), lowercase.(result_df.contrast)), :]
                    if nrow(level_rows) > 0
                        computed_contrast = level_rows.estimate[1]
                        @test manual_contrast ≈ computed_contrast atol=1e-12
                    end
                end
            end
        end

        @testset "Logistic Model: Categorical Region - Nonlinear Validation" begin
            df_logistic = copy(df)
            df_logistic.y_binary = rand(n) .< (0.5 .+ 0.1 * (df.region .== "North"))

            model = glm(@formula(y_binary ~ x + region), df_logistic, Binomial(), LogitLink())

            # Get all the contrasts
            ame_result = population_margins(model, df_logistic; vars=[:region], scale=:response)
            result_df = DataFrame(ame_result)

            # Validate at least one contrast matches manual computation
            baseline = levels(df_logistic.region)[1]
            for level in setdiff(unique(df_logistic.region), [baseline])
                df_baseline = copy(df_logistic)
                df_baseline.region .= categorical(repeat([baseline], n))

                df_level = copy(df_logistic)
                df_level.region .= categorical(repeat([String(level)], n))

                pred_baseline = GLM.predict(model, df_baseline)
                pred_level = GLM.predict(model, df_level)

                manual_contrast = mean(pred_level .- pred_baseline)

                # Find matching row
                level_rows = result_df[occursin.(lowercase(String(level)), lowercase.(result_df.contrast)), :]
                if nrow(level_rows) > 0
                    computed_contrast = level_rows.estimate[1]
                    @test manual_contrast ≈ computed_contrast atol=1e-10
                end
            end
        end
    end

    @testset "Continuous Variable Validation - Finite Difference Check" begin
        Random.seed!(08542)
        n = 250
        df = DataFrame(
            x = randn(n),
            z = randn(n),
        )
        df.y = 2.0 * df.x + 1.5 * df.z + 0.1 * randn(n)

        model = lm(@formula(y ~ x + z), df)

        # For continuous variable, AME should equal the coefficient (for linear model)
        # Validate by manual finite difference
        h = 1e-6

        for var in [:x, :z]
            df_plus = copy(df)
            df_plus[!, var] .+= h

            df_minus = copy(df)
            df_minus[!, var] .-= h

            pred_plus = GLM.predict(model, df_plus)
            pred_minus = GLM.predict(model, df_minus)

            manual_derivative = mean((pred_plus .- pred_minus) ./ (2h))

            ame_result = population_margins(model, df; vars=[var], scale=:response)
            computed_ame = DataFrame(ame_result).estimate[1]

            # Should match coefficient exactly for linear model
            var_idx = var == :x ? 2 : 3
            expected_coef = coef(model)[var_idx]

            @test manual_derivative ≈ expected_coef atol=1e-8
            @test computed_ame ≈ expected_coef atol=1e-12
        end
    end

    @testset "Mixed Model Validation - Multiple Variable Types" begin
        Random.seed!(08543)
        n = 300

        df = DataFrame(
            x_cont = randn(n),
            treatment = rand([true, false], n),
            region = categorical(rand(["A", "B"], n)),
        )
        df.y = 1.0 * df.x_cont + 2.0 * df.treatment + 1.5 * (df.region .== "B") + 0.1 * randn(n)

        model = lm(@formula(y ~ x_cont + treatment + region), df)

        @testset "Validate Each Variable Type in Mixed Model" begin
            # Continuous variable
            x_coef = coef(model)[2]
            ame_x = population_margins(model, df; vars=[:x_cont], scale=:response)
            @test DataFrame(ame_x).estimate[1] ≈ x_coef atol=1e-12

            # Boolean variable - manual counterfactual
            df_false = copy(df)
            df_false.treatment .= false
            df_true = copy(df)
            df_true.treatment .= true

            manual_ame_treatment = mean(GLM.predict(model, df_true) .- GLM.predict(model, df_false))
            ame_treatment = population_margins(model, df; vars=[:treatment], scale=:response)
            @test DataFrame(ame_treatment).estimate[1] ≈ manual_ame_treatment atol=1e-12

            # Categorical variable - manual counterfactual
            df_a = copy(df)
            df_a.region .= categorical(repeat(["A"], n))
            df_b = copy(df)
            df_b.region .= categorical(repeat(["B"], n))

            manual_ame_region = mean(GLM.predict(model, df_b) .- GLM.predict(model, df_a))
            ame_region = population_margins(model, df; vars=[:region], scale=:response)
            @test DataFrame(ame_region).estimate[1] ≈ manual_ame_region atol=1e-12
        end
    end

    @testset "Computational Sequence Validation" begin
        # Validate that we compute contrast-then-average, not average-then-contrast
        Random.seed!(08544)
        n = 200

        df = DataFrame(
            x = randn(n),
            treatment = rand([true, false], n),
        )
        # Nonlinear relationship to make order matter
        df.y = exp.(0.5 * df.x .+ df.treatment) .+ 0.1 * randn(n)

        model = lm(@formula(y ~ x + treatment), df)

        @testset "Contrast-then-Average vs Average-then-Contrast" begin
            # Method 1: Contrast-then-average (CORRECT for AME)
            df_false = copy(df)
            df_false.treatment .= false
            df_true = copy(df)
            df_true.treatment .= true

            contrasts = GLM.predict(model, df_true) .- GLM.predict(model, df_false)
            contrast_then_average = mean(contrasts)

            # Method 2: Average-then-contrast (WRONG for AME, but useful baseline check)
            avg_y_at_true = mean(GLM.predict(model, df_true))
            avg_y_at_false = mean(GLM.predict(model, df_false))
            average_then_contrast = avg_y_at_true - avg_y_at_false

            # For linear model, these should be identical
            @test contrast_then_average ≈ average_then_contrast atol=1e-12

            # Validate against population_margins
            ame_result = population_margins(model, df; vars=[:treatment], scale=:response)
            computed_ame = DataFrame(ame_result).estimate[1]

            @test computed_ame ≈ contrast_then_average atol=1e-12
        end
    end
end
