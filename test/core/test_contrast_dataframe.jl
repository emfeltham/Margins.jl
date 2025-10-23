using Test
using DataFrames, GLM
using Margins
using Random

@testset "DataFrame method for contrast results" begin
    Random.seed!(456)
    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = randn(n),
        treatment = rand(0:1, n)
    )

    # Fit model
    model = lm(@formula(y ~ x + z + treatment), df)

    @testset "Contrast from MarginsResult" begin
        result = profile_margins(model, df, cartesian_grid(x=[0, 1]); type=:predictions)
        contrast_result = contrast(result, 1, 2, vcov(model))

        # Test basic DataFrame conversion
        df_contrast = DataFrame(contrast_result)
        @test size(df_contrast) == (1, 10)  # Updated: now includes row1 and row2
        @test all(col in names(df_contrast) for col in ["contrast", "se", "t_stat", "p_value",
                                                         "ci_lower", "ci_upper", "estimate1", "estimate2",
                                                         "row1", "row2"])
        @test all(isfinite, df_contrast.contrast)
        @test all(isfinite, df_contrast.se)
        @test df_contrast.contrast[1] ≈ contrast_result.contrast
        @test df_contrast.se[1] ≈ contrast_result.se
        @test df_contrast.row1[1] == 1
        @test df_contrast.row2[1] == 2

        # Verify no gradient column by default
        @test !("gradient" in names(df_contrast))
    end

    @testset "Contrast from DataFrame with integer indices" begin
        result = profile_margins(model, df, cartesian_grid(treatment=[0, 1]); type=:predictions)
        df_pred = DataFrame(result; include_gradients=true)

        contrast_result = contrast(df_pred, 1, 2, vcov(model))

        # Convert without gradient
        df_contrast = DataFrame(contrast_result)
        @test size(df_contrast, 1) == 1
        @test all(col in names(df_contrast) for col in ["contrast", "se", "t_stat", "p_value"])
        @test !("gradient" in names(df_contrast))

        # Convert with gradient
        df_contrast_grad = DataFrame(contrast_result; include_gradient=true)
        @test "gradient" in names(df_contrast_grad)
        @test length(df_contrast_grad.gradient[1]) == length(coef(model))
        @test df_contrast_grad.gradient[1] ≈ contrast_result.gradient
    end

    @testset "Contrast from DataFrame with NamedTuple specification" begin
        result = profile_margins(model, df, cartesian_grid(x=[0, 1], treatment=[0, 1]);
                                type=:predictions)
        df_pred = DataFrame(result; include_gradients=true)

        # Contrast at treatment=1 across x values
        contrast_result = contrast(df_pred, (x=1, treatment=1), (x=0, treatment=1), vcov(model))

        df_contrast = DataFrame(contrast_result)
        @test size(df_contrast, 1) == 1
        @test all(col in names(df_contrast) for col in ["contrast", "se", "row1", "row2"])
        @test df_contrast.row1[1] == contrast_result.row1
        @test df_contrast.row2[1] == contrast_result.row2

        # With gradient
        df_contrast_grad = DataFrame(contrast_result; include_gradient=true)
        @test "gradient" in names(df_contrast_grad)
        @test length(df_contrast_grad.gradient[1]) == length(coef(model))
    end

    @testset "Contrast with effects" begin
        # Test with effects instead of predictions
        result = profile_margins(model, df, cartesian_grid(treatment=[0, 1]);
                                type=:effects, vars=[:x])
        df_eff = DataFrame(result; include_gradients=true)

        contrast_result = contrast(df_eff, (treatment=1,), (treatment=0,), vcov(model))
        df_contrast = DataFrame(contrast_result)

        @test size(df_contrast, 1) == 1
        @test all(isfinite, df_contrast.contrast)
        @test all(isfinite, df_contrast.se)
    end

    @testset "Statistical content validation" begin
        # Verify statistical content is preserved correctly
        result = profile_margins(model, df, cartesian_grid(x=[-1, 0, 1]); type=:predictions)
        contrast_result = contrast(result, 1, 3, vcov(model))
        df_contrast = DataFrame(contrast_result)

        # Check confidence interval consistency
        z_crit = 1.96  # approximately
        manual_ci_lower = df_contrast.contrast[1] - z_crit * df_contrast.se[1]
        manual_ci_upper = df_contrast.contrast[1] + z_crit * df_contrast.se[1]
        @test df_contrast.ci_lower[1] ≈ manual_ci_lower atol=0.01
        @test df_contrast.ci_upper[1] ≈ manual_ci_upper atol=0.01

        # Check t-statistic
        manual_t = df_contrast.contrast[1] / df_contrast.se[1]
        @test df_contrast.t_stat[1] ≈ manual_t
    end

    @testset "Edge cases" begin
        # Test with all available fields
        result = profile_margins(model, df, cartesian_grid(x=[0, 1]); type=:predictions)
        df_pred = DataFrame(result; include_gradients=true)

        # Use NamedTuple spec to get row1, row2, and gradient
        contrast_result = contrast(df_pred, (x=1,), (x=0,), vcov(model))
        df_full = DataFrame(contrast_result; include_gradient=true)

        # Should have all possible columns
        expected_cols = ["contrast", "se", "t_stat", "p_value", "ci_lower", "ci_upper",
                        "estimate1", "estimate2", "row1", "row2", "gradient"]
        @test all(col in names(df_full) for col in expected_cols)
        @test size(df_full) == (1, length(expected_cols))
    end
end
