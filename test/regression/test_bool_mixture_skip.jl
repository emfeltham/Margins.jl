# Test for Bool variable mixture skip fix
# Regression test for: https://github.com/...
# When a Bool variable is filled with fractional values during grid completion,
# it should be skipped for contrasts to avoid type mismatch errors.

using Test
using Margins, GLM, DataFrames, StatsModels
using Random

@testset "Bool Mixture Skip for Profile Margins" begin
    Random.seed!(123456)

    @testset "Bool variable filled by grid completion" begin
        # Create test data with multiple Bool variables
        n = 100
        df = DataFrame(
            x1 = randn(n),
            bool_explicit = rand(Bool, n),  # Will be in reference grid
            bool_implicit = rand(Bool, n),  # Will NOT be in reference grid
            y = randn(n)
        )

        # Fit model with both Bool variables
        model = lm(@formula(y ~ x1 + bool_explicit + bool_implicit), df)

        # Create reference grid that only specifies bool_explicit
        # bool_implicit will be filled with typical value (frequency) during grid completion
        ref_grid = DataFrame(
            x1 = [0.0, 1.0],
            bool_explicit = [false, true]
        )

        # This should work without error - bool_implicit should be skipped
        result = profile_margins(model, df, ref_grid; type=:effects)
        result_df = DataFrame(result)

        # Verify that only x1 has effects computed (not bool_implicit)
        @test nrow(result_df) == 2  # Only x1 at 2 profile points
        @test all(result_df.variable .== "x1")
        @test all(result_df.contrast .== "dy/dx")
    end

    @testset "Bool variable with 0.0 and 1.0 Float values (edge case)" begin
        # Edge case: Bool variable filled with exact 0.0 or 1.0 Float values
        # These should still be treated as discrete, not mixture
        n = 100
        df = DataFrame(
            bool_var = rand(Bool, n),
            x1 = randn(n),
            y = randn(n)
        )

        model = lm(@formula(y ~ bool_var + x1), df)

        # Create reference grid with Float values that are exactly 0.0 or 1.0
        ref_grid = DataFrame(
            bool_var = [0.0, 1.0],  # Float, not Bool, but values are exact
            x1 = [0.0, 0.0]
        )

        # This should compute effects for x1 only
        # bool_var with exact 0.0/1.0 is scenario-defining
        result = profile_margins(model, df, ref_grid; type=:effects)
        result_df = DataFrame(result)

        @test !any(result_df.variable .== "bool_var")
        @test all(result_df.variable .== "x1")
    end
end
