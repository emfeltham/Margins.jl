# Test for boolean variable auto-detection in second_differences
# Tests the fix for auto-detecting categorical contrasts

using Test
using Margins
using DataFrames
using GLM
using Random
using CategoricalArrays

@testset "Second Differences - Boolean Variable Auto-detection" begin
    Random.seed!(12345)
    n = 200

    df = DataFrame(
        x = randn(n),
        treated = rand(Bool, n),  # Boolean variable
        region = rand(["north", "south", "west"], n)
    )

    # Convert to categorical
    df.treated_cat = categorical(df.treated)

    # Model with interaction
    df.y = 0.5 .+ 0.3 .* df.x .+
           0.4 .* Float64.(df.treated) .+
           0.2 .* df.x .* Float64.(df.treated) .+
           randn(n) .* 0.4

    model = lm(@formula(y ~ x * treated_cat), df)
    Σ = vcov(model)

    @testset "Boolean variable with categorical modifier" begin
        # Create AME with boolean as focal variable, region as modifier
        ames = population_margins(
            model, df;
            vars=:treated_cat,
            scenarios=(region=["north", "south", "west"],),
            type=:effects,
            contrasts=:pairwise
        )

        # Check that the contrast is NOT "derivative"
        df_ames = DataFrame(ames)
        contrasts = unique(filter(row -> row.variable == "treated_cat", df_ames).contrast)
        println("Contrasts found: ", contrasts)
        @test length(contrasts) == 1
        @test contrasts[1] != "derivative"  # Should be "true - false" or similar

        # This should now work with the fix (auto-detects the contrast)
        sd = second_differences(ames, :treated_cat, :region, Σ; all_contrasts=true)

        # Should have 3 pairwise region comparisons
        @test nrow(sd) == 3
        @test all(isfinite.(sd.second_diff))
        @test all(sd.se .>= 0)
        @test all(0 .<= sd.p_value .<= 1)

        println("✓ Boolean variable auto-detection works!")
        println("  Found contrast: ", contrasts[1])
        println("  Computed ", nrow(sd), " second differences")
    end

    @testset "Manual contrast specification still works" begin
        # Create AME again
        ames = population_margins(
            model, df;
            vars=:treated_cat,
            scenarios=(region=["north", "south", "west"],),
            type=:effects,
            contrasts=:pairwise
        )

        # Get actual contrast
        df_ames = DataFrame(ames)
        actual_contrast = unique(filter(row -> row.variable == "treated_cat", df_ames).contrast)[1]

        # Explicit specification should also work
        sd_explicit = second_differences(
            ames, :treated_cat, :region, Σ;
            contrast=actual_contrast,
            all_contrasts=false
        )

        @test nrow(sd_explicit) == 3
        @test all(isfinite.(sd_explicit.second_diff))

        println("✓ Manual contrast specification still works!")
    end
end

println("\n✓ All boolean auto-detection tests passed!")
