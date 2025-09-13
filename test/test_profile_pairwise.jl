# Test case for profile pairwise contrasts functionality
using Random, DataFrames, CategoricalArrays, GLM, Margins

@testset "Profile pairwise contrasts" begin
    Random.seed!(789)
    n = 50
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        cat_var = categorical(rand(["Low", "Medium", "High"], n))
    )
    m = lm(@formula(y ~ x + cat_var), df)
    grid = means_grid(df)
    
    # Test basic functionality
    result = profile_margins(m, df, grid; type=:effects, vars=[:cat_var], contrasts=:pairwise)
    df_result = DataFrame(result)
    
    @test nrow(df_result) >= 1  # Should have contrasts
    @test all(isfinite, df_result.estimate)
    @test all(isfinite, df_result.se)
    
    # Test result structure
    @test "contrast" in names(df_result)
    @test all(contains.(df_result.contrast, " vs "))  # Should have "X vs Y" format
    
    # Test that we get the expected number of pairwise contrasts
    n_levels = length(levels(df.cat_var))
    expected_contrasts = (n_levels * (n_levels - 1)) ÷ 2  # n-choose-2
    @test nrow(df_result) == expected_contrasts
    
    # Test error handling
    @test_throws ArgumentError profile_margins(m, df, grid; type=:effects, vars=[:cat_var], contrasts=:invalid)
    
    println("✅ Profile pairwise contrasts test passed!")
end