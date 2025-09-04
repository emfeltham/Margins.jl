using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Error handling and input validation" begin
    Random.seed!(131415)
    n = 100
    df = DataFrame(
        y = Float64.(randn(n)),  # Ensure Float64
        x = Float64.(randn(n)),  # Ensure Float64
        z = Float64.(randn(n))   # Ensure Float64
    )

    m = lm(@formula(y ~ x + z), df)

    # Test invalid type parameter
    @testset "Invalid type parameter" begin
        @test_throws ArgumentError population_margins(m, df; type=:invalid)
        @test_throws ArgumentError profile_margins(m, df, means_grid(df); type=:invalid)
    end

    # Test invalid scale parameter (deprecated target parameter) 
    @testset "Invalid scale parameter (deprecated target)" begin
        @test_throws MethodError population_margins(m, df; type=:effects, target=:invalid)  # target param no longer exists
    end

    # Test invalid scale parameter
    @testset "Invalid scale parameter" begin
        @test_throws ArgumentError population_margins(m, df; type=:predictions, scale=:invalid)
    end

    # Test empty vars
    @testset "Empty vars parameter" begin
        # Empty vars should throw informative error
        @test_throws ArgumentError population_margins(m, df; type=:effects, vars=Symbol[])
    end

    # Test invalid variable names
    @testset "Invalid variable names" begin
        @test_throws Margins.MarginsError population_margins(m, df; type=:effects, vars=[:nonexistent])
    end

    # Test mismatched model and data
    @testset "Mismatched model and data" begin
        df_wrong = DataFrame(a=randn(n), b=randn(n))
        @test_throws Exception population_margins(m, df_wrong; type=:effects)
    end

    # Test invalid reference grid for profile_margins
    @testset "Invalid reference grid" begin
        invalid_grid = DataFrame(nonexistent_var=[1.0])  # Variable not in data
        @test_throws Exception profile_margins(m, df, invalid_grid; type=:effects)
    end

    # Test data type compatibility
    @testset "Data type compatibility" begin
        df_mixed = DataFrame(
            y = randn(50),
            x = rand(1:10, 50),  # Integer
            z = rand(Bool, 50)   # Boolean
        )
        m_mixed = lm(@formula(y ~ x + z), df_mixed)
        
        # Mixed data types should work fine (Int/Bool are properly handled)
        result = profile_margins(m_mixed, df_mixed, means_grid(df_mixed); type=:effects)
        @test nrow(DataFrame(result)) >= 1  # Should succeed
    end
end