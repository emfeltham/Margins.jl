using Test
using Random
using Statistics
using DataFrames, GLM, StatsModels
using Margins

@testset "Bool Column Profile Handling" begin
    Random.seed!(12345)
    n = 200
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        treated = rand(Bool, n),
        urban = rand(Bool, n),
        y = rand(Bool, n)
    )
    
    m = glm(@formula(y ~ x + z + treated + urban), df, Binomial(), LogitLink())
    
    @testset "Profile :means with Bool Columns" begin
        # Should not throw InexactError when Bool columns are present
        result = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x])
        
        @test nrow(DataFrame(result)) == 1
        @test all(isfinite, DataFrame(result).estimate)
        @test all(DataFrame(result).se .> 0)
    end
    
    @testset "Bool Fractional Values" begin
        # Test explicit fractional Bool specifications
        result = profile_margins(m, df, cartesian_grid(treated=[0.0, 0.3, 0.7, 1.0], urban=[0.0, 1.0]); type=:effects, vars=[:x])
        
        # Should create 4×2 = 8 scenarios
        @test nrow(DataFrame(result)) == 8
        @test all(isfinite, DataFrame(result).estimate)
        @test all(DataFrame(result).se .> 0)
        
        # Check profile columns are present
        @test "treated" in names(DataFrame(result))
        @test "urban" in names(DataFrame(result))
        
        # Check fractional values are preserved
        @test Set(DataFrame(result).treated) == Set([0.0, 0.3, 0.7, 1.0])
        @test Set(DataFrame(result).urban) == Set([0.0, 1.0])
    end
    
    @testset ":all Specification with Bool Columns" begin
        # Test with mixed specifications
        result = profile_margins(m, df, cartesian_grid(x=mean(df.x), z=mean(df.z), treated=[0, 1], urban=0.5); type=:effects, vars=[:x])
        
        # Should create 2×1 = 2 scenarios (treated=[0,1], urban=0.5)
        @test nrow(DataFrame(result)) == 2
        @test all(isfinite, DataFrame(result).estimate)
        
        # Check that Real columns got mean values, Bool got specified values (profile format uses direct names)
        @test "x" in names(DataFrame(result))
        @test "z" in names(DataFrame(result)) 
        @test "treated" in names(DataFrame(result))
        @test "urban" in names(DataFrame(result))
        
        # urban should be 0.5 for all rows
        @test all(DataFrame(result).urban .== 0.5)
        # treated should be [0, 1]
        @test Set(DataFrame(result).treated) == Set([0, 1])
    end
    
    @testset "Mixed Variable Types in Profiles" begin
        # Test complex scenario with all variable types
        result = profile_margins(m, df, cartesian_grid(x=[-1, 0, 1], z=mean(df.z), treated=[0.0, 0.4, 1.0], urban=[false, true]); type=:predictions, scale=:response)
        
        # Should create 3×1×3×2 = 18 scenarios
        @test nrow(DataFrame(result)) == 18
        @test all(0.0 .<= DataFrame(result).estimate .<= 1.0)  # Response scale bounded [0,1]
        @test all(isfinite, DataFrame(result).estimate)
        
        # Check all profile columns present (bare names for profile analysis)
        @test "x" in names(DataFrame(result))
        @test "z" in names(DataFrame(result))
        @test "treated" in names(DataFrame(result)) 
        @test "urban" in names(DataFrame(result))
    end
    
    @testset "Bool Column Default Behavior" begin
        # When using means grid, Bool columns should default to false
        result = profile_margins(m, df, means_grid(df); type=:predictions, scale=:response)
        
        @test nrow(DataFrame(result)) == 1
        
        # Check that all variables are present in the result
        result_df = DataFrame(result)
        
        # Should have columns for all variables in the model (bare names for profile analysis)
        expected_cols = ["x", "z", "treated", "urban"]
        @test all(col -> col in names(result_df), expected_cols)
        
        # Bool columns should be set to frequency-weighted values (new behavior)  
        @test 0.0 <= result_df.treated[1] <= 1.0  # Should be probability of true
        @test 0.0 <= result_df.urban[1] <= 1.0    # Should be probability of true
        @test typeof(result_df.treated[1]) == Float64
        @test typeof(result_df.urban[1]) == Float64
    end
end

# Bool derivative computation is handled by FormulaCompiler and is out of scope
# for this profile-specific fix. The main issue was Bool columns in profile scenarios
# causing InexactError when converting Float64 means back to Bool.