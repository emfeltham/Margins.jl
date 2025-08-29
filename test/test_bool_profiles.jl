using Test
using Random
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
        result = profile_margins(m, df; at=:means, type=:effects, vars=[:x])
        
        @test nrow(result.table) == 1
        @test all(isfinite, result.table.dydx)
        @test all(result.table.se .> 0)
    end
    
    @testset "Bool Fractional Values" begin
        # Test explicit fractional Bool specifications
        scenarios = Dict(
            :treated => [0.0, 0.3, 0.7, 1.0],
            :urban => [0.0, 1.0]
        )
        
        result = profile_margins(m, df; at=scenarios, type=:effects, vars=[:x])
        
        # Should create 4×2 = 8 scenarios
        @test nrow(result.table) == 8
        @test all(isfinite, result.table.dydx)
        @test all(result.table.se .> 0)
        
        # Check profile columns are present
        @test "at_treated" in names(result.table)
        @test "at_urban" in names(result.table)
        
        # Check fractional values are preserved
        @test Set(result.table.at_treated) == Set([0.0, 0.3, 0.7, 1.0])
        @test Set(result.table.at_urban) == Set([0.0, 1.0])
    end
    
    @testset ":all Specification with Bool Columns" begin
        # :all should only affect Real (non-Bool) columns
        scenarios = Dict(
            :all => :mean,           # Apply to x, z only (not treated, urban)
            :treated => [0, 1],      # Override Bool column
            :urban => 0.5            # Fractional Bool value
        )
        
        result = profile_margins(m, df; at=scenarios, type=:effects, vars=[:x])
        
        # Should create 2×1 = 2 scenarios (treated=[0,1], urban=0.5)
        @test nrow(result.table) == 2
        @test all(isfinite, result.table.dydx)
        
        # Check that Real columns got mean values, Bool got specified values
        @test "at_x" in names(result.table)
        @test "at_z" in names(result.table) 
        @test "at_treated" in names(result.table)
        @test "at_urban" in names(result.table)
        
        # urban should be 0.5 for all rows
        @test all(result.table.at_urban .== 0.5)
        # treated should be [0, 1]
        @test Set(result.table.at_treated) == Set([0, 1])
    end
    
    @testset "Mixed Variable Types in Profiles" begin
        # Test complex scenario with all variable types
        scenarios = Dict(
            :x => [-1, 0, 1],        # Continuous Float64
            :z => :mean,             # Summary statistic
            :treated => [0.0, 0.4, 1.0],  # Fractional Bool
            :urban => [false, true]   # Bool literals
        )
        
        result = profile_margins(m, df; at=scenarios, type=:predictions, scale=:response)
        
        # Should create 3×1×3×2 = 18 scenarios
        @test nrow(result.table) == 18
        @test all(0.0 .<= result.table.dydx .<= 1.0)  # Response scale bounded [0,1]
        @test all(isfinite, result.table.dydx)
        
        # Check all profile columns present
        @test "at_x" in names(result.table)
        @test "at_z" in names(result.table)
        @test "at_treated" in names(result.table) 
        @test "at_urban" in names(result.table)
    end
    
    @testset "Bool Column Default Behavior" begin
        # When at=:means, Bool columns should default to false
        result = profile_margins(m, df; at=:means, type=:predictions, scale=:response)
        
        @test nrow(result.table) == 1
        
        # Check that all variables get profile columns with :means
        profile_cols = filter(n -> startswith(string(n), "at_"), names(result.table))
        
        # Should have profile columns for all variables in the model
        expected_profile_cols = ["at_x", "at_z", "at_treated", "at_urban", "at_y"]
        @test sort(profile_cols) == sort(expected_profile_cols)
        
        # Bool columns should be set to false (default value)
        @test result.table.at_treated[1] == false
        @test result.table.at_urban[1] == false
    end
end

# Bool derivative computation is handled by FormulaCompiler and is out of scope
# for this profile-specific fix. The main issue was Bool columns in profile scenarios
# causing InexactError when converting Float64 means back to Bool.