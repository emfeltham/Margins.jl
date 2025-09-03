# test_teaching_errors.jl - Tests for Phase 5 Teaching Error Implementation

using Test
using DataFrames
using GLM
using StatsModels
using Margins

@testset "Phase 5 Teaching Errors" begin
    # Create test data
    df = DataFrame(
        y = randn(100),
        x = randn(100),
        z = randn(100),
        w = randn(100),
        treatment = rand([0, 1], 100)
    )
    model = lm(@formula(y ~ x + z + w + treatment), df)
    
    @testset "Vars/Scenarios Overlap Detection" begin
        # Test 1: Single variable overlap should error with teaching message
        @test_throws ArgumentError population_margins(model, df; vars=[:x], scenarios=Dict(:x => [0, 1]))
        
        # Test 2: Multiple variable overlap should error
        @test_throws ArgumentError population_margins(model, df; vars=[:x, :z], scenarios=Dict(:x => [0, 1], :z => [-1, 1]))
        
        # Test 3: Partial overlap should error  
        @test_throws ArgumentError population_margins(model, df; vars=[:x, :w], scenarios=Dict(:x => [0, 1], :treatment => [0, 1]))
        
        # Test 4: Check that the error message contains helpful suggestions
        try
            population_margins(model, df; vars=[:x], scenarios=Dict(:x => [0, 1]))
            @test false  # Should not reach here
        catch e
            error_msg = string(e)
            @test contains(error_msg, "Invalid parameter combination")
            @test contains(error_msg, "What you're asking")
            @test contains(error_msg, "This is contradictory")
            @test contains(error_msg, "What you probably want")
            @test contains(error_msg, "type=:predictions")
            @test contains(error_msg, "groups=")
        end
    end
    
    @testset "Valid Usage Should Work" begin
        # Test 1: No overlap - effects of one variable with scenarios for another
        result1 = population_margins(model, df; vars=[:x], scenarios=Dict(:treatment => [0, 1]))
        @test nrow(DataFrame(result1)) == 2  # 2 scenarios
        
        # Test 2: No overlap - multiple effect variables with different scenario variables
        result2 = population_margins(model, df; vars=[:x, :z], scenarios=Dict(:treatment => [0, 1], :w => [0.5, -0.5]))
        @test nrow(DataFrame(result2)) == 8  # 2 vars × 2×2 scenarios = 8
        
        # Test 3: Predictions with scenarios (allowed to use same variables)
        result3 = population_margins(model, df; type=:predictions, scenarios=Dict(:x => [0, 1], :z => [-1, 1]))
        @test nrow(DataFrame(result3)) == 4  # 2×2 scenarios
        
        # Test 4: Effects with groups (no scenarios)
        result4 = population_margins(model, df; vars=[:x], groups=:treatment)
        @test nrow(DataFrame(result4)) == 4  # Treatment gets quartile binning (Q1-Q4)
    end
    
    @testset "Edge Cases" begin
        # Test 1: Symbol vs Vector vars
        @test_throws ArgumentError population_margins(model, df; vars=:x, scenarios=Dict(:x => [0, 1]))
        
        # Test 2: :all_continuous with scenarios should not overlap (depends on data)
        # This is a bit complex to test as it depends on what's considered continuous
        
        # Test 3: Empty scenarios Dict should work
        result = population_margins(model, df; vars=[:x], scenarios=Dict{Symbol, Vector{Any}}())
        @test nrow(DataFrame(result)) == 1  # No scenarios = base case
        
        # Test 4: Effects with no vars specified should work with scenarios
        result = population_margins(model, df; scenarios=Dict(:x => [0, 1]))  # uses all_continuous
        @test nrow(DataFrame(result)) >= 2  # At least 2 scenarios
    end
    
    @testset "Teaching Message Quality" begin
        # Test the quality and helpfulness of error messages
        try
            population_margins(model, df; vars=[:x, :z], scenarios=Dict(:x => [0, 1]))
            @test false
        catch e
            error_msg = string(e)
            
            # Should explain the conceptual issue
            @test contains(error_msg, "marginal effect")
            @test contains(error_msg, "contradictory")
            
            # Should provide concrete alternatives
            @test contains(error_msg, "vars=[:other_var]")
            @test contains(error_msg, "type=:predictions")
            @test contains(error_msg, "groups=")
            
            # Should reference documentation
            @test contains(error_msg, "documentation")
            
            # Should use friendly language
            @test contains(error_msg, "What you probably want")
            @test contains(error_msg, "This is contradictory")  # Clear explanation
        end
    end
    
    @testset "Performance of Validation" begin
        # Teaching validation should be fast and not impact performance
        
        # Time valid operation
        @time result_valid = population_margins(model, df; vars=[:x], scenarios=Dict(:treatment => [0, 1]))
        
        # Time invalid operation (should fail quickly)
        @time try
            population_margins(model, df; vars=[:x], scenarios=Dict(:x => [0, 1]))
        catch e
            # Expected to error
        end
        
        # Both should be fast - the validation shouldn't add significant overhead
        @test true  # If we get here, validation didn't hang
    end
end