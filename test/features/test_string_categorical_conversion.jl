# test_string_categorical_conversion.jl
# Tests for automatic string-to-categorical conversion in reference grids

using Test
using Margins
using DataFrames, GLM, CategoricalArrays, Random
using Tables

# Import the function we're testing (may need adjustment based on module structure)
using Margins: process_reference_grid

@testset "String-to-Categorical Conversion" begin
    Random.seed!(12345)
    
    # Setup test data with mixed variable types
    n = 100
    test_data = DataFrame(
        # Categorical variables
        education = categorical(rand(["High School", "College", "Graduate"], n)),
        region = categorical(rand(["North", "South", "East", "West"], n)),
        ordered_rating = categorical(rand(["Poor", "Fair", "Good", "Excellent"], n), ordered=true),
        
        # Continuous variables  
        age = rand(25:65, n),
        income = rand(30000:120000, n),
        
        # Boolean variables
        treatment = rand(Bool, n),
        
        # Outcome
        outcome = randn(n)
    )
    
    # Fit test model
    model = lm(@formula(outcome ~ age + income + education + region + treatment), test_data)
    
    @testset "Basic String Conversion" begin
        # Test grid with strings for categorical variables
        grid = DataFrame(
            education = ["High School", "College"],
            region = ["North", "South"],
            age = [30, 50],
            treatment = [true, false]
        )
        
        processed_grid = process_reference_grid(test_data, grid)
        
        # Check that categorical columns are properly converted
        @test isa(processed_grid.education, CategoricalVector)
        @test isa(processed_grid.region, CategoricalVector)
        
        # Check that non-categorical columns are unchanged
        @test eltype(processed_grid.age) == Int64
        @test eltype(processed_grid.treatment) == Bool
        
        # Check that levels match original data
        @test levels(processed_grid.education) == levels(test_data.education)
        @test levels(processed_grid.region) == levels(test_data.region)
        
        # Check that values are correctly converted
        @test string.(processed_grid.education) == ["High School", "College"]
        @test string.(processed_grid.region) == ["North", "South"]
    end
    
    @testset "Ordered Categorical Preservation" begin
        # Test with ordered categorical
        grid = DataFrame(ordered_rating = ["Poor", "Excellent"])
        processed_grid = process_reference_grid(test_data, grid)
        
        @test isordered(processed_grid.ordered_rating)
        @test levels(processed_grid.ordered_rating) == levels(test_data.ordered_rating)
        @test string.(processed_grid.ordered_rating) == ["Poor", "Excellent"]
    end
    
    @testset "Error Handling - Invalid Levels" begin
        # Test with invalid education level
        grid = DataFrame(education = ["High School", "PhD"])  # "PhD" not in original levels
        
        @test_throws ErrorException process_reference_grid(test_data, grid)
        
        # Check error message is informative
        try
            process_reference_grid(test_data, grid)
        catch e
            @test occursin("Level 'PhD' not found", e.msg)
            @test occursin("Available levels:", e.msg)
        end
    end
    
    @testset "Mixed Column Types" begin
        # Test with mix of categorical strings, non-categorical values, and missing columns
        grid = DataFrame(
            education = ["College", "Graduate"],           # Categorical - should convert
            age = [35, 45],                              # Numeric - should pass through  
            income = [50000.0, 75000.0],                 # Numeric - should pass through
            nonexistent = [1, 2]                         # Doesn't exist in data - should pass through
        )
        
        processed_grid = process_reference_grid(test_data, grid)
        
        # Categorical column should be converted
        @test isa(processed_grid.education, CategoricalVector)
        @test levels(processed_grid.education) == levels(test_data.education)
        
        # Numeric columns should be unchanged
        @test eltype(processed_grid.age) == Int64
        @test eltype(processed_grid.income) == Float64
        
        # Non-existent column should be unchanged
        @test eltype(processed_grid.nonexistent) == Int64
    end
    
    @testset "Empty Grid" begin
        # Test with empty grid
        empty_grid = DataFrame()
        processed_empty = process_reference_grid(test_data, empty_grid)
        @test nrow(processed_empty) == 0
        @test ncol(processed_empty) == 0
    end
    
    @testset "No Categorical Columns in Grid" begin
        # Test with grid containing no categorical columns
        grid = DataFrame(
            age = [30, 40, 50],
            income = [40000, 60000, 80000],
            treatment = [true, false, true]
        )
        
        processed_grid = process_reference_grid(test_data, grid)
        
        # Should be identical to input (no conversions needed)
        @test processed_grid.age == grid.age
        @test processed_grid.income == grid.income
        @test processed_grid.treatment == grid.treatment
    end
    
    @testset "Integration with cartesian_grid" begin
        # Test that the conversion works with actual cartesian_grid output
        using Margins: cartesian_grid
        
        # Create a cartesian grid with string specifications
        grid = cartesian_grid(
            education = ["High School", "Graduate"],
            treatment = [true, false],
            age = [30, 50]
        )
        
        processed_grid = process_reference_grid(test_data, grid)
        
        # Should have proper categorical conversion
        @test isa(processed_grid.education, CategoricalVector)
        @test levels(processed_grid.education) == levels(test_data.education)
        
        # Should preserve the cartesian product structure
        @test nrow(processed_grid) == 2 * 2 * 2  # 2 education × 2 treatment × 2 age
    end
    
    @testset "Profile Margins Integration" begin
        # Test full integration with profile_margins - this should work after we integrate
        # For now, just test that processed grids work with profile_margins
        
        grid = DataFrame(
            age = [35],
            income = [60000],
            treatment = [true]
        )
        
        # This should work (no categorical conversion needed)
        result = profile_margins(model, test_data, grid; type=:effects, vars=[:age])
        @test isa(result, Margins.MarginsResult)
        
        # Test with manually processed categorical grid
        cat_grid = DataFrame(
            education = [test_data.education[1]],  # Use actual categorical value
            age = [35],
            treatment = [true]
        )
        
        result2 = profile_margins(model, test_data, cat_grid; type=:effects, vars=[:age])
        @test isa(result2, Margins.MarginsResult)
    end
    
    @testset "Edge Cases" begin
        # Test with single row grid
        single_row_grid = DataFrame(education = ["College"])
        processed_single = process_reference_grid(test_data, single_row_grid)
        @test nrow(processed_single) == 1
        @test isa(processed_single.education, CategoricalVector)
        
        # Test with duplicate values
        dup_grid = DataFrame(education = ["College", "College", "High School"])
        processed_dup = process_reference_grid(test_data, dup_grid)
        @test nrow(processed_dup) == 3
        @test string.(processed_dup.education) == ["College", "College", "High School"]
    end
end

@testset "CategoricalArrays Integration" begin
    # Test specific CategoricalArrays functionality
    Random.seed!(12345)
    
    # Create data with specific level ordering
    education_levels = ["Elementary", "High School", "College", "Graduate"]
    data = DataFrame(
        education = categorical(rand(education_levels, 50), levels=education_levels, ordered=false)
    )
    
    @testset "Level Ordering Preservation" begin
        # Test that custom level ordering is preserved
        grid = DataFrame(education = ["Graduate", "Elementary", "College"])  # Different order than levels
        processed = process_reference_grid(data, grid)
        
        # Should preserve original level ordering, not grid ordering
        @test levels(processed.education) == education_levels
        @test string.(processed.education) == ["Graduate", "Elementary", "College"]  # Values in grid order
    end
    
    @testset "Ordered Property Preservation" begin
        # Test with ordered categorical
        ordered_data = DataFrame(
            rating = categorical(["Good", "Excellent", "Poor"], levels=["Poor", "Fair", "Good", "Excellent"], ordered=true)
        )
        
        grid = DataFrame(rating = ["Excellent", "Poor"])
        processed = process_reference_grid(ordered_data, grid)
        
        @test isordered(processed.rating)
        @test levels(processed.rating) == ["Poor", "Fair", "Good", "Excellent"]
    end
end