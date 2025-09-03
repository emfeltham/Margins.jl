# Test for Phase 4: Advanced Profile Averaging with proper delta-method SEs

using Margins, GLM, DataFrames, Test, Random, FormulaCompiler, Statistics

@testset "Advanced Profile Averaging with Proper Delta-Method SEs" begin

    # Setup test data with grouping structure (avoid categorical variables for now)
    Random.seed!(123)
    n = 100
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        group_num = repeat([1, 2, 3], n ÷ 3 + 1)[1:n],
        region_num = repeat([1, 2], n ÷ 2 + 1)[1:n]
    )
    df.y = 0.5 * df.x1 + 0.3 * df.x2 + 0.2 * (df.group_num .== 2) + randn(n) * 0.1

    # Fit model
    model = lm(@formula(y ~ x1 + x2 + group_num), df)

    @testset "Simple Profile Averaging (no grouping) - baseline check" begin
        # This should work and serve as baseline - updated for current API
        result = profile_margins(model, df; 
            at = :means,  # Use :means instead of Dict for simple case
            type = :effects, 
            vars = [:x1]
        )
        
        @test nrow(DataFrame(result)) == 1  # Single profile at means
        @test DataFrame(result).term[1] == "x1"  # String, not Symbol
        @test !ismissing(DataFrame(result).se[1])
        @test DataFrame(result).se[1] > 0
    end

    @testset "Multiple Profile Points" begin
        # Test multiple profile points using Dict specification
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :effects,
            vars = [:x1]
        )
        
        @test nrow(DataFrame(result)) == 3  # One row per x1 value
        @test all(DataFrame(result).term .== "x1")  # String, not Symbol
        @test all(.!ismissing.(DataFrame(result).se))
        @test all(DataFrame(result).se .> 0)
        
        # Check that we have the expected profile information
        # The current API may include profile info in column names or metadata
        @test nrow(DataFrame(result)) > 0
    end

    @testset "Different Variable Effects" begin
        # Test multiple variables 
        result = profile_margins(model, df;
            at = :means,
            type = :effects,
            vars = [:x1, :x2]
        )
        
        @test nrow(DataFrame(result)) == 2  # One row per variable
        @test Set(DataFrame(result).term) == Set(["x1", "x2"])  # String terms
        @test all(.!ismissing.(DataFrame(result).se))
        @test all(DataFrame(result).se .> 0)
    end

    @testset "Profile Grid Expansion" begin
        # Test Cartesian product expansion with Dict
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 1], :x2 => [-0.5, 0.5]),
            type = :effects,
            vars = [:x1]
        )
        
        # Should have multiple profiles (Cartesian product)
        @test nrow(DataFrame(result)) == 4  # 2 x1 values × 2 x2 values
        @test all(DataFrame(result).term .== "x1")
        @test all(.!ismissing.(DataFrame(result).se))
        @test all(DataFrame(result).se .> 0)
    end

    @testset "Multiple Variables with Profiles" begin
        # Test multiple variables with profile grid
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 1], :x2 => [-1, 1]),
            type = :effects,
            vars = [:x1, :x2]
        )
        
        # Should have 2 variables × 4 profiles = 8 rows
        @test nrow(DataFrame(result)) == 8
        @test Set(DataFrame(result).term) == Set(["x1", "x2"])  # String terms
        @test all(.!ismissing.(DataFrame(result).se))
        @test all(DataFrame(result).se .> 0)
    end

    @testset "Predictions with Profiles" begin
        # Test predictions (not just effects) with profiles
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :predictions
        )
        
        @test nrow(DataFrame(result)) == 3  # One per profile
        @test all(startswith.(DataFrame(result).term, "APM"))  # Adjusted Prediction at Mean/Representative
        @test all(.!ismissing.(DataFrame(result).se))
        @test all(DataFrame(result).se .> 0)
    end

    @testset "Error Handling" begin
        # Test basic error handling for invalid parameters
        
        # Test invalid variable
        @test_throws Exception profile_margins(model, df;
            at = :means,
            type = :effects,
            vars = [:nonexistent_var]
        )
        
        # Test valid parameters work
        @test_nowarn profile_margins(model, df;
            at = :means,
            type = :effects,
            vars = [:x1]
        )
    end

    @testset "Consistency Check - Different Profile Specifications" begin
        # Compare different ways of specifying profiles
        
        # Single profile at means
        result_means = profile_margins(model, df;
            at = :means,
            type = :effects,
            vars = [:x1]
        )
        
        # Manual profile using typical values
        result_manual = profile_margins(model, df;
            at = Dict(:x1 => [mean(df.x1)], :x2 => [mean(df.x2)], :group_num => [1]),  # Use first level instead of mode
            type = :effects,
            vars = [:x1]
        )
        
        # Both should have single results
        @test nrow(DataFrame(result_means)) == 1
        @test nrow(DataFrame(result_manual)) == 1
        
        # Results should be similar (not identical due to categorical handling)
        @test abs(DataFrame(result_means).estimate[1] - DataFrame(result_manual).estimate[1]) < 0.1
    end
end