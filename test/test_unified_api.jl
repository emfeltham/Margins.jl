# julia --project="." test/test_unified_api.jl > test/test_unified_api.txt 2>&1

using Test, Margins, GLM, DataFrames, CategoricalArrays

@testset "Unified API Correctness Tests" begin
    # Generate test data with all variable types
    n = 100
    df = DataFrame(
        y = randn(n),
        x_continuous = randn(n),
        x_boolean = rand([true, false], n),
        x_categorical = categorical(rand(["A", "B", "C"], n))
    )
    
    model = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), df)
    
    @testset "Individual vs Unified Results Match" begin
        # Test that unified dispatcher gives same results as direct calls
        result_mixed = population_margins(model, df; type=:effects, vars=[:x_continuous, :x_boolean, :x_categorical])
        result_cont = population_margins(model, df; type=:effects, vars=[:x_continuous])
        result_bool = population_margins(model, df; type=:effects, vars=[:x_boolean])
        result_cat = population_margins(model, df; type=:effects, vars=[:x_categorical])
        
        # Results should match
        @test isapprox(result_mixed.estimates[1], result_cont.estimates[1], rtol=1e-6)
        @test isapprox(result_mixed.estimates[2], result_bool.estimates[1], rtol=1e-6)
        @test isapprox(result_mixed.estimates[3], result_cat.estimates[1], rtol=1e-6)
    end
    
    @testset "Backend Consistency for Continuous Variables" begin
        # Test that FD and AD give similar results for continuous
        result_fd = population_margins(model, df; type=:effects, vars=[:x_continuous], backend=:fd)
        result_ad = population_margins(model, df; type=:effects, vars=[:x_continuous], backend=:ad)
        
        @test isapprox(result_fd.estimates[1], result_ad.estimates[1], rtol=1e-4)
        @test isapprox(result_fd.standard_errors[1], result_ad.standard_errors[1], rtol=1e-3)
    end
    
    @testset "Variable Type Detection" begin
        # Test that the unified system correctly identifies variable types
        # This tests the _detect_variable_type function indirectly
        
        # Continuous variables should work with both backends
        result_fd_cont = population_margins(model, df; type=:effects, vars=[:x_continuous], backend=:fd)
        result_ad_cont = population_margins(model, df; type=:effects, vars=[:x_continuous], backend=:ad)
        
        # Both should succeed and give reasonable results
        @test length(result_fd_cont.estimates) == 1
        @test length(result_ad_cont.estimates) == 1
        @test !isnan(result_fd_cont.estimates[1])
        @test !isnan(result_ad_cont.estimates[1])
        
        # Boolean variables should use DataScenario optimization
        result_bool = population_margins(model, df; type=:effects, vars=[:x_boolean])
        @test length(result_bool.estimates) == 1
        @test !isnan(result_bool.estimates[1])
        
        # Categorical variables should use DataScenario optimization  
        result_cat = population_margins(model, df; type=:effects, vars=[:x_categorical])
        @test length(result_cat.estimates) == 1
        @test !isnan(result_cat.estimates[1])
    end
    
    @testset "Mixed Variable Types Processing" begin
        # Test that mixed variable types work together seamlessly
        result_all = population_margins(model, df; type=:effects, vars=[:x_continuous, :x_boolean, :x_categorical])
        
        # Should have results for all three variables
        @test length(result_all.estimates) == 3
        @test length(result_all.standard_errors) == 3
        @test length(result_all.variables) == 3
        
        # All results should be finite
        @test all(isfinite.(result_all.estimates))
        @test all(isfinite.(result_all.standard_errors))
        @test all(result_all.standard_errors .> 0)  # SEs should be positive
        
        # Terms should match variable names
        @test "x_continuous" in result_all.variables
        @test "x_boolean" in result_all.variables  
        @test "x_categorical" in result_all.variables
    end
    
    @testset "Consistency Across Different Data Sizes" begin
        # Test that results are consistent across different sample sizes
        small_df = df[1:50, :]
        large_df = vcat(df, df)  # Double the data
        
        model_small = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), small_df)
        model_large = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), large_df)
        
        # Results should be reasonable (not testing exact equality due to sampling)
        result_small = population_margins(model_small, small_df; type=:effects, vars=[:x_continuous])
        result_large = population_margins(model_large, large_df; type=:effects, vars=[:x_continuous])
        
        @test length(result_small.estimates) == 1
        @test length(result_large.estimates) == 1
        @test isfinite(result_small.estimates[1])
        @test isfinite(result_large.estimates[1])
    end
end
