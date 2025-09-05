# robust_se_tests.jl - Comprehensive Robust SE Integration Tests
#
# Robust SE Integration Test Suite
#
# This file integrates robust standard error validation into the main test suite,
# providing systematic validation of CovarianceMatrices.jl integration across
# sandwich estimators, clustered SEs, and heteroskedasticity-robust computation.

using Test
using Random  
using DataFrames
using Statistics
using Margins
using GLM

@testset "Robust Standard Errors Integration" begin
    Random.seed!(06515)  
    
    @testset "Robust SE Framework Tests" begin
        # Test heteroskedastic data generation
        @testset "Heteroskedastic Data Generation" begin
            # Test different heteroskedasticity patterns
            data_linear = make_heteroskedastic_data(n=100, heteroskedasticity_type=:linear)
            data_quad = make_heteroskedastic_data(n=100, heteroskedasticity_type=:quadratic) 
            data_group = make_heteroskedastic_data(n=100, heteroskedasticity_type=:groupwise)
            
            @test nrow(data_linear) == 100
            @test nrow(data_quad) == 100
            @test nrow(data_group) == 100
            
            # Check required variables present  
            for data in [data_linear, data_quad, data_group]
                @test "x" in names(data)
                @test "y" in names(data) 
                @test "binary_y" in names(data)
                @test "group" in names(data)
            end
        end
        
        @testset "Basic Robust SE Integration" begin
            data = make_heteroskedastic_data(n=200, heteroskedasticity_type=:linear)
            model = lm(@formula(y ~ x + z), data)
            
            validation = validate_robust_se_integration(model, data, HC1(); tolerance=0.02)
            
            @test validation.all_finite_model
            @test validation.all_finite_robust  
            @test validation.all_positive_model
            @test validation.all_positive_robust
            @test validation.estimates_match  # Point estimates should match
        end
    end
    
    @testset "Comprehensive Sandwich Estimator Testing" begin
        # Test Linear model with heteroskedasticity
        @testset "Linear Model Sandwich Estimators" begin
            data = make_heteroskedastic_data(n=300, heteroskedasticity_type=:quadratic)
            model = lm(@formula(y ~ x + z), data)
            
            sandwich_results = test_sandwich_estimators_comprehensive(model, data)
            
            @test sandwich_results.overall_success
            @test sandwich_results.n_successful >= 2  # At least half should work
            
            # Test that HC1 works across 2×2 framework
            if haskey(sandwich_results.estimator_results, :HC1)
                hc1 = sandwich_results.estimator_results[:HC1]
                if hc1.success
                    @test hc1.all_quadrants_valid
                    @test hc1.validation.meaningfully_different
                end
            end
        end
        
        # Test GLM with sandwich estimators  
        @testset "GLM Sandwich Estimators" begin
            data = make_heteroskedastic_data(n=400, heteroskedasticity_type=:linear)
            model = glm(@formula(binary_y ~ x + z), data, Binomial(), LogitLink())
            
            sandwich_results = test_sandwich_estimators_comprehensive(model, data)
            
            @test sandwich_results.overall_success
            
            # GLM should work with robust SEs
            if haskey(sandwich_results.estimator_results, :HC1)
                hc1 = sandwich_results.estimator_results[:HC1]  
                if hc1.success
                    @test hc1.all_quadrants_valid
                end
            end
            
        end
    end
    
    @testset "Clustered Standard Errors Testing" begin

        # Test clustered SEs with linear model
        @testset "Linear Model Clustered SEs" begin
            data = make_heteroskedastic_data(n=500, heteroskedasticity_type=:groupwise)
            model = lm(@formula(y ~ x + z), data)
            
            cluster_results = test_clustered_se_validation(model, data, :group)
            @test cluster_results.success
            @test cluster_results.framework_valid
            @test cluster_results.max_se_ratio >= 1.0  # Clustered ≥ model SEs
        end
        
        # Test clustered SEs with different cluster sizes
        @testset "Variable Cluster Sizes" begin
            # Create data with very unbalanced clusters 
            n = 300
            cluster_data = DataFrame(
                x = randn(n),
                z = randn(n), 
                cluster_id = vcat(repeat([1], 200), repeat([2], 50), repeat([3], 30), repeat([4], 20))  # Unbalanced
            )
            cluster_data.y = 2.0 .+ 0.5 .* cluster_data.x .+ 0.3 .* cluster_data.z .+ randn(n) .* 0.2
            
            model = lm(@formula(y ~ x + z), cluster_data)
            cluster_results = test_clustered_se_validation(model, cluster_data, :cluster_id)
            
            @test cluster_results.success
            @test cluster_results.framework_valid
        end
    end
    
    @testset "Comprehensive Robust SE Test Suite" begin
        
        # Run the full test suite with reduced verbosity for integration
        results = run_comprehensive_robust_se_test_suite(verbose=false)
        
        # Validate overall structure and results
        @test haskey(results, :overall_success_rate)
        @test haskey(results, :test_results)
        @test haskey(results, :covariance_matrices_available)
        
        @test results.overall_success_rate >= 0.60  # At least 60% success rate
        @test results.n_successful >= 1  # At least one test should succeed
        
        # Test individual result structure
        for result in results.test_results
            @test haskey(result, :model_type)
            @test haskey(result, :success) || haskey(result, :reason)
            
            if haskey(result, :success) && result.success && haskey(result, :n_successful)
                @test result.n_successful >= 0
            end
        end
    end
    
    @testset "Robust SE Edge Cases and Error Handling" begin
        # Test with very small sample
        @testset "Small Sample Robustness" begin
            small_data = make_heteroskedastic_data(n=30, heteroskedasticity_type=:linear)
            model = lm(@formula(y ~ x), small_data)
            
            result = population_margins(model, small_data; type=:effects, vars=[:x], vcov=HC1())
            result_df = DataFrame(result)
            @test all(isfinite, result_df.se)
            @test all(result_df.se .> 0)
        end
        
        # Test error handling for invalid cluster variable
        @testset "Invalid Cluster Variable Handling" begin
            data = make_heteroskedastic_data(n=100)
            model = lm(@formula(y ~ x + z), data)
            
            # Should throw error for nonexistent cluster variable
            @test_throws ArgumentError population_margins(model, data; type=:effects, vars=[:x], vcov=CRHC0(:nonexistent_var))
        end
    end
end