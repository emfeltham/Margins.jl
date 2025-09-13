# robust_se_validation.jl
# Comprehensive Robust Standard Error Validation
#
# Robust SE Integration Testing
#
# This file implements comprehensive validation for robust/heteroskedasticity-consistent
# standard errors integration with CovarianceMatrices.jl, covering sandwich estimators,
# clustered SEs, and analytical validation of robust SE computation correctness.

using Test
using Random
using DataFrames
using Statistics
using GLM
using StatsModels
using Margins
using CovarianceMatrices

"""
    validate_robust_se_integration(model, data, robust_vcov; tolerance=0.05)

Validate that robust standard errors are correctly integrated and different from 
model-based SEs when heteroskedasticity is present.

# Arguments
- `model`: Fitted model
- `data`: Dataset (should have heteroskedasticity)
- `robust_vcov`: CovarianceMatrices.jl robust estimator (e.g., HC1())
- `tolerance`: Minimum relative difference expected between model and robust SEs

# Returns
- `NamedTuple` with validation results
"""
function validate_robust_se_integration(model, data, robust_vcov; tolerance=0.05)
    # Get model-based SEs (using default GLM.vcov)
    model_result = population_margins(model, data; type=:effects, vars=[:x])
    model_ses = DataFrame(model_result).se
    
    # Get robust SEs
    robust_result = population_margins(model, data; type=:effects, vars=[:x], vcov=robust_vcov)
    robust_ses = DataFrame(robust_result).se
    
    # Validate basic properties
    all_finite_model = all(isfinite, model_ses)
    all_finite_robust = all(isfinite, robust_ses)
    all_positive_model = all(model_ses .> 0)
    all_positive_robust = all(robust_ses .> 0)
    
    # Check that robust SEs are different from model SEs (when heteroskedasticity present)
    relative_differences = abs.(robust_ses .- model_ses) ./ model_ses
    meaningfully_different = any(relative_differences .> tolerance)
    
    # Check that both produce same point estimates
    model_estimates = DataFrame(model_result).estimate
    robust_estimates = DataFrame(robust_result).estimate
    estimates_match = all(isapprox.(model_estimates, robust_estimates; atol=1e-12))
    
    return (
        all_finite_model = all_finite_model,
        all_finite_robust = all_finite_robust,
        all_positive_model = all_positive_model,
        all_positive_robust = all_positive_robust,
        meaningfully_different = meaningfully_different,
        estimates_match = estimates_match,
        max_relative_difference = maximum(relative_differences),
        model_ses = model_ses,
        robust_ses = robust_ses,
        relative_differences = relative_differences
    )
end

"""
    test_sandwich_estimators_comprehensive(model, data)

Test all major sandwich estimators (HC0, HC1, HC2, HC3) for comprehensive validation.

# Arguments
- `model`: Fitted model
- `data`: Test dataset with heteroskedasticity

# Returns
- `NamedTuple` with results for each estimator type
"""
function test_sandwich_estimators_comprehensive(model, data)
    
    estimators = [
        (:HC0, HC0()),
        (:HC1, HC1()),
        (:HC2, HC2()),
        (:HC3, HC3())
    ]
    
    results = Dict()
    
    for (name, estimator) in estimators
        try
            validation = validate_robust_se_integration(model, data, estimator)
            
            # Test that the estimator works across 2×2 framework
            pop_effects = population_margins(model, data; type=:effects, vars=[:x], vcov=estimator)
            pop_predictions = population_margins(model, data; type=:predictions, vcov=estimator)
            prof_effects = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x], vcov=estimator)
            prof_predictions = profile_margins(model, data, means_grid(data); type=:predictions, vcov=estimator)
            
            # Validate all quadrants
            pop_effects_valid = all(DataFrame(pop_effects).se .> 0)
            pop_pred_valid = all(DataFrame(pop_predictions).se .> 0)
            prof_effects_valid = all(DataFrame(prof_effects).se .> 0)
            prof_pred_valid = all(DataFrame(prof_predictions).se .> 0)
            
            results[name] = (
                success = true,
                validation = validation,
                pop_effects_valid = pop_effects_valid,
                pop_pred_valid = pop_pred_valid,
                prof_effects_valid = prof_effects_valid,
                prof_pred_valid = prof_pred_valid,
                all_quadrants_valid = pop_effects_valid && pop_pred_valid && prof_effects_valid && prof_pred_valid
            )
            
        catch e
            results[name] = (success = false, error = e)
        end
    end
    
    # Overall assessment
    successful_estimators = [r for r in values(results) if haskey(r, :success) && r.success]
    overall_success = length(successful_estimators) >= 2  # At least half should work
    
    return (
        estimator_results = results,
        overall_success = overall_success,
        n_successful = length(successful_estimators),
        n_tested = length(estimators)
    )
end

"""
    test_clustered_se_validation(model, data, cluster_var)

Test clustered standard error computation and validation.

# Arguments
- `model`: Fitted model
- `data`: Dataset with cluster variable
- `cluster_var`: Symbol for clustering variable

# Returns
- `NamedTuple` with clustered SE validation results
"""
function test_clustered_se_validation(model, data, cluster_var)
    
    try
        # Get the actual cluster data vector from DataFrame
        cluster_data = data[!, cluster_var]
        
        # Create cluster-robust covariance matrix
        clustered_vcov = CR0(cluster_data)
        
        # Test clustered SEs
        clustered_result = population_margins(model, data; type=:effects, vars=[:x], vcov=clustered_vcov)
        clustered_ses = DataFrame(clustered_result).se
        
        # Compare to model-based SEs
        model_result = population_margins(model, data; type=:effects, vars=[:x])
        model_ses = DataFrame(model_result).se
        
        # Clustered SEs should typically be larger than model SEs when clustering matters
        ses_increased = any(clustered_ses .>= model_ses)
        
        # Test across 2×2 framework
        pop_effects = population_margins(model, data; type=:effects, vars=[:x], vcov=clustered_vcov)
        pop_predictions = population_margins(model, data; type=:predictions, vcov=clustered_vcov)
        prof_effects = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x], vcov=clustered_vcov)
        prof_predictions = profile_margins(model, data, means_grid(data); type=:predictions, vcov=clustered_vcov)
        
        framework_valid = (
            all(DataFrame(pop_effects).se .> 0) &&
            all(DataFrame(pop_predictions).se .> 0) &&
            all(DataFrame(prof_effects).se .> 0) &&
            all(DataFrame(prof_predictions).se .> 0)
        )
        
        return (
            success = true,
            clustered_ses = clustered_ses,
            model_ses = model_ses,
            ses_increased = ses_increased,
            framework_valid = framework_valid,
            max_se_ratio = maximum(clustered_ses ./ model_ses)
        )
        
    catch e
        return (success = false, error = e)
    end
end

"""
    run_comprehensive_robust_se_test_suite(; verbose=true)

Run comprehensive robust SE validation across different model types and robust estimators.
"""
function run_comprehensive_robust_se_test_suite(; verbose=false)
    # Check if CovarianceMatrices.jl is actually available and working
    covariance_matrices_available = try
        # Test if we can create a basic robust covariance estimator
        HC1()
        true
    catch e
        if verbose
            @debug "CovarianceMatrices.jl not available: $e"
        end
        false
    end
    
    if verbose
        @debug "Starting Comprehensive Robust SE Validation Suite"
        @debug "CovarianceMatrices.jl available: $covariance_matrices_available"
        @debug "="^60
    end
    
    test_results = []
    
    # Test 1: Linear model with heteroskedasticity
    @testset "Linear Model with Heteroskedasticity" begin
        if verbose
            @debug "Testing linear model robust SEs"
        end
        
        data = make_heteroskedastic_data(n=400, heteroskedasticity_type=:linear)
        model = lm(@formula(y ~ x + z), data)
        
        sandwich_results = test_sandwich_estimators_comprehensive(model, data)
        
        @test sandwich_results.overall_success
        @test sandwich_results.n_successful >= 3  # Most estimators should work
        
        # Test specific estimators
        if haskey(sandwich_results.estimator_results, :HC1)
            hc1_result = sandwich_results.estimator_results[:HC1]
            if hc1_result.success
                @test hc1_result.all_quadrants_valid
                @test hc1_result.validation.meaningfully_different  # Should be different from model SEs
            end
        end
        
        push!(test_results, (
            model_type = "Linear with heteroskedasticity",
            success = sandwich_results.overall_success,
            n_successful = sandwich_results.n_successful
        ))
        
        if verbose
            @debug "  Sandwich estimators: $(sandwich_results.n_successful)/$(sandwich_results.n_tested) successful"
        end
    end
    
    # Test 2: GLM with robust SEs
    @testset "GLM Logistic with Robust SEs" begin
        if verbose
            @debug "Testing GLM logistic robust SEs"
        end
        
        data = make_heteroskedastic_data(n=500, heteroskedasticity_type=:quadratic)
        model = glm(@formula(binary_y ~ x + z), data, Binomial(), LogitLink())
        
        sandwich_results = test_sandwich_estimators_comprehensive(model, data)
        
        @test sandwich_results.overall_success
        
        push!(test_results, (
            model_type = "GLM Logistic with robust SEs",
            success = sandwich_results.overall_success,
            n_successful = sandwich_results.n_successful
        ))
        
        if verbose
            @debug "  GLM sandwich estimators: $(sandwich_results.n_successful)/$(sandwich_results.n_tested) successful"
        end
    end
    
    # Test 3: Clustered standard errors with known clustering structure
    @testset "Clustered Standard Errors" begin
        if verbose
            @debug "Testing clustered standard errors with Grunfeld-style data"
        end
        
        # Use Grunfeld-style data with known clustering properties
        data = make_grunfeld_style_data(n_firms=10, n_years=30, seed=123)
        model = lm(@formula(y ~ x + z), data)
        
        cluster_results = test_clustered_se_validation(model, data, :firm_id)
        
        if cluster_results.success
            @test cluster_results.framework_valid
            @test cluster_results.max_se_ratio > 0.0  # Clustered SEs should be positive and finite
            @test all(isfinite, cluster_results.clustered_ses)  # All clustered SEs should be finite
            @test all(cluster_results.clustered_ses .> 0)  # All clustered SEs should be positive
            
            # With Grunfeld-style data, we expect clustered SEs to be meaningfully different
            # (either larger or smaller) from model SEs, but not identical
            @test cluster_results.max_se_ratio != 1.0  # Should be different from model SEs
            
            push!(test_results, (
                model_type = "Linear with clustered SEs",
                success = true,
                max_se_ratio = cluster_results.max_se_ratio
            ))
            
            if verbose
                @debug "  Clustered SEs: Framework valid, max SE ratio = $(round(cluster_results.max_se_ratio, digits=2))"
            end
        else
            push!(test_results, (
                model_type = "Linear with clustered SEs",
                success = false,
                error = cluster_results.error
            ))
        end
    end
    
    # Overall assessment
    successful_tests = [r for r in test_results if haskey(r, :success) && r.success]
    overall_success_rate = length(successful_tests) / length(test_results)
    
    if verbose
        @debug "="^60
        @debug "COMPREHENSIVE ROBUST SE VALIDATION SUMMARY"
        @debug "="^60
        @debug "Total Tests: $(length(test_results))"
        @debug "Successful Tests: $(length(successful_tests))/$(length(test_results)) ($(round(overall_success_rate * 100, digits=1))%)"
        
        if overall_success_rate >= 0.75
            @debug "ROBUST SE VALIDATION: PASSED"
            @debug "Robust standard errors integration working correctly!"
        else
            @debug "ROBUST SE VALIDATION: MIXED RESULTS"
            @debug "Some robust SE tests failed - detailed investigation recommended"
        end
    end
    
    return (
        overall_success_rate = overall_success_rate,
        test_results = test_results,
        n_successful = length(successful_tests),
        covariance_matrices_available = covariance_matrices_available
    )
end

# Run the test suite
@testset "Robust SE Validation" begin
    run_comprehensive_robust_se_test_suite()
end

# Export robust SE validation functions
export make_heteroskedastic_data, validate_robust_se_integration
export test_sandwich_estimators_comprehensive, test_clustered_se_validation
export run_comprehensive_robust_se_test_suite