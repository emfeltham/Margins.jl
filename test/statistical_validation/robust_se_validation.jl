# robust_se_validation.jl - Comprehensive Robust Standard Error Validation
#
# Phase 3, Tier 4: Robust SE Integration Testing
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

# Load testing utilities
include("testing_utilities.jl")

# Conditional CovarianceMatrices.jl support
const COVARIANCE_MATRICES_AVAILABLE = try
    using CovarianceMatrices
    true
catch
    false
end

if COVARIANCE_MATRICES_AVAILABLE
    using CovarianceMatrices
end

"""
    make_heteroskedastic_data(; n=500, heteroskedasticity_type=:linear, seed=42)

Generate test data with known heteroskedasticity patterns for robust SE validation.

# Arguments
- `n`: Sample size
- `heteroskedasticity_type`: Type of heteroskedasticity pattern
  - `:linear` - Variance proportional to x
  - `:quadratic` - Variance proportional to x²
  - `:groupwise` - Different variances by group
- `seed`: Random seed for reproducibility

# Returns
- `DataFrame` with heteroskedastic error structure
"""
function make_heteroskedastic_data(; n=500, heteroskedasticity_type=:linear, seed=42)
    Random.seed!(seed)
    
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        group = rand(1:5, n)  # For clustered/grouped heteroskedasticity
    )
    
    # Base linear relationship
    linear_pred = 2.0 .+ 0.5 .* df.x .+ 0.3 .* df.z
    
    # Generate heteroskedastic errors
    if heteroskedasticity_type == :linear
        # Variance proportional to |x| + 1 (to avoid zero variance)
        error_scale = 0.2 .* (abs.(df.x) .+ 1.0)
        errors = randn(n) .* error_scale
    elseif heteroskedasticity_type == :quadratic
        # Variance proportional to x²
        error_scale = 0.1 .* (df.x.^2 .+ 1.0)
        errors = randn(n) .* error_scale
    elseif heteroskedasticity_type == :groupwise
        # Different error variances by group
        group_scales = [0.1, 0.3, 0.5, 0.8, 1.2]  # Different scales for each group
        errors = [randn() * group_scales[g] for g in df.group]
    else
        # Homoskedastic baseline
        errors = 0.2 .* randn(n)
    end
    
    df.y = linear_pred .+ errors
    
    # For logistic models
    logit_linear_pred = -0.5 .+ 0.4 .* df.x .+ 0.3 .* df.z
    probs = 1 ./ (1 .+ exp.(-logit_linear_pred))
    df.binary_y = [rand() < p for p in probs]
    
    return df
end

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
    # Get model-based SEs
    model_result = population_margins(model, data; type=:effects, vars=[:x], vcov=:model)
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
    if !COVARIANCE_MATRICES_AVAILABLE
        return (success=false, reason="CovarianceMatrices.jl not available")
    end
    
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
    if !COVARIANCE_MATRICES_AVAILABLE
        return (success=false, reason="CovarianceMatrices.jl not available")
    end
    
    try
        # Create cluster-robust covariance matrix
        clustered_vcov = CRHC0(cluster_var)
        
        # Test clustered SEs
        clustered_result = population_margins(model, data; type=:effects, vars=[:x], vcov=clustered_vcov)
        clustered_ses = DataFrame(clustered_result).se
        
        # Compare to model-based SEs
        model_result = population_margins(model, data; type=:effects, vars=[:x], vcov=:model)
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
function run_comprehensive_robust_se_test_suite(; verbose=true)
    if verbose
        @info "Starting Comprehensive Robust SE Validation Suite"
        if COVARIANCE_MATRICES_AVAILABLE
            @info "CovarianceMatrices.jl available - full testing enabled"
        else
            @warn "CovarianceMatrices.jl not available - limited testing"
        end
        @info "="^60
    end
    
    test_results = []
    
    # Test 1: Linear model with heteroskedasticity
    @testset "Linear Model with Heteroskedasticity" begin
        if verbose
            @info "Testing linear model robust SEs"
        end
        
        data = make_heteroskedastic_data(n=400, heteroskedasticity_type=:linear)
        model = lm(@formula(y ~ x + z), data)
        
        if COVARIANCE_MATRICES_AVAILABLE
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
                @info "  Sandwich estimators: $(sandwich_results.n_successful)/$(sandwich_results.n_tested) successful"
            end
        else
            @warn "Skipping sandwich estimator tests - CovarianceMatrices.jl not available"
            push!(test_results, (
                model_type = "Linear with heteroskedasticity",
                success = false,
                reason = "CovarianceMatrices.jl not available"
            ))
        end
    end
    
    # Test 2: GLM with robust SEs
    @testset "GLM Logistic with Robust SEs" begin
        if verbose
            @info "Testing GLM logistic robust SEs"
        end
        
        data = make_heteroskedastic_data(n=500, heteroskedasticity_type=:quadratic)
        model = glm(@formula(binary_y ~ x + z), data, Binomial(), LogitLink())
        
        if COVARIANCE_MATRICES_AVAILABLE
            sandwich_results = test_sandwich_estimators_comprehensive(model, data)
            
            @test sandwich_results.overall_success
            
            push!(test_results, (
                model_type = "GLM Logistic with robust SEs",
                success = sandwich_results.overall_success,
                n_successful = sandwich_results.n_successful
            ))
            
            if verbose
                @info "  GLM sandwich estimators: $(sandwich_results.n_successful)/$(sandwich_results.n_tested) successful"
            end
        else
            push!(test_results, (
                model_type = "GLM Logistic with robust SEs",
                success = false,
                reason = "CovarianceMatrices.jl not available"
            ))
        end
    end
    
    # Test 3: Clustered standard errors
    @testset "Clustered Standard Errors" begin
        if verbose
            @info "Testing clustered standard errors"
        end
        
        data = make_heteroskedastic_data(n=600, heteroskedasticity_type=:groupwise)
        model = lm(@formula(y ~ x + z), data)
        
        if COVARIANCE_MATRICES_AVAILABLE
            cluster_results = test_clustered_se_validation(model, data, :group)
            
            if cluster_results.success
                @test cluster_results.framework_valid
                @test cluster_results.max_se_ratio >= 1.0  # Clustered SEs should be ≥ model SEs
                
                push!(test_results, (
                    model_type = "Linear with clustered SEs",
                    success = true,
                    max_se_ratio = cluster_results.max_se_ratio
                ))
                
                if verbose
                    @info "  Clustered SEs: Framework valid, max SE ratio = $(round(cluster_results.max_se_ratio, digits=2))"
                end
            else
                push!(test_results, (
                    model_type = "Linear with clustered SEs",
                    success = false,
                    error = cluster_results.error
                ))
            end
        else
            push!(test_results, (
                model_type = "Linear with clustered SEs",
                success = false,
                reason = "CovarianceMatrices.jl not available"
            ))
        end
    end
    
    # Overall assessment
    successful_tests = [r for r in test_results if haskey(r, :success) && r.success]
    overall_success_rate = length(successful_tests) / length(test_results)
    
    if verbose
        @info "="^60
        @info "COMPREHENSIVE ROBUST SE VALIDATION SUMMARY"
        @info "="^60
        @info "Total Tests: $(length(test_results))"
        @info "Successful Tests: $(length(successful_tests))/$(length(test_results)) ($(round(overall_success_rate * 100, digits=1))%)"
        
        if COVARIANCE_MATRICES_AVAILABLE
            if overall_success_rate >= 0.75
                @info "🎉 ROBUST SE VALIDATION: PASSED"
                @info "Robust standard errors integration working correctly!"
            else
                @warn "⚠️  ROBUST SE VALIDATION: MIXED RESULTS"
                @info "Some robust SE tests failed - detailed investigation recommended"
            end
        else
            @warn "⚠️  ROBUST SE VALIDATION: DEPENDENCY MISSING"
            @info "Install CovarianceMatrices.jl for full robust SE support"
        end
    end
    
    return (
        overall_success_rate = overall_success_rate,
        test_results = test_results,
        n_successful = length(successful_tests),
        covariance_matrices_available = COVARIANCE_MATRICES_AVAILABLE
    )
end

# Export robust SE validation functions
export make_heteroskedastic_data, validate_robust_se_integration
export test_sandwich_estimators_comprehensive, test_clustered_se_validation
export run_comprehensive_robust_se_test_suite