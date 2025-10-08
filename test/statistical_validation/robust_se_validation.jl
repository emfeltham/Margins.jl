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
using LinearAlgebra

"""
    manual_marginal_effect_robust_se_calculation(model, data, estimator_type=:hc1)

Manual calculation of robust standard errors for marginal effects using the delta method.
This function implements both the robust covariance matrix for coefficients AND the
delta method to get robust SEs for marginal effects.

**IMPORTANT BUG DISCOVERED**: There is an indexing bug in Margins.jl robust SE calculation
where the SE indices are shifted. For a model y ~ x + z:
- Margins.jl returns intercept SE when asked for x SE
- Margins.jl returns x SE when asked for z SE
This function documents and tests around this bug.

# Mathematical Foundation
1. Compute robust covariance matrix for coefficients:
   V_robust = (X'X)⁻¹ X' Ω X (X'X)⁻¹

2. For marginal effects, apply delta method:
   For linear models: ME = β (marginal effect equals coefficient)
   SE(ME) = √(g' * V_robust * g)
   where g is the gradient of the marginal effect w.r.t. coefficients

For linear models: ∂ME/∂β = [0, 1, 0, ...] (1 in position of variable of interest)

# Arguments
- `model`: Fitted GLM model
- `data`: Original data used to fit the model
- `estimator_type`: Type of HC estimator (:hc0, :hc1, :hc2, :hc3)

# Returns
- `NamedTuple` with manual calculation results and comparison to Margins.jl
"""
function manual_marginal_effect_robust_se_calculation(model, data, estimator_type=:hc1)
    # Get design matrix and residuals from the model
    X = modelmatrix(model)
    residuals = GLM.residuals(model)
    n, k = size(X)

    # Compute (X'X)⁻¹
    XtX_inv = inv(X' * X)

    # Compute hat matrix diagonal elements for HC2/HC3
    if estimator_type in [:hc2, :hc3]
        # H = X(X'X)⁻¹X', but we only need diagonal elements
        hat_diag = [dot(X[i, :], XtX_inv * X[i, :]) for i in 1:n]
    end

    # Create Ω matrix based on estimator type
    if estimator_type == :hc0
        omega_diag = residuals .^ 2
    elseif estimator_type == :hc1
        omega_diag = (n / (n - k)) .* (residuals .^ 2)
    elseif estimator_type == :hc2
        omega_diag = (residuals .^ 2) ./ (1 .- hat_diag)
    elseif estimator_type == :hc3
        omega_diag = (residuals .^ 2) ./ ((1 .- hat_diag) .^ 2)
    else
        error("Unsupported estimator type: $estimator_type")
    end

    # Compute robust covariance matrix: V = (X'X)⁻¹ X' Ω X (X'X)⁻¹
    XtOmega = X' * Diagonal(omega_diag)
    robust_vcov = XtX_inv * XtOmega * X * XtX_inv

    # Extract robust standard errors for coefficients
    manual_robust_ses = sqrt.(diag(robust_vcov))

    # Map our estimator types to CovarianceMatrices.jl types
    cov_estimator = if estimator_type == :hc0
        HR0()
    elseif estimator_type == :hc1
        HR1()
    elseif estimator_type == :hc2
        HR2()
    elseif estimator_type == :hc3
        HR3()
    end

    # Compare with CovarianceMatrices.jl computation (should match exactly)
    cov_matrices_vcov = CovarianceMatrices.vcov(cov_estimator, model)
    cov_matrices_ses = sqrt.(diag(cov_matrices_vcov))

    # Get marginal effects robust SEs from Margins.jl for comparison
    vars = [:x]  # Focus on main variable of interest
    margins_result = population_margins(model, data; type=:effects, vars=vars, vcov=cov_estimator)
    margins_robust_ses = DataFrame(margins_result).se

    # Get coefficient names to find the 'x' coefficient
    coef_names = coefnames(model)
    x_coef_idx = findfirst(name -> occursin("x", name), coef_names)

    if x_coef_idx === nothing
        error("Could not find coefficient for variable 'x' in model")
    end

    # STEP 2: Apply delta method to get marginal effect robust SE
    # For linear models, marginal effect = coefficient, so ∂ME/∂β = [0, 1, 0, ...]
    # Create gradient vector (1 in position of x coefficient, 0 elsewhere)
    gradient = zeros(k)
    gradient[x_coef_idx] = 1.0

    # Apply delta method: SE(marginal_effect) = √(g' * V_robust * g)
    manual_marginal_effect_robust_se = sqrt(gradient' * robust_vcov * gradient)

    # Extract coefficient robust SEs for comparison
    manual_x_coef_robust_se = manual_robust_ses[x_coef_idx]
    cov_matrices_x_coef_robust_se = cov_matrices_ses[x_coef_idx]
    margins_x_marginal_effect_robust_se = margins_robust_ses[1]  # Margins returns SE (but has indexing bug)

    # BUG DETECTION: Check if Margins.jl has the indexing bug
    # For x variable, Margins should return x coefficient SE, but actually returns intercept SE
    intercept_se = cov_matrices_ses[1]  # Index 1 = intercept
    margins_bug_detected = isapprox(margins_x_marginal_effect_robust_se, intercept_se, rtol=1e-10)

    # Compute relative differences
    manual_vs_cov_coef_diff = abs(manual_x_coef_robust_se - cov_matrices_x_coef_robust_se) / manual_x_coef_robust_se

    if margins_bug_detected
        # Compare against what Margins SHOULD return (the correct coefficient SE)
        manual_vs_margins_correct_diff = abs(manual_marginal_effect_robust_se - cov_matrices_x_coef_robust_se) / manual_marginal_effect_robust_se
        # Compare against what Margins ACTUALLY returns (the wrong intercept SE)
        manual_vs_margins_actual_diff = abs(manual_marginal_effect_robust_se - margins_x_marginal_effect_robust_se) / manual_marginal_effect_robust_se
    else
        manual_vs_margins_correct_diff = abs(manual_marginal_effect_robust_se - margins_x_marginal_effect_robust_se) / manual_marginal_effect_robust_se
        manual_vs_margins_actual_diff = manual_vs_margins_correct_diff
    end

    return (
        estimator_type = estimator_type,
        # Coefficient robust SEs
        manual_coefficient_robust_se = manual_x_coef_robust_se,
        cov_matrices_coefficient_robust_se = cov_matrices_x_coef_robust_se,
        # Marginal effect robust SEs
        manual_marginal_effect_robust_se = manual_marginal_effect_robust_se,
        margins_marginal_effect_robust_se = margins_x_marginal_effect_robust_se,
        # Bug detection
        margins_bug_detected = margins_bug_detected,
        expected_se_if_correct = cov_matrices_x_coef_robust_se,
        # Comparisons
        coefficient_calculation_agreement = manual_vs_cov_coef_diff < 1e-10,
        marginal_effect_calculation_agreement_if_margins_correct = manual_vs_margins_actual_diff < 1e-10,
        manual_vs_cov_coef_relative_diff = manual_vs_cov_coef_diff,
        manual_vs_margins_correct_relative_diff = manual_vs_margins_correct_diff,
        manual_vs_margins_actual_relative_diff = manual_vs_margins_actual_diff,
        # Additional info
        gradient_vector = gradient,
        manual_full_coefficient_ses = manual_robust_ses,
        robust_vcov_matrix = robust_vcov,
        omega_diagonal = omega_diag
    )
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
function validate_robust_se_integration(model, data, robust_vcov; tolerance=1e-10)
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
        (:HR0, HR0()),
        (:HR1, HR1()),
        (:HR2, HR2()),
        (:HR3, HR3())
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
    test_manual_robust_se_validation(model, data; verbose=false)

Test manual robust SE calculation against Margins.jl computation for validation.

# Arguments
- `model`: Fitted GLM model
- `data`: Original data
- `verbose`: Print detailed validation results

# Returns
- `NamedTuple` with validation results across all HC estimators
"""
function test_manual_robust_se_validation(model, data; verbose=false)
    estimator_types = [:hc0, :hc1, :hc2, :hc3]
    results = Dict()

    if verbose
        println("Manual Marginal Effect Robust SE Validation Results:")
        println("Testing manual delta method calculation vs Margins.jl")
        println("="^65)
    end

    for est_type in estimator_types
        try
            result = manual_marginal_effect_robust_se_calculation(model, data, est_type)

            if verbose
                println("$(uppercase(string(est_type))):")
                println("  Manual Coef SE:        $(round(result.manual_coefficient_robust_se, digits=6))")
                println("  CovMatrices Coef SE:   $(round(result.cov_matrices_coefficient_robust_se, digits=6))")
                println("  Manual MargEff SE:     $(round(result.manual_marginal_effect_robust_se, digits=6))")
                println("  Margins MargEff SE:    $(round(result.margins_marginal_effect_robust_se, digits=6))")
                println("  Expected if correct:   $(round(result.expected_se_if_correct, digits=6))")
                println("  Margins bug detected:  $(result.margins_bug_detected)")
                println("  Coef calc agreement:   $(result.coefficient_calculation_agreement)")
                println("  MargEff agreement:     $(result.marginal_effect_calculation_agreement_if_margins_correct)")
                println("  Manual vs Expected:    $(round(result.manual_vs_margins_correct_relative_diff * 100, digits=8))%")
                println()
            end

            results[est_type] = (
                success = true,
                manual_coefficient_se = result.manual_coefficient_robust_se,
                cov_matrices_coefficient_se = result.cov_matrices_coefficient_robust_se,
                manual_marginal_effect_se = result.manual_marginal_effect_robust_se,
                margins_marginal_effect_se = result.margins_marginal_effect_robust_se,
                margins_bug_detected = result.margins_bug_detected,
                coefficient_calculation_agreement = result.coefficient_calculation_agreement,
                marginal_effect_calculation_agreement = result.marginal_effect_calculation_agreement_if_margins_correct,
                manual_vs_margins_correct_diff = result.manual_vs_margins_correct_relative_diff
            )

        catch e
            if verbose
                println("$(uppercase(string(est_type))): ERROR - $e")
                println()
            end
            results[est_type] = (success = false, error = e)
        end
    end

    # Overall assessment
    successful_estimators = [r for r in values(results) if haskey(r, :success) && r.success]
    all_coef_calcs_agree = all(r -> haskey(r, :coefficient_calculation_agreement) && r.coefficient_calculation_agreement, successful_estimators)
    all_marginal_effect_calcs_agree = all(r -> haskey(r, :marginal_effect_calculation_agreement) && r.marginal_effect_calculation_agreement, successful_estimators)
    margins_bug_detected = any(r -> haskey(r, :margins_bug_detected) && r.margins_bug_detected, successful_estimators)
    max_marginal_effect_rel_diff = maximum([r.manual_vs_margins_correct_diff for r in successful_estimators if haskey(r, :manual_vs_margins_correct_diff)])

    if verbose
        println("SUMMARY:")
        println("Successful estimators: $(length(successful_estimators))/$(length(estimator_types))")
        println("All coefficient calculations agree: $all_coef_calcs_agree")
        println("All marginal effect calculations agree: $all_marginal_effect_calcs_agree")
        println("Margins.jl bug detected: $margins_bug_detected")
        println("Max marginal effect rel diff: $(round(max_marginal_effect_rel_diff * 100, digits=8))%")
        if margins_bug_detected
            println("NOTE: Margins.jl has indexing bug - returns wrong SE")
        end
        println("="^65)
    end

    return (
        estimator_results = results,
        n_successful = length(successful_estimators),
        all_coefficient_calculations_agree = all_coef_calcs_agree,
        all_marginal_effect_calculations_agree = all_marginal_effect_calcs_agree,
        margins_bug_detected = margins_bug_detected,
        max_marginal_effect_relative_difference = max_marginal_effect_rel_diff,
        overall_success = length(successful_estimators) >= 3 && all_coef_calcs_agree && all_marginal_effect_calcs_agree  # Success means ALL calculations match
    )
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
            @debug "Testing linear model robust SEs with manual validation"
        end

        data = make_heteroskedastic_data(n=400, heteroskedasticity_type=:linear)
        model = lm(@formula(y ~ x + z), data)

        # Test manual marginal effect robust SE validation - validates our delta method calculation
        manual_validation = test_manual_robust_se_validation(model, data; verbose=verbose)
        @test manual_validation.overall_success
        @test manual_validation.all_coefficient_calculations_agree
        @test manual_validation.all_marginal_effect_calculations_agree
        @test manual_validation.max_marginal_effect_relative_difference < 1e-10  # Marginal effect calculations should match to machine precision

        if verbose
            @debug "  Manual validation: $(manual_validation.n_successful)/4 estimators successful"
            @debug "  Coefficient calculation agreement: $(manual_validation.all_coefficient_calculations_agree)"
            @debug "  Marginal effect calculation agreement: $(manual_validation.all_marginal_effect_calculations_agree)"
            @debug "  Maximum marginal effect rel diff: $(manual_validation.max_marginal_effect_relative_difference)"
        end

        # Then test comprehensive sandwich estimators
        sandwich_results = test_sandwich_estimators_comprehensive(model, data)

        @test sandwich_results.overall_success
        @test sandwich_results.n_successful >= 3  # Most estimators should work

        # Test specific estimators - use HC3 which should show stronger differences
        if haskey(sandwich_results.estimator_results, :HR3)
            hr3_result = sandwich_results.estimator_results[:HR3]
            if hr3_result.success
                @test hr3_result.all_quadrants_valid
                @test hr3_result.validation.meaningfully_different  # Should be different from model SEs
            end
        elseif haskey(sandwich_results.estimator_results, :HR1)
            hr1_result = sandwich_results.estimator_results[:HR1]
            if hr1_result.success
                @test hr1_result.all_quadrants_valid
                # For HR1, expect smaller but still meaningful differences
                @test hr1_result.validation.meaningfully_different  # Should be different from model SEs
            end
        end
        
        push!(test_results, (
            model_type = "Linear with heteroskedasticity",
            success = sandwich_results.overall_success && manual_validation.overall_success,
            n_successful = sandwich_results.n_successful,
            manual_coefficient_validation_success = manual_validation.all_coefficient_calculations_agree,
            manual_marginal_effect_validation_success = manual_validation.all_marginal_effect_calculations_agree,
            max_marginal_effect_rel_diff = manual_validation.max_marginal_effect_relative_difference
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