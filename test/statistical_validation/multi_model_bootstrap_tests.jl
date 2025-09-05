# multi_model_bootstrap_tests.jl - Multi-Model Bootstrap Testing Suite
#
# Multi-Model Bootstrap Testing Utilities
#
# This file implements systematic bootstrap validation across different model types:
# Linear models, GLM (logistic, Poisson), and various complexity levels.
# Tests all 2×2 framework quadrants for each model type.

using Test
using Random
using DataFrames
using Statistics
using GLM
using StatsModels
using Margins
using Printf

"""
    create_glm_model_func(family, link)
    
Create a model function for GLM with specified family and link.
Returns a function that takes (formula, data) and returns a fitted GLM.
"""
function create_glm_model_func(family, link)
    return (formula, data) -> glm(formula, data, family, link)
end

"""
    multi_model_bootstrap_test_suite()

Comprehensive bootstrap testing across multiple model types and complexities.
Tests systematic coverage following the pattern from analytical validation.
"""
function multi_model_bootstrap_test_suite()
    Random.seed!(06515)  # Reproducible results
    
    # Define test model configurations
    test_models = [
        # Linear models
        (
            name = "Linear: y ~ x + z", 
            model_func = lm,
            data_func = () -> make_simple_test_data(n=800, formula_type=:linear),
            formula = @formula(y ~ x + z),
            vars = [:x, :z],
            expected_agreement = 0.90  # High for linear models
        ),
        (
            name = "Linear: y ~ x (simple)",
            model_func = lm, 
            data_func = () -> make_simple_test_data(n=600, formula_type=:linear),
            formula = @formula(y ~ x),
            vars = [:x],
            expected_agreement = 0.95  # Very high for simple linear
        ),
        
        # GLM: Logistic regression
        (
            name = "Logistic: y ~ x + z",
            model_func = create_glm_model_func(Binomial(), LogitLink()),
            data_func = () -> make_glm_test_data(n=800, family=:binomial),
            formula = @formula(y ~ x + z),
            vars = [:x, :z], 
            expected_agreement = 0.80  # Moderate for GLM
        ),
        (
            name = "Logistic: y ~ x (simple)",
            model_func = create_glm_model_func(Binomial(), LogitLink()),
            data_func = () -> make_glm_test_data(n=600, family=:binomial), 
            formula = @formula(y ~ x),
            vars = [:x],
            expected_agreement = 0.85  # Good for simple GLM
        ),
        
        # GLM: Poisson regression
        (
            name = "Poisson: y ~ x + z",
            model_func = create_glm_model_func(Poisson(), LogLink()),
            data_func = () -> make_glm_test_data(n=600, family=:poisson),
            formula = @formula(y ~ x + z),
            vars = [:x, :z],
            expected_agreement = 0.75  # Lower for Poisson (count data variability)
        ),
        (
            name = "Poisson: y ~ x (simple)", 
            model_func = create_glm_model_func(Poisson(), LogLink()),
            data_func = () -> make_glm_test_data(n=500, family=:poisson),
            formula = @formula(y ~ x),
            vars = [:x],
            expected_agreement = 0.80  # Better for simple Poisson
        ),
        
        # More complex models
        (
            name = "Linear: Multiple variables",
            model_func = lm,
            data_func = () -> make_econometric_data(n=600),
            formula = @formula(log_wage ~ float_wage + int_education + int_experience),
            vars = [:float_wage, :int_education],
            expected_agreement = 0.85  # Good for econometric spec
        ),
        (
            name = "Logistic: Econometric spec",
            model_func = create_glm_model_func(Binomial(), LogitLink()),
            data_func = () -> make_econometric_data(n=800), 
            formula = @formula(union_member ~ float_wage + int_education),
            vars = [:float_wage, :int_education],
            expected_agreement = 0.75  # Moderate for complex GLM
        )
    ]
    
    return test_models
end

"""
    run_single_model_bootstrap_test(model_config; n_bootstrap=150, verbose=true)

Run bootstrap validation for a single model configuration.

# Arguments
- `model_config`: Model configuration tuple with name, model_func, data_func, formula, vars
- `n_bootstrap`: Number of bootstrap samples
- `verbose`: Print detailed results

# Returns
- `NamedTuple` with test results and validation statistics
"""
function run_single_model_bootstrap_test(model_config; n_bootstrap=150, verbose=true)
    if verbose
        @info "Testing: $(model_config.name)"
    end
    
    # Generate test data
    data = model_config.data_func()
    
    # Run 2×2 framework bootstrap validation
    try
        results = bootstrap_validate_2x2_framework(
            model_config.model_func, model_config.formula, data;
            vars=model_config.vars, n_bootstrap=n_bootstrap
        )
        
        # Assess results
        success = results.overall_success
        agreement_rate = results.overall_agreement_rate
        meets_expectation = agreement_rate >= model_config.expected_agreement
        
        if verbose
            @info "  Overall Agreement Rate: $(round(agreement_rate * 100, digits=1))% (expected: ≥$(round(model_config.expected_agreement * 100, digits=1))%)"
            @info "  Successful Quadrants: $(results.n_successful_quadrants)/4"
            
            # Detail by quadrant
            for (quadrant_name, quadrant_result) in results.quadrants
                if haskey(quadrant_result, :success) && quadrant_result.success
                    if haskey(quadrant_result, :validation)
                        quad_agreement = round(quadrant_result.validation.agreement_rate * 100, digits=1)
                        @info "    $(quadrant_name): $(quad_agreement)% agreement"
                    end
                else
                    reason = haskey(quadrant_result, :skipped) ? " ($(quadrant_result.reason))" : " (failed)"
                    @info "    $(quadrant_name): Skipped$reason"
                end
            end
            
            status = meets_expectation ? "✅ PASSED" : "❌ BELOW EXPECTATION"
            @info "  Result: $status"
        end
        
        return (
            model_name = model_config.name,
            success = success,
            agreement_rate = agreement_rate,
            expected_agreement = model_config.expected_agreement,
            meets_expectation = meets_expectation,
            n_successful_quadrants = results.n_successful_quadrants,
            detailed_results = results
        )
        
    catch e
        @error "Bootstrap test failed for $(model_config.name): $e"
        return (
            model_name = model_config.name,
            success = false,
            agreement_rate = 0.0,
            expected_agreement = model_config.expected_agreement,
            meets_expectation = false,
            n_successful_quadrants = 0,
            error = e
        )
    end
end

"""
    run_comprehensive_bootstrap_test_suite(; n_bootstrap=150, verbose=true)

Run bootstrap validation across all model types in the test suite.

# Arguments
- `n_bootstrap`: Number of bootstrap samples per model
- `verbose`: Print detailed progress and results

# Returns
- `NamedTuple` with overall results and individual model results
"""
function run_comprehensive_bootstrap_test_suite(; n_bootstrap=150, verbose=true)
    if verbose
        @info "Starting Comprehensive Bootstrap SE Validation Suite"
        @info "Bootstrap samples per model: $n_bootstrap"
        @info "="^60
    end
    
    test_models = multi_model_bootstrap_test_suite()
    individual_results = []
    
    for model_config in test_models
        result = run_single_model_bootstrap_test(model_config; n_bootstrap=n_bootstrap, verbose=verbose)
        push!(individual_results, result)
        
        if verbose
            @info ""  # Spacing between models
        end
    end
    
    # Overall assessment
    successful_models = [r for r in individual_results if r.success]
    models_meeting_expectation = [r for r in individual_results if r.meets_expectation]
    
    overall_success_rate = length(successful_models) / length(individual_results)
    expectation_success_rate = length(models_meeting_expectation) / length(individual_results)
    mean_agreement_rate = mean([r.agreement_rate for r in successful_models])
    
    if verbose
        @info "="^60
        @info "COMPREHENSIVE BOOTSTRAP VALIDATION SUMMARY"
        @info "="^60
        @info "Total Models Tested: $(length(individual_results))"
        @info "Successful Models: $(length(successful_models))/$(length(individual_results)) ($(round(overall_success_rate * 100, digits=1))%)"
        @info "Models Meeting Expectations: $(length(models_meeting_expectation))/$(length(individual_results)) ($(round(expectation_success_rate * 100, digits=1))%)"
        @info "Mean Agreement Rate: $(round(mean_agreement_rate * 100, digits=1))%"
        
        if expectation_success_rate >= 0.80
            @info "🎉 BOOTSTRAP VALIDATION SUITE: PASSED"
            @info "Standard errors show good bootstrap agreement across model types!"
        else
            @warn "⚠️  BOOTSTRAP VALIDATION SUITE: MIXED RESULTS"
            @info "Some models below expectation - detailed investigation recommended"
        end
    end
    
    return (
        overall_success_rate = overall_success_rate,
        expectation_success_rate = expectation_success_rate,
        mean_agreement_rate = mean_agreement_rate,
        individual_results = individual_results,
        n_models_tested = length(individual_results),
        n_successful = length(successful_models),
        n_meeting_expectation = length(models_meeting_expectation)
    )
end

"""
    quick_bootstrap_validation_check(; n_bootstrap=100)

Quick bootstrap validation check for development/CI purposes.
Tests a subset of models with fewer bootstrap samples.
"""
function quick_bootstrap_validation_check(; n_bootstrap=100)
    @debug "Quick Bootstrap Validation Check (n_bootstrap=$n_bootstrap)"
    
    # Test subset of models for speed
    quick_models = [
        (
            name = "Linear: y ~ x",
            model_func = lm,
            data_func = () -> make_simple_test_data(n=400, formula_type=:linear),
            formula = @formula(y ~ x),
            vars = [:x],
            expected_agreement = 0.90
        ),
        (
            name = "Logistic: y ~ x",
            model_func = create_glm_model_func(Binomial(), LogitLink()),
            data_func = () -> make_glm_test_data(n=400, family=:binomial),
            formula = @formula(y ~ x),
            vars = [:x],
            expected_agreement = 0.80
        )
    ]
    
    results = []
    for model_config in quick_models
        result = run_single_model_bootstrap_test(model_config; n_bootstrap=n_bootstrap, verbose=false)
        push!(results, result)
    end
    
    success_rate = mean([r.meets_expectation for r in results])
    
    @debug "Quick validation success rate: $(round(success_rate * 100, digits=1))%"
    
    return success_rate >= 0.75  # 75% threshold for quick check
end

# Export multi-model bootstrap testing functions
export multi_model_bootstrap_test_suite, run_single_model_bootstrap_test
export run_comprehensive_bootstrap_test_suite, quick_bootstrap_validation_check
export create_glm_model_func