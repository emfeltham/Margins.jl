# elasticity_se_validation.jl - Bootstrap Standard Error Validation for Elasticity Measures
#
# Elasticity SE Validation
#
# This file implements comprehensive bootstrap validation for elasticity standard errors
# including :elasticity, :semielasticity_dyex, and :semielasticity_eydx measures.
# Validates that delta-method standard errors match bootstrap estimates for all
# elasticity transformations across both population and profile approaches.

using Random
using DataFrames
using Statistics
using GLM
using StatsModels
using Margins
using Test
using Printf

# Load testing utilities (needed when running standalone)
if !isdefined(Main, :make_econometric_data)
    include("testing_utilities.jl")
    include("bootstrap_se_validation.jl")
    include("analytical_se_validation.jl")
end

"""
    bootstrap_validate_population_elasticity(model_func, formula, data, measure; vars=nothing, n_bootstrap=200)

Bootstrap validation for population elasticity measures (AME with elasticity transformation).

# Arguments
- `model_func`: Function to fit model (e.g., `lm`)
- `formula`: Model formula
- `data`: Original dataset  
- `measure`: Elasticity measure (`:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx`)
- `vars`: Variables to test (auto-detected if nothing)
- `n_bootstrap`: Number of bootstrap samples

# Returns
- NamedTuple with computed SEs, bootstrap SEs, and validation statistics
"""
function bootstrap_validate_population_elasticity(model_func, formula, data, measure; vars=nothing, n_bootstrap=200)
    # Fit original model
    original_model = model_func(formula, data)
    
    # Compute original elasticity margins
    if !isnothing(vars)
        original_result = population_margins(original_model, data; type=:effects, vars=vars, measure=measure, backend=:fd)
    else
        original_result = population_margins(original_model, data; type=:effects, measure=measure, backend=:fd)
    end
    
    original_df = DataFrame(original_result)
    computed_ses = original_df.se
    var_names = original_df.variable
    
    # Bootstrap computation
    boot_means, boot_ses, n_successful = bootstrap_margins_computation(
        model_func, formula, data, population_margins;
        n_bootstrap=n_bootstrap, vars=vars, type=:effects, measure=measure
    )
    
    # Validate agreement
    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; var_names=var_names)
    
    return (
        computed_ses = computed_ses,
        bootstrap_ses = boot_ses,
        validation = validation,
        measure = measure,
        n_successful_bootstrap = n_successful,
        var_names = var_names,
        estimates = original_df.estimate
    )
end

"""
    bootstrap_validate_profile_elasticity(model_func, formula, data, reference_grid, measure; vars=nothing, n_bootstrap=200)

Bootstrap validation for profile elasticity measures at specific reference points.

# Arguments
- `model_func`: Function to fit model
- `formula`: Model formula
- `data`: Original dataset
- `reference_grid`: Reference grid for profile evaluation (DataFrame or grid function)
- `measure`: Elasticity measure
- `vars`: Variables to test
- `n_bootstrap`: Number of bootstrap samples

# Returns
- NamedTuple with computed SEs, bootstrap SEs, and validation statistics
"""
function bootstrap_validate_profile_elasticity(model_func, formula, data, reference_grid, measure; vars=nothing, n_bootstrap=200)
    # Fit original model
    original_model = model_func(formula, data)
    
    # Compute original profile elasticity margins
    if !isnothing(vars)
        original_result = profile_margins(original_model, data, reference_grid; type=:effects, vars=vars, measure=measure, backend=:fd)
    else
        original_result = profile_margins(original_model, data, reference_grid; type=:effects, measure=measure, backend=:fd)
    end
    
    original_df = DataFrame(original_result)
    computed_ses = original_df.se
    var_names = original_df.variable
    
    # Bootstrap computation using modified function for profile margins
    boot_means, boot_ses, n_successful = bootstrap_profile_computation(
        model_func, formula, data, reference_grid;
        n_bootstrap=n_bootstrap, vars=vars, measure=measure
    )
    
    # Validate agreement
    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; var_names=var_names)
    
    return (
        computed_ses = computed_ses,
        bootstrap_ses = boot_ses,
        validation = validation,
        measure = measure,
        n_successful_bootstrap = n_successful,
        var_names = var_names,
        estimates = original_df.estimate
    )
end

"""
    bootstrap_profile_computation(model_func, formula, data, reference_grid; 
                                 n_bootstrap=200, vars=nothing, measure=:effect, kwargs...)

Bootstrap computation specifically for profile margins with reference grids.
"""
function bootstrap_profile_computation(model_func, formula, data, reference_grid; 
                                     n_bootstrap=200, vars=nothing, measure=:effect, seed=123, kwargs...)
    Random.seed!(seed)
    n_obs = nrow(data)
    
    bootstrap_results = []
    
    for i in 1:n_bootstrap
        try
            # Bootstrap sample
            boot_data = bootstrap_sample_with_replacement(data, n_obs)
            
            # Refit model on bootstrap sample
            boot_model = model_func(formula, boot_data)
            
            # Compute profile margins on bootstrap sample
            if !isnothing(vars)
                result = profile_margins(boot_model, boot_data, reference_grid; vars=vars, measure=measure, backend=:fd, kwargs...)
            else
                result = profile_margins(boot_model, boot_data, reference_grid; measure=measure, backend=:fd, kwargs...)
            end
            
            boot_estimates = DataFrame(result).estimate
            push!(bootstrap_results, boot_estimates)
            
        catch e
            # Skip failed bootstrap samples (convergence failures, etc.)
            continue
        end
    end
    
    n_successful = length(bootstrap_results)
    
    if n_successful < 10
        @debug "Bootstrap failed: only $n_successful successful samples out of $n_bootstrap"
        @test false
        return NaN, NaN, 0  # Return appropriate values to continue
    end
    
    # Convert to matrix for easier computation
    if n_successful > 0
        result_matrix = hcat(bootstrap_results...)  # Each column is one bootstrap sample
        bootstrap_means = vec(mean(result_matrix, dims=2))
        bootstrap_ses = vec(std(result_matrix, dims=2))
    else
        @debug "No successful bootstrap samples"
        @test false
        return NaN, NaN, 0  # Return appropriate values to continue
    end
    
    return bootstrap_means, bootstrap_ses, n_successful
end

"""
    run_comprehensive_elasticity_se_tests()

Run comprehensive elasticity SE validation across multiple measures and model types.
"""
function run_comprehensive_elasticity_se_tests()
    @info "Comprehensive Elasticity SE Validation Framework"
    
    # Generate test data
    data = make_econometric_data(n=300, seed=42)
    
    # Test measures to validate
    measures = [:elasticity, :semielasticity_dyex, :semielasticity_eydx]
    
    # Test models
    models_to_test = [
        ("Linear Model", lm, @formula(wage ~ int_education + int_experience), data),
        ("GLM Logit", formula_data -> glm(formula_data[1], formula_data[2], Binomial(), LogitLink()), 
         @formula(union_member ~ int_education + int_experience), data)
    ]
    
    validation_results = []
    
    for (model_name, model_func, formula, test_data) in models_to_test
        @info "Model validation: $model_name"
        
        for measure in measures
            @info "Elasticity measure validation: $measure"
            
            try
                # Test population elasticity
                pop_result = bootstrap_validate_population_elasticity(
                    model_func, formula, test_data, measure; n_bootstrap=100
                )
                
                @info "Population $measure validation: Agreement rate = $(round(pop_result.validation.agreement_rate, digits=3))"
                
                # Test profile elasticity at means
                means_ref_grid = means_grid(test_data)
                profile_result = bootstrap_validate_profile_elasticity(
                    model_func, formula, test_data, means_ref_grid, measure; n_bootstrap=100
                )
                
                @info "Profile $measure validation: Agreement rate = $(round(profile_result.validation.agreement_rate, digits=3))"
                
                push!(validation_results, (
                    model = model_name,
                    measure = measure,
                    population_agreement = pop_result.validation.agreement_rate,
                    profile_agreement = profile_result.validation.agreement_rate,
                    population_max_error = pop_result.validation.max_relative_error,
                    profile_max_error = profile_result.validation.max_relative_error
                ))
                
            catch e
                @info "Validation error encountered: $e"
                push!(validation_results, (
                    model = model_name,
                    measure = measure,
                    population_agreement = NaN,
                    profile_agreement = NaN,
                    population_max_error = NaN,
                    profile_max_error = NaN
                ))
            end
        end
    end
    
    return validation_results
end

# Test integration with main test suite
@testset "Elasticity SE Validation" begin
    # Generate consistent test data
    # n=500 needed for delta-method elasticity SEs to converge with bootstrap
    # (elasticity involves nonlinear transformation ∂y/∂x * x̄/ȳ, amplifying small-sample bias)
    Random.seed!(06515)
    data = make_econometric_data(n=500, seed=42)

    # Test all variables (not just one) so agreement_rate is a proportion rather
    # than binary 0/1, which makes the tests robust to RNG stream differences
    # across Julia versions.  B=500 gives bootstrap SE precision of ~4.5%.

    @testset "Population Elasticity SE Bootstrap Validation" begin
        model = lm(@formula(wage ~ int_education + int_experience), data)

        result = bootstrap_validate_population_elasticity(
            lm, @formula(wage ~ int_education + int_experience), data, :elasticity;
            n_bootstrap=500
        )

        @test result.validation.agreement_rate > 0.8
        @test result.validation.max_relative_error < 0.3
        @test result.n_successful_bootstrap >= 400
        @test all(result.computed_ses .> 0)
        @test all(result.bootstrap_ses .> 0)
    end

    @testset "Profile Elasticity SE Bootstrap Validation" begin
        means_ref_grid = means_grid(data)
        result = bootstrap_validate_profile_elasticity(
            lm, @formula(wage ~ int_education + int_experience), data, means_ref_grid, :elasticity;
            n_bootstrap=500
        )

        @test result.validation.agreement_rate > 0.8
        @test result.validation.max_relative_error < 0.3
        @test result.n_successful_bootstrap >= 400
        @test all(result.computed_ses .> 0)
        @test all(result.bootstrap_ses .> 0)
    end

    @testset "Semielasticity SE Validation" begin
        for measure in [:semielasticity_dyex, :semielasticity_eydx]
            result = bootstrap_validate_population_elasticity(
                lm, @formula(wage ~ int_education + int_experience), data, measure;
                n_bootstrap=500
            )

            @test result.validation.agreement_rate > 0.7
            @test result.validation.max_relative_error < 0.4
            @test result.n_successful_bootstrap >= 400
            @test all(result.computed_ses .> 0)
        end
    end

    @testset "GLM Elasticity SE Validation" begin
        result = bootstrap_validate_population_elasticity(
            (f, d) -> glm(f, d, Binomial(), LogitLink()),
            @formula(union_member ~ int_education + int_experience),
            data, :elasticity;
            n_bootstrap=500
        )

        @test result.validation.agreement_rate > 0.7
        @test result.validation.max_relative_error < 0.5
        @test all(result.computed_ses .> 0)
        @test all(result.bootstrap_ses .> 0)
    end
end

# Optionally run comprehensive tests if environment variable is set
if get(ENV, "MARGINS_COMPREHENSIVE_TESTS", "false") == "true"
    @testset "Comprehensive Elasticity SE Testing" begin
        results = run_comprehensive_elasticity_se_tests()
        
        # Overall validation - at least 80% of tests should have good agreement
        good_agreements = sum(r.population_agreement > 0.8 for r in results if !isnan(r.population_agreement))
        total_tests = sum(!isnan(r.population_agreement) for r in results)
        
        @test good_agreements / total_tests > 0.8
        
        @info "Elasticity SE Validation Summary:"
        @info "$(good_agreements) of $(total_tests) tests satisfied agreement criteria (>80% agreement rate)"
    end
end

@info "Elasticity SE validation framework loaded successfully"