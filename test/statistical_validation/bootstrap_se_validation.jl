# bootstrap_se_validation.jl - Systematic Bootstrap Standard Error Validation
#
# Systematic Bootstrap Validation Framework
# 
# This file implements comprehensive bootstrap validation for standard errors
# across all model types and 2×2 framework quadrants, extending beyond the
# single linear model test to provide systematic empirical SE validation.

using Random
using DataFrames
using Statistics
using GLM
using StatsModels
using Margins
using Test
using Printf

"""
    bootstrap_sample_with_replacement(data, n_obs)
    
Create a bootstrap sample with replacement from the original data.
"""
function bootstrap_sample_with_replacement(data, n_obs)
    boot_indices = rand(1:nrow(data), n_obs)
    return data[boot_indices, :]
end

"""
    bootstrap_margins_computation(model_func, formula, data, margins_func; 
                                n_bootstrap=200, vars=nothing, kwargs...)

Generic bootstrap computation for any margins function and model type.

# Arguments
- `model_func`: Function to fit model (e.g., `lm`, `m -> glm(m, data, Binomial(), LogitLink())`)
- `formula`: Model formula
- `data`: Original dataset
- `margins_func`: Function to compute margins (e.g., `population_margins`, `profile_margins`)
- `n_bootstrap`: Number of bootstrap samples
- `vars`: Variables to test (auto-detected if nothing)
- `kwargs...`: Additional arguments passed to margins_func

# Returns
- `(bootstrap_estimates, bootstrap_ses)`: Bootstrap means and standard errors
"""
function bootstrap_margins_computation(model_func, formula, data, margins_func; 
                                     n_bootstrap=200, vars=nothing, seed=123, kwargs...)
    Random.seed!(seed)
    n_obs = nrow(data)
    
    bootstrap_results = []
    
    for i in 1:n_bootstrap
        try
            # Bootstrap sample
            boot_data = bootstrap_sample_with_replacement(data, n_obs)
            
            # Refit model on bootstrap sample
            if hasmethod(model_func, Tuple{FormulaTerm, AbstractDataFrame})
                boot_model = model_func(formula, boot_data)
            else
                # For GLM with additional arguments
                boot_model = model_func(formula, boot_data)
            end
            
            # Compute margins on bootstrap sample
            if haskey(kwargs, :reference_grid) && margins_func === profile_margins
                # Handle profile_margins with reference grid as positional argument
                reference_grid = kwargs[:reference_grid]
                remaining_kwargs = Dict(k => v for (k, v) in kwargs if k !== :reference_grid)
                # Use :ad backend by default unless explicitly overridden
                backend = get(remaining_kwargs, :backend, :ad)
                remaining_kwargs_no_backend = Dict(k => v for (k, v) in remaining_kwargs if k !== :backend)
                if !isnothing(vars)
                    result = margins_func(boot_model, boot_data, reference_grid; vars=vars, backend=backend, remaining_kwargs_no_backend...)
                else
                    result = margins_func(boot_model, boot_data, reference_grid; backend=backend, remaining_kwargs_no_backend...)
                end
            else
                # Handle population_margins or other functions with keyword arguments
                # Use :ad backend by default unless explicitly overridden
                backend = get(kwargs, :backend, :ad)
                kwargs_no_backend = Dict(k => v for (k, v) in kwargs if k !== :backend)
                if !isnothing(vars)
                    result = margins_func(boot_model, boot_data; vars=vars, backend=backend, kwargs_no_backend...)
                else
                    result = margins_func(boot_model, boot_data; backend=backend, kwargs_no_backend...)
                end
            end
            
            boot_estimates = DataFrame(result).estimate
            push!(bootstrap_results, boot_estimates)
            
        catch e
            # Skip failed bootstrap samples (convergence failures, etc.)
            continue
        end
    end
    
    if isempty(bootstrap_results)
        @debug "All bootstrap samples failed for model: $formula"
        @test false
        return NaN, NaN  # Return appropriate values to continue
    end
    
    # Convert to matrix for easier computation
    boot_matrix = hcat(bootstrap_results...)'  # Each row is a bootstrap sample
    
    # Compute empirical standard errors
    bootstrap_ses = [std(boot_matrix[:, j]) for j in 1:size(boot_matrix, 2)]
    bootstrap_means = [mean(boot_matrix[:, j]) for j in 1:size(boot_matrix, 2)]
    
    return bootstrap_means, bootstrap_ses, length(bootstrap_results)
end

"""
    validate_bootstrap_se_agreement(computed_ses, bootstrap_ses; tolerance=0.15, var_names=nothing)

Validate that computed standard errors agree with bootstrap SEs within tolerance.

# Arguments
- `computed_ses`: Standard errors from delta method
- `bootstrap_ses`: Standard errors from bootstrap
- `tolerance`: Relative error tolerance (default 15%)
- `var_names`: Variable names for reporting

# Returns
- `NamedTuple` with agreement statistics and individual results
"""
function validate_bootstrap_se_agreement(computed_ses, bootstrap_ses; tolerance=0.15, var_names=nothing)
    @assert length(computed_ses) == length(bootstrap_ses)
    
    if isnothing(var_names)
        var_names = ["var_$i" for i in 1:length(computed_ses)]
    end
    
    agreements = Bool[]
    ratios = Float64[]
    
    for i in 1:length(computed_ses)
        ratio = computed_ses[i] / bootstrap_ses[i]
        relative_error = abs(ratio - 1.0)
        agreement = relative_error < tolerance
        
        push!(agreements, agreement)
        push!(ratios, ratio)
    end
    
    agreement_rate = mean(agreements)
    mean_ratio = mean(ratios)
    max_relative_error = maximum(abs.(ratios .- 1.0))
    
    return (
        agreement_rate = agreement_rate,
        agreements = agreements,
        ratios = ratios,
        mean_ratio = mean_ratio,
        max_relative_error = max_relative_error,
        var_names = var_names,
        tolerance = tolerance,
        n_variables = length(computed_ses)
    )
end

"""
    bootstrap_validate_population_effects(model_func, formula, data; vars=nothing, n_bootstrap=200)

Bootstrap validation for population marginal effects (AME).
"""
function bootstrap_validate_population_effects(model_func, formula, data; vars=nothing, n_bootstrap=200)
    # Fit original model
    original_model = model_func(formula, data)
    
    # Compute original margins
    if !isnothing(vars)
        original_result = population_margins(original_model, data; type=:effects, vars=vars, backend=:fd)
    else
        original_result = population_margins(original_model, data; type=:effects, backend=:fd)
    end
    
    original_df = DataFrame(original_result)
    computed_ses = original_df.se
    var_names = original_df.variable
    
    # Bootstrap computation
    boot_means, boot_ses, n_successful = bootstrap_margins_computation(
        model_func, formula, data, population_margins;
        n_bootstrap=n_bootstrap, vars=vars, type=:effects
    )
    
    # Validate agreement
    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; var_names=var_names)
    
    return (
        computed_ses = computed_ses,
        bootstrap_ses = boot_ses,
        validation = validation,
        n_bootstrap_successful = n_successful,
        var_names = var_names
    )
end

"""
    bootstrap_validate_population_predictions(model_func, formula, data; n_bootstrap=200)

Bootstrap validation for population predictions (AAP).
"""
function bootstrap_validate_population_predictions(model_func, formula, data; n_bootstrap=200)
    # Fit original model
    original_model = model_func(formula, data)
    
    # Compute original margins
    original_result = population_margins(original_model, data; type=:predictions, backend=:fd)
    original_df = DataFrame(original_result)
    computed_ses = original_df.se
    
    # Bootstrap computation
    boot_means, boot_ses, n_successful = bootstrap_margins_computation(
        model_func, formula, data, population_margins;
        n_bootstrap=n_bootstrap, type=:predictions
    )
    
    # Validate agreement
    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; var_names=["prediction"])
    
    return (
        computed_ses = computed_ses,
        bootstrap_ses = boot_ses,
        validation = validation,
        n_bootstrap_successful = n_successful
    )
end

"""
    bootstrap_validate_profile_effects(model_func, formula, data; vars=nothing, reference_grid=nothing, n_bootstrap=200)

Bootstrap validation for profile marginal effects (MEM).
Uses means_grid(data) as default reference grid if not provided.
"""
function bootstrap_validate_profile_effects(model_func, formula, data; vars=nothing, reference_grid=nothing, n_bootstrap=200)
    # Fit original model
    original_model = model_func(formula, data)
    
    # Create reference grid if not provided
    if isnothing(reference_grid)
        reference_grid = means_grid(data)
    end
    
    # Compute original margins
    if !isnothing(vars)
        original_result = profile_margins(original_model, data, reference_grid; type=:effects, vars=vars, backend=:ad)
    else
        original_result = profile_margins(original_model, data, reference_grid; type=:effects, backend=:ad)
    end
    
    original_df = DataFrame(original_result)
    computed_ses = original_df.se
    var_names = original_df.variable
    
    # Bootstrap computation
    boot_means, boot_ses, n_successful = bootstrap_margins_computation(
        model_func, formula, data, profile_margins;
        n_bootstrap=n_bootstrap, vars=vars, type=:effects, reference_grid=reference_grid, backend=:ad
    )
    
    # Validate agreement
    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; var_names=var_names)
    
    return (
        computed_ses = computed_ses,
        bootstrap_ses = boot_ses,
        validation = validation,
        n_bootstrap_successful = n_successful,
        var_names = var_names
    )
end

"""
    bootstrap_validate_profile_predictions(model_func, formula, data; reference_grid=nothing, n_bootstrap=200)

Bootstrap validation for profile predictions (APM).
Uses means_grid(data) as default reference grid if not provided.
"""
function bootstrap_validate_profile_predictions(model_func, formula, data; reference_grid=nothing, n_bootstrap=200)
    # Fit original model
    original_model = model_func(formula, data)
    
    # Create reference grid if not provided
    if isnothing(reference_grid)
        reference_grid = means_grid(data)
    end
    
    # Compute original margins
    original_result = profile_margins(original_model, data, reference_grid; type=:predictions, backend=:ad)
    original_df = DataFrame(original_result)
    computed_ses = original_df.se
    
    # Bootstrap computation
    boot_means, boot_ses, n_successful = bootstrap_margins_computation(
        model_func, formula, data, profile_margins;
        n_bootstrap=n_bootstrap, type=:predictions, reference_grid=reference_grid, backend=:ad
    )
    
    # Validate agreement
    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; var_names=["prediction"])
    
    return (
        computed_ses = computed_ses,
        bootstrap_ses = boot_ses,
        validation = validation,
        n_bootstrap_successful = n_successful
    )
end

"""
    bootstrap_validate_2x2_framework(model_func, formula, data; vars=nothing, n_bootstrap=150)

Comprehensive bootstrap validation across all four 2×2 framework quadrants.

# Arguments
- `model_func`: Function to create model (e.g., `lm` or `(f,d) -> glm(f, d, Binomial(), LogitLink())`)
- `formula`: Model formula
- `data`: Dataset
- `vars`: Variables for effects testing (auto-detected if nothing)
- `n_bootstrap`: Number of bootstrap samples per quadrant

# Returns  
- `NamedTuple` with validation results for all four quadrants
"""
function bootstrap_validate_2x2_framework(model_func, formula, data; vars=nothing, n_bootstrap=150)
    results = Dict()
    
    # Determine continuous variables if not specified
    if isnothing(vars)
        # Fit model to extract variable information
        sample_model = model_func(formula, data)
        coef_names = GLM.coefnames(sample_model)
        
        # Extract continuous variable names (simple heuristic)
        potential_vars = String[]
        for coef_name in coef_names[2:end]  # Skip intercept
            for col_name in names(data)
                if coef_name == col_name && isa(data[!, col_name], AbstractVector{<:Number})
                    push!(potential_vars, col_name)
                    break
                end
            end
        end
        
        vars = [Symbol(name) for name in potential_vars[1:min(2, length(potential_vars))]]
    end
    
    # 1. Population Effects (AME)
    if !isempty(vars)
        try
            results[:population_effects] = bootstrap_validate_population_effects(
                model_func, formula, data; vars=vars, n_bootstrap=n_bootstrap
            )
            results[:population_effects] = merge(results[:population_effects], (success=true,))
        catch e
            results[:population_effects] = (success=false, error=e)
        end
    else
        results[:population_effects] = (success=false, skipped=true, reason="No continuous variables for effects")
    end
    
    # 2. Population Predictions (AAP)
    try
        results[:population_predictions] = bootstrap_validate_population_predictions(
            model_func, formula, data; n_bootstrap=n_bootstrap
        )
        results[:population_predictions] = merge(results[:population_predictions], (success=true,))
    catch e
        results[:population_predictions] = (success=false, error=e)
    end
    
    # 3. Profile Effects (MEM)
    if !isempty(vars)
        try
            results[:profile_effects] = bootstrap_validate_profile_effects(
                model_func, formula, data; vars=vars, reference_grid=means_grid(data), n_bootstrap=n_bootstrap
            )
            results[:profile_effects] = merge(results[:profile_effects], (success=true,))
        catch e
            results[:profile_effects] = (success=false, error=e)
        end
    else
        results[:profile_effects] = (success=false, skipped=true, reason="No continuous variables for effects")
    end
    
    # 4. Profile Predictions (APM)  
    try
        results[:profile_predictions] = bootstrap_validate_profile_predictions(
            model_func, formula, data; reference_grid=means_grid(data), n_bootstrap=n_bootstrap
        )
        results[:profile_predictions] = merge(results[:profile_predictions], (success=true,))
    catch e
        results[:profile_predictions] = (success=false, error=e)
    end
    
    # Overall assessment
    successful_quadrants = [r for r in values(results) if haskey(r, :success) && r.success]
    overall_agreement_rates = [r.validation.agreement_rate for r in successful_quadrants if haskey(r, :validation)]
    
    overall_success = length(successful_quadrants) >= 2  # At least half successful
    overall_agreement = length(overall_agreement_rates) > 0 ? mean(overall_agreement_rates) : 0.0
    
    return (
        quadrants = results,
        overall_success = overall_success,
        overall_agreement_rate = overall_agreement,
        n_successful_quadrants = length(successful_quadrants),
        model_formula = formula,
        variables_tested = vars
    )
end

# Export bootstrap validation functions
export bootstrap_margins_computation, validate_bootstrap_se_agreement
export bootstrap_validate_population_effects, bootstrap_validate_population_predictions
export bootstrap_validate_profile_effects, bootstrap_validate_profile_predictions  
export bootstrap_validate_2x2_framework