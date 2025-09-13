# analytical_se_validation.jl - Analytical Standard Error Validation Utilities
#
# This file implements hand-calculated analytical standard error verification for 
# Margins.jl, providing the same level of mathematical rigor for standard errors
# that we have for point estimates in statistical_validation.jl.
#
# Analytical SE validation for linear
# and GLM models using exact mathematical formulas.

using GLM
using DataFrames
using Statistics
using StatsModels

"""
    analytical_linear_se(model, data, var_symbol)

Hand-calculate standard error for linear model marginal effect.
For linear model y ~ x: SE(∂y/∂x) = SE(β₁) = sqrt(vcov[2,2])

# Arguments
- `model`: Fitted linear model (lm)
- `data`: DataFrame used to fit model
- `var_symbol`: Symbol of variable to compute SE for

# Returns
- Analytical standard error for comparison with computed SE

# Mathematical Foundation
In linear models, marginal effects are constant and equal to coefficients.
Therefore, SE(marginal effect) = SE(coefficient), extracted directly from vcov matrix.
"""
function analytical_linear_se(model, data, var_symbol)
    # Get coefficient names to find position of variable
    coef_names = GLM.coefnames(model)
    var_name = string(var_symbol)
    
    # Find position in coefficient vector
    var_index = findfirst(name -> name == var_name, coef_names)
    
    if var_index === nothing
        throw(ArgumentError("Variable $var_symbol not found in model coefficients: $(coef_names)"))
    end
    
    # Extract SE from variance-covariance matrix diagonal
    vcov_matrix = GLM.vcov(model)
    analytical_se = sqrt(vcov_matrix[var_index, var_index])
    
    return analytical_se
end

"""
    analytical_logistic_se(model, data, var_symbol, at_values)

Hand-calculate SE for logistic marginal effect using delta method.
For logistic: ∂μ/∂x = β₁ × μ(1-μ), so SE via delta method.

# Arguments  
- `model`: Fitted logistic model (glm with LogitLink)
- `data`: DataFrame used to fit model
- `var_symbol`: Symbol of variable to compute SE for
- `at_values`: Dict or NamedTuple specifying covariate values for evaluation

# Returns
- Analytical standard error using delta method formula

# Mathematical Foundation
For logistic regression with chain rule: ∂μ/∂x = β₁ × μ(1-μ)
Delta method: SE = |∂f/∂β| × SE(β₁) = |μ(1-μ)| × SE(β₁)
where μ = 1/(1 + exp(-η)) at the specified covariate values.
"""
function analytical_logistic_se(model, data, var_symbol, at_values)
    # Get coefficient information
    coef_names = GLM.coefnames(model)
    var_name = string(var_symbol)
    
    # Find position in coefficient vector
    var_index = findfirst(name -> name == var_name, coef_names)
    
    if var_index === nothing
        throw(ArgumentError("Variable $var_symbol not found in model coefficients: $(coef_names)"))
    end
    
    # Get coefficient SE from vcov matrix
    vcov_matrix = GLM.vcov(model)
    coef_se = sqrt(vcov_matrix[var_index, var_index])
    
    # Construct linear predictor at specified values
    coeffs = GLM.coef(model)
    linear_pred = coeffs[1]  # Intercept
    
    # Add contribution from each variable
    for (i, coef_name) in enumerate(coef_names[2:end])  # Skip intercept
        coef_idx = i + 1  # Account for intercept
        
        # Find base variable name (handle categorical expansions)
        base_var = nothing
        for var_key in keys(at_values)
            if startswith(coef_name, string(var_key))
                base_var = var_key
                break
            elseif coef_name == string(var_key)
                base_var = var_key
                break
            end
        end
        
        if base_var !== nothing
            linear_pred += coeffs[coef_idx] * at_values[base_var]
        else
            # For categorical variables not in at_values, assume baseline (0)
            # This handles categorical expansions like "genderFemale" 
        end
    end
    
    # Compute probability and derivative factor
    mu = 1 / (1 + exp(-linear_pred))
    derivative_factor = abs(mu * (1 - mu))  # |μ(1-μ)|
    
    # Apply delta method: SE = |derivative| × SE(β)
    analytical_se = derivative_factor * coef_se
    
    return analytical_se
end

"""
    analytical_poisson_se(model, data, var_symbol, at_values)

Hand-calculate SE for Poisson marginal effect using delta method.
For Poisson with log link: ∂μ/∂x = β₁ × μ, so SE via delta method.

# Arguments
- `model`: Fitted Poisson model (glm with LogLink)  
- `data`: DataFrame used to fit model
- `var_symbol`: Symbol of variable to compute SE for
- `at_values`: Dict or NamedTuple specifying covariate values for evaluation

# Returns
- Analytical standard error using delta method formula

# Mathematical Foundation
For Poisson regression with log link: ∂μ/∂x = β₁ × μ
Delta method: SE = |∂f/∂β| × SE(β₁) = |μ| × SE(β₁)
where μ = exp(η) at the specified covariate values.
"""
function analytical_poisson_se(model, data, var_symbol, at_values)
    # Get coefficient information
    coef_names = GLM.coefnames(model)
    var_name = string(var_symbol)
    
    # Find position in coefficient vector
    var_index = findfirst(name -> name == var_name, coef_names)
    
    if var_index === nothing
        throw(ArgumentError("Variable $var_symbol not found in model coefficients: $(coef_names)"))
    end
    
    # Get coefficient SE from vcov matrix
    vcov_matrix = GLM.vcov(model)
    coef_se = sqrt(vcov_matrix[var_index, var_index])
    
    # Construct linear predictor at specified values
    coeffs = GLM.coef(model)
    linear_pred = coeffs[1]  # Intercept
    
    # Add contribution from each variable
    for (i, coef_name) in enumerate(coef_names[2:end])  # Skip intercept
        coef_idx = i + 1  # Account for intercept
        
        # Find base variable name
        base_var = nothing
        for var_key in keys(at_values)
            if startswith(coef_name, string(var_key)) || coef_name == string(var_key)
                base_var = var_key
                break
            end
        end
        
        if base_var !== nothing
            linear_pred += coeffs[coef_idx] * at_values[base_var]
        end
    end
    
    # Compute mean and derivative factor
    mu = exp(linear_pred)
    derivative_factor = abs(mu)  # |μ|
    
    # Apply delta method: SE = |derivative| × SE(β)
    analytical_se = derivative_factor * coef_se
    
    return analytical_se
end

"""
    compute_population_linear_se(model, data, var_symbol)

Compute population-level (AME) standard error for linear model analytically.
For linear models, population and profile SEs are identical since effects are constant.

# Arguments
- `model`: Fitted linear model
- `data`: DataFrame used to fit model  
- `var_symbol`: Variable symbol

# Returns
- Analytical population SE (same as coefficient SE for linear models)
"""
function compute_population_linear_se(model, data, var_symbol)
    # For linear models, AME SE = coefficient SE (constant marginal effects)
    return analytical_linear_se(model, data, var_symbol)
end

"""
    verify_linear_se_consistency(model, data, var_symbol; tolerance=1e-12)

Verify that computed standard errors match analytical values for linear models.
Tests both population and profile approaches.

# Arguments
- `model`: Fitted linear model
- `data`: DataFrame used to fit model
- `var_symbol`: Variable to test
- `tolerance`: Numerical tolerance for comparison

# Returns
- NamedTuple with validation results for both population and profile approaches
"""
function verify_linear_se_consistency(model, data, var_symbol; tolerance=1e-12)
    analytical_se = analytical_linear_se(model, data, var_symbol)
    
    # Test population margins SE
    pop_result = population_margins(model, data; type=:effects, vars=[var_symbol])
    pop_df = DataFrame(pop_result)
    pop_se = pop_df.se[1]
    pop_matches = abs(pop_se - analytical_se) < tolerance
    
    # Test profile margins SE  
    profile_result = profile_margins(model, data, means_grid(data); type=:effects, vars=[var_symbol])
    prof_df = DataFrame(profile_result)
    prof_se = prof_df.se[1]
    prof_matches = abs(prof_se - analytical_se) < tolerance
    
    return (
        analytical_se = analytical_se,
        population_se = pop_se,
        profile_se = prof_se,
        population_matches = pop_matches,
        profile_matches = prof_matches,
        both_match = pop_matches && prof_matches,
        max_deviation = max(abs(pop_se - analytical_se), abs(prof_se - analytical_se))
    )
end

"""
    verify_glm_se_chain_rule(model, data, var_symbol, at_values; tolerance=1e-10, model_type=:logistic)

Verify that computed GLM standard errors match analytical delta method calculations.

# Arguments
- `model`: Fitted GLM model
- `data`: DataFrame used to fit model
- `var_symbol`: Variable to test
- `at_values`: Specific covariate values for profile evaluation
- `tolerance`: Numerical tolerance for comparison
- `model_type`: :logistic or :poisson for appropriate analytical function

# Returns
- NamedTuple with validation results comparing computed vs analytical SEs
"""
function verify_glm_se_chain_rule(model, data, var_symbol, at_values; tolerance=1e-10, model_type=:logistic)
    # Compute analytical SE using appropriate method
    if model_type == :logistic
        analytical_se = analytical_logistic_se(model, data, var_symbol, at_values)
    elseif model_type == :poisson
        analytical_se = analytical_poisson_se(model, data, var_symbol, at_values)
    else
        throw(ArgumentError("Unsupported model_type: $model_type. Use :logistic or :poisson"))
    end
    
    # Test computed profile SE at the same point
    reference_grid = DataFrame(at_values)  # Convert Dict to DataFrame for reference grid
    profile_result = profile_margins(model, data, reference_grid; type=:effects, vars=[var_symbol], 
                                   scale=:response)  # Response scale for chain rule
    prof_df = DataFrame(profile_result) 
    computed_se = prof_df.se[1]
    
    se_matches = abs(computed_se - analytical_se) < tolerance
    
    return (
        analytical_se = analytical_se,
        computed_se = computed_se,
        matches = se_matches,
        deviation = abs(computed_se - analytical_se),
        relative_error = abs(computed_se - analytical_se) / analytical_se
    )
end

# Export analytical SE validation functions
export analytical_linear_se, analytical_logistic_se, analytical_poisson_se
export compute_population_linear_se
export verify_linear_se_consistency, verify_glm_se_chain_rule