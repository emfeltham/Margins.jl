# analytical_se_validation.jl - Analytical Standard Error Validation Utilities
#
# This file implements hand-calculated analytical standard error verification for
# Margins.jl, providing the same level of mathematical rigor for standard errors
# that we have for point estimates in statistical_validation.jl.
#
# **SCOPE**: Main effects models only. Interaction terms are currently unsupported
# due to complex coefficient name parsing requirements in _construct_eval_vector().
# Models with interactions will throw an ArgumentError to prevent silent incorrectness.
#
# Analytical SE validation for linear and GLM models using exact mathematical formulas.

using GLM
using LinearAlgebra  # For dot product
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
    
    if isnothing(var_index)
        throw(ArgumentError("Variable $var_symbol not found in model coefficients: $(coef_names)"))
    end
    
    # Extract SE from variance-covariance matrix diagonal
    vcov_matrix = GLM.vcov(model)
    analytical_se = sqrt(vcov_matrix[var_index, var_index])
    
    return analytical_se
end

"""
    analytical_logistic_se(model, data, var_symbol, at_values)

**MATHEMATICALLY CORRECT** SE calculation for logistic marginal effects using full gradient.

**NOTE**: This function was corrected after discovering that the original simplified delta method
was masking a fundamental bug in the main implementation. The mixed model test in
test_mixedmodels.jl correctly implemented the full gradient and exposed the bug.

For logistic marginal effects AME = mean(β_x × μ(1-μ)), the gradient w.r.t. β_j is:
∂(AME)/∂β_j = mean(δ_{j,x} × μ(1-μ) + β_x × μ(1-μ)(1-2μ) × X_j)

This accounts for:
1. Direct effect when j = x variable index
2. Chain rule effect through link function derivative dependency on all parameters

# Arguments
- `model`: Fitted logistic model (glm with LogitLink)
- `data`: DataFrame used to fit model
- `var_symbol`: Symbol of variable to compute SE for
- `at_values`: Dict or NamedTuple specifying covariate values for evaluation

# Returns
- Analytical standard error using complete gradient and full variance matrix

# Mathematical Foundation
Uses full chain rule: ∂(AME)/∂β = E[∂(β_x × w)/∂β] where w = μ(1-μ)
Delta method: SE = sqrt(g' × Σ × g) where g is the complete gradient vector

This matches the mathematical approach used in the mixed model test that exposed the bug.
"""
function analytical_logistic_se(model, data, var_symbol, at_values)
    # Get model components
    coef_names = GLM.coefnames(model)
    var_name = string(var_symbol)
    coeffs = GLM.coef(model)
    vcov_matrix = GLM.vcov(model)

    # Find variable index
    var_index = findfirst(name -> name == var_name, coef_names)
    if isnothing(var_index)
        throw(ArgumentError("Variable $var_symbol not found in model coefficients: $(coef_names)"))
    end

    # Construct evaluation vector matching model matrix structure
    x_eval = _construct_eval_vector(model, coef_names, at_values)

    # Compute linear predictor and probability at evaluation point
    η = dot(coeffs, x_eval)
    μ = 1 / (1 + exp(-η))

    # Compute complete gradient vector using full chain rule
    g = zeros(length(coeffs))

    for j in 1:length(coeffs)
        # Direct effect: ∂(β_x × w)/∂β_j when j == var_index
        direct_effect = (j == var_index) ? μ * (1 - μ) : 0.0

        # Chain rule effect: β_x × ∂w/∂β_j = β_x × (∂w/∂η) × (∂η/∂β_j)
        # where ∂w/∂η = μ(1-μ)(1-2μ) and ∂η/∂β_j = x_eval[j]
        chain_effect = coeffs[var_index] * μ * (1 - μ) * (1 - 2*μ) * x_eval[j]

        g[j] = direct_effect + chain_effect
    end

    # Full variance computation: SE = sqrt(g' × Σ × g)
    variance = g' * vcov_matrix * g
    analytical_se = sqrt(variance)

    return analytical_se
end

"""
    _construct_eval_vector(model, coef_names, at_values)

**CORRECTED** evaluation vector construction that properly handles interaction terms.

**CRITICAL FIX**: The original version had a fundamental bug where interaction terms
like "y & group: B" would incorrectly match variable "y" due to startswith() logic.
This caused wrong gradient computations for any model with interactions.

**NOTE**: This is a complex problem that requires proper parsing of StatsModels coefficient names.
For now, this function throws an error for interaction models to prevent silent incorrectness.

Handles intercept, continuous variables, categorical expansions, and interaction detection.
"""
function _construct_eval_vector(model, coef_names, at_values)
    x_eval = zeros(length(coef_names))
    x_eval[1] = 1.0  # Intercept

    # Check for interaction terms - these require special handling
    interaction_terms = [name for name in coef_names if contains(name, " & ")]
    if !isempty(interaction_terms)
        throw(ArgumentError(
            "INTERACTION TERMS DETECTED: Analytical SE validation does not currently support " *
            "models with interaction terms due to complex coefficient name parsing requirements. " *
            "\nInteraction terms found: $(interaction_terms)" *
            "\n\nThis prevents silent incorrect calculations. Use only main effects models for " *
            "analytical SE validation until interaction support is implemented."
        ))
    end

    for (i, coef_name) in enumerate(coef_names[2:end])
        coef_idx = i + 1

        # Find matching variable in at_values
        matched = false
        for (var_key, var_val) in pairs(at_values)
            var_key_str = string(var_key)

            # Exact match for continuous variables
            if coef_name == var_key_str
                x_eval[coef_idx] = var_val
                matched = true
                break
            # Categorical expansion (e.g., "genderFemale" matches "gender" = "Female")
            # FIXED: Use more precise matching to avoid interaction term confusion
            elseif startswith(coef_name, var_key_str) &&
                   !contains(coef_name, " & ") &&  # Not an interaction
                   (startswith(coef_name, var_key_str * ":") ||  # Standard categorical
                    coef_name == var_key_str * string(var_val))  # Direct categorical
                expected_level = coef_name[length(var_key_str)+1:end]
                if string(var_val) == expected_level
                    x_eval[coef_idx] = 1.0
                else
                    x_eval[coef_idx] = 0.0
                end
                matched = true
                break
            end
        end

        # If no match found, assume baseline/reference level (0)
        if !matched
            x_eval[coef_idx] = 0.0
        end
    end

    return x_eval
end

"""
    analytical_poisson_se(model, data, var_symbol, at_values)

**MATHEMATICALLY CORRECT** SE calculation for Poisson marginal effects using full gradient.

**NOTE**: This function was corrected to match the full gradient approach used for logistic models.

For Poisson marginal effects AME = mean(β_x × μ), the gradient w.r.t. β_j is:
∂(AME)/∂β_j = mean(δ_{j,x} × μ + β_x × μ × X_j)

This accounts for:
1. Direct effect when j = x variable index
2. Chain rule effect through link function derivative dependency on all parameters

# Arguments
- `model`: Fitted Poisson model (glm with LogLink)
- `data`: DataFrame used to fit model
- `var_symbol`: Symbol of variable to compute SE for
- `at_values`: Dict or NamedTuple specifying covariate values for evaluation

# Returns
- Analytical standard error using complete gradient and full variance matrix

# Mathematical Foundation
Uses full chain rule: ∂(AME)/∂β = E[∂(β_x × μ)/∂β] where μ = exp(η)
Delta method: SE = sqrt(g' × Σ × g) where g is the complete gradient vector
"""
function analytical_poisson_se(model, data, var_symbol, at_values)
    # Get model components
    coef_names = GLM.coefnames(model)
    var_name = string(var_symbol)
    coeffs = GLM.coef(model)
    vcov_matrix = GLM.vcov(model)

    # Find variable index
    var_index = findfirst(name -> name == var_name, coef_names)
    if isnothing(var_index)
        throw(ArgumentError("Variable $var_symbol not found in model coefficients: $(coef_names)"))
    end

    # Construct evaluation vector matching model matrix structure
    x_eval = _construct_eval_vector(model, coef_names, at_values)

    # Compute linear predictor and mean at evaluation point
    η = dot(coeffs, x_eval)
    μ = exp(η)

    # Compute complete gradient vector using full chain rule
    g = zeros(length(coeffs))

    for j in 1:length(coeffs)
        # Direct effect: ∂(β_x × μ)/∂β_j when j == var_index
        direct_effect = (j == var_index) ? μ : 0.0

        # Chain rule effect: β_x × ∂μ/∂β_j = β_x × (∂μ/∂η) × (∂η/∂β_j)
        # where ∂μ/∂η = μ (for log link) and ∂η/∂β_j = x_eval[j]
        chain_effect = coeffs[var_index] * μ * x_eval[j]

        g[j] = direct_effect + chain_effect
    end

    # Full variance computation: SE = sqrt(g' × Σ × g)
    variance = g' * vcov_matrix * g
    analytical_se = sqrt(variance)

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

**LIMITATION**: Only supports main effects models. Models with interaction terms
will throw an ArgumentError to prevent incorrect calculations.

# Arguments
- `model`: Fitted GLM model (**main effects only**)
- `data`: DataFrame used to fit model
- `var_symbol`: Variable to test
- `at_values`: Specific covariate values for profile evaluation
- `tolerance`: Numerical tolerance for comparison
- `model_type`: :logistic or :poisson for appropriate analytical function

# Returns
- NamedTuple with validation results comparing computed vs analytical SEs

# Throws
- `ArgumentError`: If model contains interaction terms (detected by " & " in coefficient names)
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
