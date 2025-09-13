# profile/contrasts.jl - Row-specific baseline contrasts

"""
    Row-specific categorical contrasts for profile margins.

This module implements "novel row-specific contrasts".
Unlike population contrasts (which average across all observations), row-specific contrasts
compute the effect of changing a categorical variable at a specific profile/covariate combination.

## Key Innovation

Traditional approach (population contrasts):
- Average across all observations: E[Y|X=level₁] - E[Y|X=baseline]

Row-specific approach (profile contrasts):  
- At specific profile: Y(profile with level₁) - Y(profile with baseline)
- Enables interpretation like "effect of treatment vs control for a 45-year-old college graduate"

## Mathematical Foundation

For categorical variable C with levels {baseline, level₁, level₂, ...} at profile x:

**Effect**: f(x, C=level₁) - f(x, C=baseline)
**Gradient**: ∇β[f(x, C=level₁) - f(x, C=baseline)] = ∇β f(x₁) - ∇β f(x₀)

Where:
- x₁ = profile with C=level₁  
- x₀ = profile with C=baseline
- f = prediction function (η or μ scale)
"""

"""
    compute_profile_categorical_contrast(engine, profile, var, scale, :ad) -> (effect, gradient)

Compute row-specific categorical contrast at a specific profile.

# Arguments
- `engine::MarginsEngine`: Pre-built computation engine
- `profile::Dict`: Covariate values defining the profile  
- `var::Symbol`: Categorical variable name
- `scale::Symbol`: `:link` or `:response` scale

# Returns
- `effect::Float64`: Contrast effect (current_level - baseline_level)
- `gradient::Vector{Float64}`: Parameter gradient for standard errors

# Examples
```julia
profile = Dict(:age => 45, :education => "college", :treated => true)
effect, grad = compute_profile_categorical_contrast(engine, profile, :treated, :response)
# Returns: effect of treated=true vs treated=false for a 45-year-old college graduate
```
"""
function compute_profile_categorical_contrast(
    engine::MarginsEngine{L}, 
    profile::Dict, 
    var::Symbol, 
    scale::Symbol,
    backend::Symbol
) where L
    # Get baseline level (extended function handles Bool variables automatically)
    baseline_level = _get_baseline_level(engine.model, var, engine.data_nt)
    current_level = profile[var]
    
    # Create profiles for current and baseline levels
    current_profile = profile
    baseline_profile = copy(profile)
    baseline_profile[var] = baseline_level
    
    # Compute predictions at both profiles using FormulaCompiler
    current_pred, current_grad = _profile_prediction_with_gradient(engine, current_profile, scale, backend)
    baseline_pred, baseline_grad = _profile_prediction_with_gradient(engine, baseline_profile, scale, backend)
    
    # Contrast effect and gradient
    effect = current_pred - baseline_pred
    gradient = current_grad .- baseline_grad
    
    return (effect, gradient)
end

"""
    compute_multiple_profile_contrasts(engine, profiles, var, scale, backend) -> (DataFrame, Matrix)

Compute row-specific categorical contrasts for multiple profiles efficiently.

# Arguments  
- `engine::MarginsEngine`: Pre-built computation engine
- `profiles::Vector{Dict}`: Vector of profiles to analyze
- `var::Symbol`: Categorical variable name
- `scale::Symbol`: `:link` or `:response` scale

# Returns
- `results::DataFrame`: Results with columns [:profile_id, :term, :estimate, :se] 
- `gradients::Matrix{Float64}`: Gradient matrix (n_profiles × n_params)

This is the main function used by `profile_margins()` for categorical variables.
"""
function compute_multiple_profile_contrasts(
    engine::MarginsEngine{L},
    profiles::Vector{Dict}, 
    var::Symbol,
    scale::Symbol,
    backend::Symbol
) where L
    n_profiles = length(profiles)
    n_params = length(engine.β)
    
    # Safely use η_buf for effects if large enough
    if length(engine.η_buf) >= n_profiles
        effects = view(engine.η_buf, 1:n_profiles)  # Reuse η_buf if large enough
    else
        @info "Buffer allocation fallback: η_buf too small for $n_profiles profiles (size=$(length(engine.η_buf)))"
        effects = Vector{Float64}(undef, n_profiles)  # Fall back to allocation
    end
    gradients = Matrix{Float64}(undef, n_profiles, n_params)
    terms = Vector{String}(undef, n_profiles)  # Keep this allocation as it's String, not Float64
    
    baseline_level = _get_baseline_level(engine.model, var, engine.data_nt)
    
    # Compute contrasts for each profile
    for (i, profile) in enumerate(profiles)
        effect, gradient = compute_profile_categorical_contrast(engine, profile, var, scale, backend)
        
        effects[i] = effect
        gradients[i, :] = gradient
        
        # Build descriptive term name
        current_level = profile[var]
        profile_desc = join(["$(k)=$(v)" for (k,v) in pairs(profile) if k != var], ", ")
        terms[i] = "$(current_level) vs $(baseline_level)"
    end
    
    # Safely use g_buf for SE computation if large enough
    if length(engine.g_buf) >= n_profiles
        ses = view(engine.g_buf, 1:n_profiles)  # Reuse g_buf if large enough
    else
        @info "Buffer allocation fallback: g_buf too small for $n_profiles profiles (size=$(length(engine.g_buf)))"
        ses = Vector{Float64}(undef, n_profiles)  # Fall back to allocation
    end
    for i in 1:n_profiles
        ses[i] = sqrt(dot(gradients[i, :], engine.Σ, gradients[i, :]))
    end
    
    # Build results DataFrame
    results = DataFrame(
        profile_id = 1:n_profiles,
        contrast = terms,
        estimate = effects,
        se = ses
    )
    
    return (results, gradients)
end

"""
    _profile_prediction_with_gradient(engine, profile, scale, backend) -> (prediction, gradient)

Helper function to compute both prediction and gradient at a profile using FormulaCompiler.

This function properly uses FormulaCompiler's API to avoid reinventing prediction logic.
"""
function _profile_prediction_with_gradient(
    engine::MarginsEngine{L}, 
    profile::Dict, 
    scale::Symbol, 
    backend::Symbol
) where L
    # Use scenario-based evaluation against the precompiled engine to preserve types/levels
    overrides = Dict{Symbol,Any}()
    for (k, v) in pairs(profile)
        overrides[k] = _normalize_override_value(v)
    end
    scenario = FormulaCompiler.create_scenario("profile", engine.data_nt, overrides)
    row = 1  # scenario fixes covariates; any row yields the same prediction

    # Prediction and gradient at this scenario (no recompilation)
    pred = _predict_with_scenario(engine.compiled, scenario, row, scale, engine.β, engine.link, engine.row_buf)
    _gradient_with_scenario!(engine.gβ_accumulator, engine.compiled, scenario, row, scale, engine.β, engine.link, engine.row_buf)
    return (pred, copy(engine.gβ_accumulator))
end

"""
    validate_contrast_specification(engine, var) -> Bool

Validate that a categorical variable can be used for contrast computation.

# Arguments
- `engine::MarginsEngine`: Computation engine  
- `var::Symbol`: Variable name to validate

# Returns  
- `true` if valid, throws `MarginsError` if invalid

# Validation checks
- Variable exists in the data
- Variable is categorical (not continuous)  
- Model has contrast information for the variable
- Variable has multiple levels (baseline + at least one other)
"""
function validate_contrast_specification(engine::MarginsEngine{L}, var::Symbol) where L
    # Check variable exists
    if !haskey(engine.data_nt, var)
        throw(MarginsError("Variable $var not found in data"))
    end
    
    # Check variable is categorical  
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, engine.data_nt)
    if var ∈ continuous_vars
        throw(MarginsError("Variable $var is continuous - contrasts only apply to categorical variables"))
    end
    
    # Check model has contrast information
    try
        _get_baseline_level(engine.model, var, engine.data_nt)
    catch e
        rethrow(e)  # _get_baseline_level already provides good error message
    end
    
    # Check variable has multiple levels
    col = getproperty(engine.data_nt, var)
    unique_levels = unique(col)
    if length(unique_levels) < 2
        throw(MarginsError("Variable $var has only one level - contrasts require at least 2 levels"))
    end
    
    return true
end

"""
    _compute_profile_pairwise_contrasts(engine, profile_dict, var, scale, backend) -> Vector{Tuple}

Compute all pairwise contrasts for a categorical variable at a specific profile.

Returns vector of (level1, level2, effect, gradient) tuples for all unique pairs.
"""
function _compute_profile_pairwise_contrasts(engine::MarginsEngine{L}, profile_dict::Dict, var::Symbol, scale::Symbol, backend::Symbol) where L
    # Get all levels for this categorical variable
    col = getproperty(engine.data_nt, var)
    levels = if col isa CategoricalArray
        CategoricalArrays.levels(col)
    elseif eltype(col) <: Bool
        [false, true]  # Standard Bool levels
    else
        unique(col)
    end
    
    # Generate all pairwise combinations (no baseline reference needed)
    contrast_pairs = [(level1, level2) for (i, level1) in enumerate(levels), (j, level2) in enumerate(levels) if i < j]
    
    results = []
    for (level1, level2) in contrast_pairs
        # Create profiles for both levels
        profile1 = copy(profile_dict)
        profile2 = copy(profile_dict)
        profile1[var] = level1
        profile2[var] = level2
        
        # Compute predictions at both profiles
        pred1, grad1 = _profile_prediction_with_gradient(engine, profile1, scale, backend)
        pred2, grad2 = _profile_prediction_with_gradient(engine, profile2, scale, backend)
        
        # Contrast effect and gradient
        effect = pred1 - pred2
        gradient = grad1 .- grad2
        
        push!(results, (level1, level2, effect, gradient))
    end
    
    return results
end
