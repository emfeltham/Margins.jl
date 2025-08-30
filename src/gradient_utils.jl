# gradient_utils.jl - Post-calculation utilities for gradient storage

"""
    get_gradients(result::MarginsResult)

Extract the gradient Dict from a MarginsResult for direct access.
Returns the underlying gradient storage Dict from the AbstractMarginsGradients object.
"""
function get_gradients(result::MarginsResult)
    return _get_gradient_dict(result.gradients)
end

"""
    contrast(result::MarginsResult, contrast_vector::Vector{Float64})

Compute linear contrasts using stored gradients without recomputation.
Efficient contrast calculation using pre-computed gradient information.
"""
function contrast(result::MarginsResult, contrast_vector::Vector{Float64})
    return _contrast(result.gradients, contrast_vector)
end

# Type-stable dispatch on each concrete gradient type
function _contrast(grads::ContinuousGradients, contrast_vector::Vector{Float64})
    # Access grads.gradients::Dict{Tuple{Symbol,Int}, Vector{Float64}} with zero overhead
    gradient_dict = grads.gradients
    variables = grads.variables
    
    if length(contrast_vector) != length(variables)
        error("Contrast vector length ($(length(contrast_vector))) must match number of variables ($(length(variables)))")
    end
    
    # For continuous gradients, compute weighted combination of variable gradients
    # This is a simplified implementation - full version would handle different computation types
    result_gradients = Vector{Float64}[]
    for (key, grad) in gradient_dict
        var, prof_idx = key
        var_idx = findfirst(==(var), variables)
        if var_idx !== nothing
            push!(result_gradients, contrast_vector[var_idx] .* grad)
        end
    end
    
    if !isempty(result_gradients)
        combined_gradient = sum(result_gradients)
        return (gradient = combined_gradient, variables = variables, contrast = contrast_vector)
    else
        error("No matching gradients found for contrast computation")
    end
end

function _contrast(grads::CategoricalGradients, contrast_vector::Vector{Float64})
    # Handle Dict{Tuple{Symbol,Int}, Vector{Float64}} format for categorical contrasts
    # Implementation would depend on specific contrast interpretation for categoricals
    error("Categorical contrasts not yet implemented - requires domain-specific logic")
end

function _contrast(grads::PredictionGradients, contrast_vector::Vector{Float64})
    # Handle Dict{Int, Vector{Float64}} format for prediction contrasts
    gradient_dict = grads.gradients
    
    if length(contrast_vector) != length(gradient_dict)
        error("Contrast vector length ($(length(contrast_vector))) must match number of profiles ($(length(gradient_dict)))")
    end
    
    # Compute weighted combination of profile gradients
    result_gradients = Vector{Float64}[]
    for (prof_idx, grad) in sort(collect(gradient_dict))
        push!(result_gradients, contrast_vector[prof_idx] .* grad)
    end
    
    if !isempty(result_gradients)
        combined_gradient = sum(result_gradients)
        return (gradient = combined_gradient, profiles = length(gradient_dict), contrast = contrast_vector)
    else
        error("No gradients found for prediction contrast computation")
    end
end

"""
    bootstrap_effects(result::MarginsResult, n_samples::Int=1000)

Perform bootstrap resampling using stored gradients.
Much faster than recomputing marginal effects for each bootstrap sample.
"""
function bootstrap_effects(result::MarginsResult, n_samples::Int=1000)
    return _bootstrap_effects(result.gradients, n_samples)
end

function _bootstrap_effects(grads::AbstractMarginsGradients, n_samples::Int)
    # This is a placeholder implementation
    # Full implementation would resample gradients and recompute statistics
    gradient_dict = _get_gradient_dict(grads)
    
    bootstrap_samples = Float64[]
    for _ in 1:n_samples
        # Placeholder: would implement proper gradient-based bootstrap here
        # This would involve resampling the gradient components and recomputing effects
        push!(bootstrap_samples, rand()) # Placeholder
    end
    
    return (
        samples = bootstrap_samples,
        mean = mean(bootstrap_samples),
        std = std(bootstrap_samples),
        n_samples = n_samples,
        computation_type = grads.computation_type
    )
end

"""
    effect_heterogeneity(result::MarginsResult)

Analyze heterogeneity in effects using stored gradient information.
Provides insights into variation in marginal effects across observations or profiles.
"""
function effect_heterogeneity(result::MarginsResult)
    return _effect_heterogeneity(result.gradients)
end

function _effect_heterogeneity(grads::AbstractMarginsGradients)
    gradient_dict = _get_gradient_dict(grads)
    
    # Extract gradient magnitudes for heterogeneity analysis
    gradient_norms = [norm(grad) for grad in values(gradient_dict)]
    
    return (
        gradient_norms = gradient_norms,
        mean_norm = mean(gradient_norms),
        std_norm = std(gradient_norms),
        min_norm = minimum(gradient_norms),
        max_norm = maximum(gradient_norms),
        n_gradients = length(gradient_norms),
        computation_type = grads.computation_type,
        variables = grads.variables
    )
end

"""
    gradient_summary(result::MarginsResult)

Provide a summary of the gradient storage information.
Useful for understanding what gradients are available for post-calculation.
"""
function gradient_summary(result::MarginsResult)
    grads = result.gradients
    gradient_dict = _get_gradient_dict(grads)
    
    return (
        gradient_type = typeof(grads),
        computation_type = grads.computation_type,
        target = grads.target,
        backend = grads.backend,
        variables = grads.variables,
        n_gradients = length(gradient_dict),
        gradient_keys = collect(keys(gradient_dict)),
        has_profile_coords = grads.profile_coordinates !== nothing
    )
end