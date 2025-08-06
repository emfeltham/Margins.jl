# categorical_variable_effects.jl
# Updated to use workspace scenarios

"""
    compute_baseline_factor_contrasts!(
    effect_estimates, standard_errors, gradient_vectors,
    focal_variable, workspace, coefficient_vector, covariance_matrix,
    inverse_link_function, first_derivative; variable_overrides
)

Compute baseline contrasts for categorical variable using workspace scenarios.
"""
function compute_baseline_factor_contrasts!(
    effect_estimates::Dict{Tuple,Float64}, 
    standard_errors::Dict{Tuple,Float64}, 
    gradient_vectors::Dict{Tuple,Vector{Float64}}, 
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector, 
    covariance_matrix::AbstractMatrix,
    inverse_link_function::Function, 
    first_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}()
)
    # Extract factor levels using workspace data
    factor_levels = extract_factor_levels(workspace.column_data[focal_variable])
    
    if length(factor_levels) < 2
        @warn "Variable $focal_variable has fewer than 2 levels, skipping contrasts"
        return
    end
    
    baseline_level = factor_levels[1]
    
    # Compute contrasts against baseline using workspace scenarios
    for comparison_level in factor_levels[2:end]
        effect, se, gradient = compute_factor_level_contrast(
            focal_variable, baseline_level, comparison_level, 
            workspace, coefficient_vector, covariance_matrix,
            inverse_link_function, first_derivative; 
            variable_overrides=variable_overrides
        )
        
        contrast_key = (baseline_level, comparison_level)
        effect_estimates[contrast_key] = effect
        standard_errors[contrast_key] = se
        gradient_vectors[contrast_key] = copy(gradient)
    end
end

"""
    compute_all_pairs_factor_contrasts!(effect_estimates, standard_errors, gradient_vectors,
                                       focal_variable, workspace, coefficient_vector, covariance_matrix,
                                       inverse_link_function, first_derivative; variable_overrides)

Compute all pairwise contrasts for categorical variable using workspace scenarios.
"""
function compute_all_pairs_factor_contrasts!(
    effect_estimates::Dict{Tuple,Float64}, 
    standard_errors::Dict{Tuple,Float64}, 
    gradient_vectors::Dict{Tuple,Vector{Float64}}, 
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector, 
    covariance_matrix::AbstractMatrix,
    inverse_link_function::Function, 
    first_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}()
)
    # Extract factor levels using workspace data
    factor_levels = extract_factor_levels(workspace.column_data[focal_variable])
    
    if length(factor_levels) < 2
        @warn "Variable $focal_variable has fewer than 2 levels, skipping contrasts"
        return
    end
    
    # Compute all pairwise contrasts using workspace scenarios
    for level_i_index in 1:length(factor_levels)-1
        for level_j_index in level_i_index+1:length(factor_levels)
            level_i, level_j = factor_levels[level_i_index], factor_levels[level_j_index]
            
            effect, se, gradient = compute_factor_level_contrast(
                focal_variable, level_i, level_j,
                workspace, coefficient_vector, covariance_matrix,
                inverse_link_function, first_derivative;
                variable_overrides=variable_overrides
            )
            
            contrast_key = (level_i, level_j)
            effect_estimates[contrast_key] = effect
            standard_errors[contrast_key] = se
            gradient_vectors[contrast_key] = copy(gradient)
        end
    end
end

"""
    compute_factor_level_contrast(focal_variable, reference_level, comparison_level,
                                 workspace, coefficient_vector, covariance_matrix,
                                 inverse_link_function, first_derivative; variable_overrides)

Core function: Compute contrast between two levels using workspace scenarios consistently.
"""
function compute_factor_level_contrast(
    focal_variable::Symbol, 
    reference_level, 
    comparison_level, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector, 
    covariance_matrix::AbstractMatrix,
    inverse_link_function::Function, 
    first_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}()
)
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    
    # Use workspace buffers for accumulators
    fill!(workspace.gradient_accumulator, 0.0)  # Reference level gradients
    fill!(workspace.computation_buffer, 0.0)    # Comparison level gradients
    
    # Initialize prediction accumulators
    reference_prediction_sum = 0.0
    comparison_prediction_sum = 0.0
    
    # STEP 1: Compute average prediction at reference level using workspace scenarios
    reference_overrides = merge(variable_overrides, Dict(focal_variable => reference_level))
    for observation_index in 1:observation_count
        # Use workspace scenario system instead of custom override logic
        evaluate_model_row!(workspace, observation_index; variable_overrides=reference_overrides)
        linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
        predicted_value = inverse_link_function(linear_predictor)
        link_derivative = first_derivative(linear_predictor)
        
        # Accumulate prediction and gradient
        reference_prediction_sum += predicted_value
        for param_index in 1:parameter_count
            workspace.gradient_accumulator[param_index] += link_derivative * workspace.model_row_buffer[param_index] / observation_count
        end
    end
    
    # STEP 2: Compute average prediction at comparison level using workspace scenarios
    comparison_overrides = merge(variable_overrides, Dict(focal_variable => comparison_level))
    for observation_index in 1:observation_count
        # Use workspace scenario system instead of custom override logic
        evaluate_model_row!(workspace, observation_index; variable_overrides=comparison_overrides)
        linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
        predicted_value = inverse_link_function(linear_predictor)
        link_derivative = first_derivative(linear_predictor)
        
        # Accumulate prediction and gradient
        comparison_prediction_sum += predicted_value
        for param_index in 1:parameter_count
            workspace.computation_buffer[param_index] += link_derivative * workspace.model_row_buffer[param_index] / observation_count
        end
    end
    
    # STEP 3: Compute final marginal effect and gradient
    reference_mean = reference_prediction_sum / observation_count
    comparison_mean = comparison_prediction_sum / observation_count
    marginal_effect = comparison_mean - reference_mean
    
    # Gradient = ∇[mean(comparison)] - ∇[mean(reference)]
    for param_index in 1:parameter_count
        workspace.gradient_accumulator[param_index] = workspace.computation_buffer[param_index] - workspace.gradient_accumulator[param_index]
    end
    
    # Compute standard error using delta method
    standard_error = sqrt(dot(workspace.gradient_accumulator, covariance_matrix * workspace.gradient_accumulator))
    
    return marginal_effect, standard_error, copy(workspace.gradient_accumulator)
end

"""
    extract_factor_levels(factor_column) -> Vector

Extract ordered list of factor levels from any column type (consistent with workspace data).
"""
function extract_factor_levels(factor_column)
    if factor_column isa CategoricalArray
        return levels(factor_column)
    elseif eltype(factor_column) <: Bool
        return [false, true]  # Canonical Boolean ordering
    elseif eltype(factor_column) <: CategoricalValue
        return levels(categorical(factor_column))
    else
        # For other types, use sorted unique values
        return sort(unique(factor_column))
    end
end
