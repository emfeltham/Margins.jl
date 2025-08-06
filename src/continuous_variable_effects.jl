# continuous_variable_effects.jl
# Using analytical derivatives

"""
    compute_continuous_variable_effects(
    focal_variables, workspace, coefficient_vector, 
    cholesky_covariance, first_derivative, second_derivative;
    variable_overrides
)

Compute marginal effects for continuous variables using FormulaCompiler.jl analytical derivatives.
"""
function compute_continuous_variable_effects(
    focal_variables::Vector{Symbol}, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    cholesky_covariance::LinearAlgebra.Cholesky,
    first_derivative::Function,
    second_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}()
)
    variable_count = length(focal_variables)
    effects = Vector{Float64}(undef, variable_count)
    standard_errors = Vector{Float64}(undef, variable_count)
    gradients = Vector{Vector{Float64}}(undef, variable_count)
    
    for (variable_index, focal_variable) in enumerate(focal_variables)
        effect, se, gradient = compute_single_continuous_effect(
            focal_variable, workspace, coefficient_vector, cholesky_covariance,
            first_derivative, second_derivative; variable_overrides=variable_overrides
        )
        
        effects[variable_index] = effect
        standard_errors[variable_index] = se
        gradients[variable_index] = gradient
    end
    
    return effects, standard_errors, gradients
end

"""
    compute_single_continuous_effect(focal_variable, workspace, coefficient_vector,
                                    cholesky_covariance, first_derivative, second_derivative;
                                    variable_overrides)

Compute marginal effect for a single continuous variable using analytical derivatives.
"""
function compute_single_continuous_effect(
    focal_variable::Symbol, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector, 
    cholesky_covariance::LinearAlgebra.Cholesky,
    first_derivative::Function, 
    second_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}()
)
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    
    # Initialize accumulators
    marginal_effect_sum = 0.0
    fill!(workspace.gradient_accumulator, 0.0)
    valid_observations = 0
    
    # Core computation loop using analytical derivatives
    for observation_index in 1:observation_count
        try
            # Evaluate model matrix row
            evaluate_model_row!(workspace, observation_index; variable_overrides=variable_overrides)
            linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
            
            # Evaluate analytical derivative (key performance improvement)
            evaluate_model_derivative!(workspace, observation_index, focal_variable; variable_overrides=variable_overrides)
            predictor_derivative = dot(workspace.derivative_buffer, coefficient_vector)
            
            # Compute marginal effect for this observation
            if all_finite_and_reasonable(linear_predictor, predictor_derivative)
                link_first_derivative = first_derivative(linear_predictor)
                observation_marginal_effect = link_first_derivative * predictor_derivative
                
                if is_finite_and_reasonable(observation_marginal_effect)
                    marginal_effect_sum += observation_marginal_effect
                    valid_observations += 1
                    
                    # Accumulate gradient using product rule
                    link_second_derivative = second_derivative(linear_predictor)
                    
                    for param_index in 1:parameter_count
                        second_order_term = link_second_derivative * workspace.model_row_buffer[param_index] * predictor_derivative
                        first_order_term = link_first_derivative * workspace.derivative_buffer[param_index]
                        workspace.gradient_accumulator[param_index] += (second_order_term + first_order_term) / observation_count
                    end
                end
            end
            
        catch e
            @warn "Error computing marginal effect for observation $observation_index: $e" maxlog=5
            continue
        end
    end
    
    if valid_observations == 0
        @warn "No valid observations for variable $focal_variable"
        return NaN, NaN, fill(NaN, parameter_count)
    end
    
    # Final computations
    average_marginal_effect = marginal_effect_sum / observation_count
    standard_error = compute_standard_error_from_gradient(workspace, cholesky_covariance)
    
    return average_marginal_effect, standard_error, copy(workspace.gradient_accumulator)
end
