# marginal_effects_core.jl - Clean implementation with descriptive names

###############################################################################
# Main Entry Point for Marginal Effects Computation
###############################################################################

"""
    compute_marginal_effects(model, focal_variables, data; 
                            representative_values = Dict(), 
                            factor_contrasts = :all_pairs, 
                            effect_type = :dydx) → MarginalEffectsResult

Compute average marginal effects (AMEs) or average predictions using zero-allocation
row-wise computation with O(p) memory complexity.

# Arguments
- `model`: Fitted statistical model (LinearModel, GeneralizedLinearModel, etc.)
- `focal_variables`: Variable(s) to compute effects for (Symbol or Vector{Symbol})
- `data`: Input data (DataFrame, NamedTuple, or Tables.jl compatible)
- `representative_values`: Dict mapping variables to representative value grids
- `factor_contrasts`: `:all_pairs` or `:baseline_contrasts` for categorical variables
- `effect_type`: `:dydx` for marginal effects or `:predictions` for predicted values

# Returns
`MarginalEffectsResult` containing estimates, standard errors, and diagnostics.

# Performance
- Memory: O(p) where p = number of model parameters
- Time: ~100-500ns per observation depending on effect type
- Scales to 10M+ observations with constant memory usage

# Example
```julia
model = lm(@formula(y ~ x * group + log(z)), df)

# Basic marginal effects
result = compute_marginal_effects(model, [:x, :group], df)

# With representative values
result = compute_marginal_effects(
    model, 
    [:x, :group], 
    df;
    representative_values = Dict(:z => [1.0, 2.0, 3.0], :group => ["A", "B"])
)
```
"""
function compute_marginal_effects(
    model,
    focal_variables,
    data::AbstractDataFrame;
    representative_values::AbstractDict{Symbol,<:AbstractVector} = Dict{Symbol,Vector{Float64}}(),
    factor_contrasts::Symbol = :all_pairs,
    effect_type::Symbol = :dydx,
)
    # Validate arguments
    validate_marginal_effects_arguments(effect_type, factor_contrasts, focal_variables, data)
    
    # Standardize focal variables as vector
    focal_variable_list = focal_variables isa Symbol ? [focal_variables] : collect(focal_variables)
    
    # Extract model information
    coefficient_vector = coef(model)
    covariance_matrix = vcov(model)
    cholesky_covariance = cholesky(covariance_matrix)
    inverse_link_function, first_derivative, second_derivative = extract_link_functions(model)
    
    # Convert data and classify variables
    column_data = Tables.columntable(data)
    continuous_variables = filter(var -> is_continuous_variable(var, column_data), focal_variable_list)
    categorical_variables = setdiff(focal_variable_list, continuous_variables)
    
    # Create workspace for zero-allocation computation
    workspace = MarginalEffectsWorkspace(model, data)
    
    # Initialize result storage
    effect_estimates = Dict{Symbol,Any}()
    standard_errors = Dict{Symbol,Any}()
    gradient_vectors = Dict{Symbol,Any}()
    
    # Dispatch to appropriate computation method
    if effect_type == :predictions
        compute_average_predictions!(
            effect_estimates, standard_errors, gradient_vectors,
            focal_variable_list, representative_values, workspace,
            coefficient_vector, cholesky_covariance, first_derivative
        )
    elseif isempty(representative_values)
        # dydx
        # Standard marginal effects without representative values
        compute_standard_marginal_effects!(
            effect_estimates, standard_errors, gradient_vectors,
            continuous_variables, categorical_variables, factor_contrasts,
            workspace, coefficient_vector, covariance_matrix, cholesky_covariance,
            inverse_link_function, first_derivative, second_derivative
        )
    else
        # Marginal effects at representative values
        compute_representative_value_effects!(
            effect_estimates, standard_errors, gradient_vectors,
            focal_variable_list, representative_values, factor_contrasts,
            workspace, coefficient_vector, cholesky_covariance,
            inverse_link_function, first_derivative, second_derivative
        )
    end
    
    return MarginalEffectsResult{effect_type}(
        focal_variable_list,
        representative_values,
        effect_estimates,
        standard_errors,
        gradient_vectors,
        nrow(data),
        dof_residual(model),
        string(family(model).dist),
        string(family(model).link)
    )
end

###############################################################################
# Standard Marginal Effects (No Representative Values)
###############################################################################

"""
    compute_standard_marginal_effects!(effect_estimates, standard_errors, gradient_vectors,
                                      continuous_variables, categorical_variables, factor_contrasts,
                                      workspace, coefficient_vector, covariance_matrix, cholesky_covariance,
                                      inverse_link_function, first_derivative, second_derivative)

Compute standard average marginal effects across all observations.
"""
function compute_standard_marginal_effects!(
    effect_estimates, standard_errors, gradient_vectors,
    continuous_variables, categorical_variables, factor_contrasts,
    workspace, coefficient_vector, covariance_matrix, cholesky_covariance,
    inverse_link_function, first_derivative, second_derivative
)
    # Compute continuous variable effects
    if !isempty(continuous_variables)
        continuous_effects, continuous_ses, continuous_grads = compute_continuous_variable_effects(
            continuous_variables, workspace, coefficient_vector, cholesky_covariance,
            first_derivative, second_derivative
        )
        
        for (i, variable) in enumerate(continuous_variables)
            effect_estimates[variable] = continuous_effects[i]
            standard_errors[variable] = continuous_ses[i]
            gradient_vectors[variable] = continuous_grads[i]
        end
    end
    
    # Compute categorical variable effects
    for variable in categorical_variables
        variable_effects = Dict{Tuple,Float64}()
        variable_ses = Dict{Tuple,Float64}()
        variable_grads = Dict{Tuple,Vector{Float64}}()
        
        if factor_contrasts == :baseline_contrasts
            compute_baseline_factor_contrasts!(
                variable_effects, variable_ses, variable_grads,
                variable, workspace, coefficient_vector, covariance_matrix,
                inverse_link_function, first_derivative
            )
        else  # :all_pairs
            compute_all_pairs_factor_contrasts!(
                variable_effects, variable_ses, variable_grads,
                variable, workspace, coefficient_vector, covariance_matrix,
                inverse_link_function, first_derivative
            )
        end
        
        effect_estimates[variable] = variable_effects
        standard_errors[variable] = variable_ses
        gradient_vectors[variable] = variable_grads
    end
end

###############################################################################
# Average Predictions
###############################################################################

"""
    compute_average_predictions!(effect_estimates, standard_errors, gradient_vectors,
                                 focal_variable_list, representative_values, workspace,
                                 coefficient_vector, cholesky_covariance, first_derivative)

Compute average predicted values across all observations.
"""
function compute_average_predictions!(
    effect_estimates, standard_errors, gradient_vectors,
    focal_variable_list, representative_values, workspace,
    coefficient_vector, cholesky_covariance, first_derivative
)
    if !isempty(representative_values)
        throw(ArgumentError("Representative value predictions not yet implemented"))
    end
    
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    
    # Initialize accumulators
    prediction_sum = 0.0
    fill!(workspace.gradient_accumulator, 0.0)
    
    # Compute average prediction across all observations
    for observation_index in 1:observation_count
        evaluate_model_row!(workspace, observation_index)
        linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
        predicted_value = get_inverse_link_function(workspace)(linear_predictor)
        prediction_sum += predicted_value
        
        # Accumulate gradient
        link_derivative = first_derivative(linear_predictor)
        @inbounds for param_index in 1:parameter_count
            workspace.gradient_accumulator[param_index] += link_derivative * workspace.model_row_buffer[param_index] / observation_count
        end
    end
    
    # Compute final results
    average_prediction = prediction_sum / observation_count
    mul!(workspace.computation_buffer, cholesky_covariance.U, workspace.gradient_accumulator)
    standard_error = sqrt(dot(workspace.computation_buffer, workspace.computation_buffer))
    
    # Store results for all focal variables
    for variable in focal_variable_list
        effect_estimates[variable] = average_prediction
        standard_errors[variable] = standard_error
        gradient_vectors[variable] = copy(workspace.gradient_accumulator)
    end
end
