# core.jl
# Merged core implementation

"""
    margins(
    model, data, variables; 
    type = :dydx,
    representative_values = nothing,
    contrasts = :pairwise
) -> MarginalEffects

Compute marginal effects or predictions using FormulaCompiler.jl analytical derivatives.
"""
function margins(
    model, data, variables; 
    type::Symbol = :dydx,
    representative_values::Union{Nothing,AbstractDict} = nothing,
    contrasts::Symbol = :pairwise
)
    
    # Input validation
    _validate_margins_arguments(type, contrasts, variables, data)
    
    # Standardize variables as vector
    focal_variables = variables isa Symbol ? [variables] : collect(variables)
    
    if !((type == :dydx) | (type == :prediction))
        throw(ArgumentError("Unknown type: $type. Must be :dydx or :prediction"))
    end
    
    # Delegate to main implementation
    compute_marginal_effects(model, focal_variables, data, representative_values, contrasts, type)
end

margins(model, data, variable::Symbol; kwargs...) = margins(model, data, [variable]; kwargs...)

"""
    compute_marginal_effects(model, focal_variables, data, representative_values, contrasts, effect_type)

Main implementation function for marginal effects computation using analytical derivatives.
"""
function compute_marginal_effects(model, focal_variables, data, representative_values, contrasts, effect_type)
    
    # Convert contrasts parameter
    factor_contrasts = if contrasts == :pairwise
        :all_pairs
    elseif contrasts == :baseline
        :baseline_contrasts
    else
        throw(ArgumentError("Unknown contrasts: $contrasts"))
    end
    
    # Convert representative_values
    rep_vals = representative_values === nothing ? Dict{Symbol,Vector{Float64}}() : representative_values
    
    # Extract model information
    coefficient_vector = coef(model)
    covariance_matrix = vcov(model)
    cholesky_covariance = cholesky(covariance_matrix)
    inverse_link_function, first_derivative, second_derivative = extract_link_functions(model)
    
    # Convert data and classify variables
    column_data = Tables.columntable(data)
    continuous_variables = filter(var -> is_continuous_variable(var, column_data), focal_variables)
    categorical_variables = setdiff(focal_variables, continuous_variables)
    
    # Create workspace with analytical derivatives
    workspace = MarginalEffectsWorkspace(model, data, continuous_variables)
    
    # Initialize result storage
    effect_estimates = Dict{Symbol,Any}()
    standard_errors = Dict{Symbol,Any}()
    gradient_vectors = Dict{Symbol,Any}()
    
    # Dispatch to appropriate computation method
    if effect_type == :predictions
        if representative_values === nothing
            throw(ArgumentError("representative_values required for type = :prediction"))
        end
        _compute_average_predictions!(
            effect_estimates, standard_errors, gradient_vectors,
            focal_variables, rep_vals, workspace,
            coefficient_vector, cholesky_covariance, first_derivative
        )
    elseif isempty(rep_vals)
        # Standard marginal effects using analytical derivatives
        _compute_standard_marginal_effects!(
            effect_estimates, standard_errors, gradient_vectors,
            continuous_variables, categorical_variables, factor_contrasts,
            workspace, coefficient_vector, covariance_matrix, cholesky_covariance,
            inverse_link_function, first_derivative, second_derivative
        )
    else
        # Representative value effects using analytical derivatives
        _compute_representative_value_effects!(
            effect_estimates, standard_errors, gradient_vectors,
            focal_variables, rep_vals, factor_contrasts,
            workspace, coefficient_vector, cholesky_covariance,
            inverse_link_function, first_derivative, second_derivative
        )
    end
    
    return MarginalEffectsResult{effect_type}(
        focal_variables,
        rep_vals,
        effect_estimates,
        standard_errors,
        gradient_vectors,
        nrow(data),
        dof_residual(model),
        string(family(model).dist),
        string(family(model).link)
    )
end

"""
    _compute_standard_marginal_effects!(...)

Compute standard marginal effects using analytical derivatives for continuous variables.
"""
function _compute_standard_marginal_effects!(
    effect_estimates, standard_errors, gradient_vectors,
    continuous_variables, categorical_variables, factor_contrasts,
    workspace, coefficient_vector, covariance_matrix, cholesky_covariance,
    inverse_link_function, first_derivative, second_derivative
)
    # Compute continuous variable effects using analytical derivatives
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
    
    # Compute categorical variable effects (using scenarios)
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

"""
    _compute_representative_value_effects!(...)

Compute marginal effects at representative values using analytical derivatives and scenarios.
"""
function _compute_representative_value_effects!(
    effect_estimates, standard_errors, gradient_vectors,
    focal_variables, representative_values, factor_contrasts,
    workspace, coefficient_vector, cholesky_covariance,
    inverse_link_function, first_derivative, second_derivative
)
    # Extract representative variable information
    representative_variables = collect(keys(representative_values))
    value_combinations = create_representative_value_grid(representative_values)
    covariance_matrix = Matrix(cholesky_covariance)
    
    # Initialize storage for each focal variable
    for variable in focal_variables
        effect_estimates[variable] = Dict{Tuple,Float64}()
        standard_errors[variable] = Dict{Tuple,Float64}()
        gradient_vectors[variable] = Dict{Tuple,Vector{Float64}}()
    end
    
    # Compute effects for each representative value combination
    for value_combination in value_combinations
        # Create override dictionary for this combination
        variable_overrides = Dict{Symbol,Any}(
            var => val for (var, val) in zip(representative_variables, value_combination)
        )
        
        # Compute effects for each focal variable
        for focal_variable in focal_variables
            if is_continuous_variable(focal_variable, workspace.column_data)
                # Continuous focal variable using analytical derivatives
                effect, se, gradient = compute_single_continuous_effect(
                    focal_variable, workspace, coefficient_vector, cholesky_covariance,
                    first_derivative, second_derivative; variable_overrides=variable_overrides
                )
                
                effect_estimates[focal_variable][value_combination] = effect
                standard_errors[focal_variable][value_combination] = se
                gradient_vectors[focal_variable][value_combination] = gradient
                
            else
                # Categorical focal variable using scenarios
                categorical_effects = Dict{Tuple,Float64}()
                categorical_ses = Dict{Tuple,Float64}()
                categorical_grads = Dict{Tuple,Vector{Float64}}()
                
                if factor_contrasts == :baseline_contrasts
                    compute_baseline_factor_contrasts!(
                        categorical_effects, categorical_ses, categorical_grads,
                        focal_variable, workspace, coefficient_vector, covariance_matrix,
                        inverse_link_function, first_derivative; variable_overrides=variable_overrides
                    )
                else
                    compute_all_pairs_factor_contrasts!(
                        categorical_effects, categorical_ses, categorical_grads,
                        focal_variable, workspace, coefficient_vector, covariance_matrix,
                        inverse_link_function, first_derivative; variable_overrides=variable_overrides
                    )
                end
                
                # Store results with extended keys
                for (level_pair, effect_value) in categorical_effects
                    extended_key = (value_combination..., level_pair...)
                    effect_estimates[focal_variable][extended_key] = effect_value
                    standard_errors[focal_variable][extended_key] = categorical_ses[level_pair]
                    gradient_vectors[focal_variable][extended_key] = categorical_grads[level_pair]
                end
            end
        end
    end
end

function _compute_average_predictions!(
    effect_estimates, standard_errors, gradient_vectors,
    focal_variables, representative_values, workspace,
    coefficient_vector, cholesky_covariance, first_derivative
)
    if isempty(representative_values)
        # Average prediction across all observations (no scenarios)
        observation_count = get_observation_count(workspace)
        parameter_count = get_parameter_count(workspace)
        
        prediction_sum = 0.0
        fill!(workspace.gradient_accumulator, 0.0)
        
        for observation_index in 1:observation_count
            evaluate_model_row!(workspace, observation_index)
            linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
            predicted_value = first_derivative(linear_predictor)  # For now, use first derivative as inverse link
            prediction_sum += predicted_value
            
            link_derivative = first_derivative(linear_predictor)
            for param_index in 1:parameter_count
                workspace.gradient_accumulator[param_index] += link_derivative * workspace.model_row_buffer[param_index] / observation_count
            end
        end
        
        average_prediction = prediction_sum / observation_count
        standard_error = compute_standard_error_from_gradient(workspace, cholesky_covariance)
        
        for variable in focal_variables
            effect_estimates[variable] = average_prediction
            standard_errors[variable] = standard_error
            gradient_vectors[variable] = copy(workspace.gradient_accumulator)
        end
    else
        # Predictions at representative values using scenario system
        validate_representative_values(representative_values, workspace.column_data)
        value_combinations = create_representative_value_grid(representative_values)
        
        for variable in focal_variables
            effect_estimates[variable] = Dict{Tuple,Float64}()
            standard_errors[variable] = Dict{Tuple,Float64}()
            gradient_vectors[variable] = Dict{Tuple,Vector{Float64}}()
            
            for value_combination in value_combinations
                variable_overrides = Dict{Symbol,Any}(
                    var => val for (var, val) in zip(collect(keys(representative_values)), value_combination)
                )
                
                # Compute prediction at this representative value combination using scenarios
                observation_count = get_observation_count(workspace)
                parameter_count = get_parameter_count(workspace)
                
                prediction_sum = 0.0
                fill!(workspace.gradient_accumulator, 0.0)
                
                for observation_index in 1:observation_count
                    evaluate_model_row!(workspace, observation_index; variable_overrides=variable_overrides)
                    linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
                    predicted_value = first_derivative(linear_predictor)  # Simplified - would use proper inverse link
                    prediction_sum += predicted_value
                    
                    link_derivative = first_derivative(linear_predictor)
                    for param_index in 1:parameter_count
                        workspace.gradient_accumulator[param_index] += link_derivative * workspace.model_row_buffer[param_index] / observation_count
                    end
                end
                
                average_prediction = prediction_sum / observation_count
                standard_error = compute_standard_error_from_gradient(workspace, cholesky_covariance)
                
                effect_estimates[variable][value_combination] = average_prediction
                standard_errors[variable][value_combination] = standard_error
                gradient_vectors[variable][value_combination] = copy(workspace.gradient_accumulator)
            end
        end
    end
end

# Helper functions
function _validate_margins_arguments(type, contrasts, variables, data)
    if type ∉ (:dydx, :prediction)
        throw(ArgumentError("type must be :dydx or :prediction, got $type"))
    end
    
    if contrasts ∉ (:pairwise, :baseline)
        throw(ArgumentError("contrasts must be :pairwise or :baseline, got $contrasts"))
    end
    
    variable_list = variables isa Symbol ? [variables] : collect(variables)
    if isempty(variable_list)
        throw(ArgumentError("variables cannot be empty"))
    end
    
    if data isa DataFrame
        data_columns = Symbol.(names(data))
        for variable in variable_list
            if variable ∉ data_columns
                throw(ArgumentError("Variable $variable not found in data columns: $data_columns"))
            end
        end
    end
    
    return true
end

function create_representative_value_grid(representative_values::AbstractDict{Symbol,<:AbstractVector})
    variables = collect(keys(representative_values))
    value_vectors = [collect(representative_values[var]) for var in variables]
    return collect(Iterators.product(value_vectors...))
end
