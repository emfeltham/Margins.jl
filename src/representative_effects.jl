###############################################################################
# Representative Values Effects
###############################################################################

"""
    compute_representative_value_effects!(effect_estimates, standard_errors, gradient_vectors,
                                         focal_variable_list, representative_values, factor_contrasts,
                                         workspace, coefficient_vector, cholesky_covariance,
                                         inverse_link_function, first_derivative, second_derivative)

Compute marginal effects at representative value combinations.
"""
function compute_representative_value_effects!(
    effect_estimates, standard_errors, gradient_vectors,
    focal_variable_list, representative_values, factor_contrasts,
    workspace, coefficient_vector, cholesky_covariance,
    inverse_link_function, first_derivative, second_derivative
)
    # Extract representative variable information
    representative_variables = collect(keys(representative_values))
    value_combinations = create_representative_value_grid(representative_values)
    covariance_matrix = vcov_from_cholesky(cholesky_covariance)
    
    # Initialize storage for each focal variable
    for variable in focal_variable_list
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
        for focal_variable in focal_variable_list
            if is_continuous_variable(focal_variable, workspace.column_data)
                # Continuous focal variable
                effect, se, gradient = compute_single_continuous_effect(
                    focal_variable, workspace, coefficient_vector, cholesky_covariance,
                    first_derivative, second_derivative; variable_overrides=variable_overrides
                )
                
                effect_estimates[focal_variable][value_combination] = effect
                standard_errors[focal_variable][value_combination] = se
                gradient_vectors[focal_variable][value_combination] = gradient
                
            else
                # Categorical focal variable
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
                    if is_boolean_variable(focal_variable, workspace.column_data)
                        # Boolean variable: use combination as key
                        effect_estimates[focal_variable][value_combination] = effect_value
                        standard_errors[focal_variable][value_combination] = categorical_ses[level_pair]
                        gradient_vectors[focal_variable][value_combination] = categorical_grads[level_pair]
                    else
                        # Multi-level categorical: extend key with level pair
                        extended_key = (value_combination..., level_pair...)
                        effect_estimates[focal_variable][extended_key] = effect_value
                        standard_errors[focal_variable][extended_key] = categorical_ses[level_pair]
                        gradient_vectors[focal_variable][extended_key] = categorical_grads[level_pair]
                    end
                end
            end
        end
    end
end

###############################################################################
# Utility Functions
###############################################################################

"""
    validate_marginal_effects_arguments(effect_type, factor_contrasts, focal_variables, data)

Validate input arguments for marginal effects computation.
"""
function validate_marginal_effects_arguments(effect_type, factor_contrasts, focal_variables, data)
    # Validate effect type
    if effect_type ∉ (:dydx, :predictions)
        throw(ArgumentError("effect_type must be :dydx or :predictions, got $effect_type"))
    end
    
    # Validate factor contrasts
    if factor_contrasts ∉ (:all_pairs, :baseline_contrasts)
        throw(ArgumentError("factor_contrasts must be :all_pairs or :baseline_contrasts, got $factor_contrasts"))
    end
    
    # Validate focal variables
    variable_list = focal_variables isa Symbol ? [focal_variables] : collect(focal_variables)
    data_columns = Symbol.(names(data))
    
    for variable in variable_list
        if variable ∉ data_columns
            throw(ArgumentError("Focal variable $variable not found in data columns: $data_columns"))
        end
    end
end

"""
    extract_link_functions(model) -> (inverse_link, first_derivative, second_derivative)

Extract link function and its derivatives from a statistical model.
"""
function extract_link_functions(model::StatisticalModel)
    model_family = family(model)
    link_object = model_family.link
    
    inverse_link = η -> linkinv(link_object, η)
    first_derivative = η -> mueta(link_object, η)
    second_derivative = η -> mueta2(link_object, η)
    
    return inverse_link, first_derivative, second_derivative
end

"""
    get_inverse_link_function(workspace::MarginalEffectsWorkspace) -> Function

Get inverse link function from workspace (temporary helper).
"""
function get_inverse_link_function(workspace::MarginalEffectsWorkspace)
    # TODO: Store link functions in workspace for efficiency
    return identity  # Placeholder - should be extracted from model
end

"""
    create_representative_value_grid(representative_values::Dict{Symbol,<:AbstractVector}) -> Vector{Tuple}

Create Cartesian product grid of representative value combinations.
"""
function create_representative_value_grid(representative_values::AbstractDict{Symbol,<:AbstractVector})
    variables = collect(keys(representative_values))
    value_vectors = [collect(representative_values[var]) for var in variables]
    return collect(Iterators.product(value_vectors...))
end

"""
    vcov_from_cholesky(cholesky_covariance::LinearAlgebra.Cholesky) -> Matrix

Convert Cholesky decomposition back to covariance matrix.
"""
function vcov_from_cholesky(cholesky_covariance::LinearAlgebra.Cholesky)
    return Matrix(cholesky_covariance)
end

"""
    is_continuous_variable(variable::Symbol, column_data::NamedTuple) -> Bool

Determine if a variable should be treated as continuous for marginal effects.
"""
function is_continuous_variable(variable::Symbol, column_data::NamedTuple)
    if !haskey(column_data, variable)
        return false
    end
    
    values = column_data[variable]
    element_type = eltype(values)
    
    # Exclude Boolean and Categorical types
    if element_type <: Bool || values isa CategoricalArray
        return false
    end
    
    # Include numeric types
    return element_type <: Real
end

"""
    is_boolean_variable(variable::Symbol, column_data::NamedTuple) -> Bool

Determine if a variable is Boolean for marginal effects computation.
"""
function is_boolean_variable(variable::Symbol, column_data::NamedTuple)
    if !haskey(column_data, variable)
        return false
    end
    
    values = column_data[variable]
    return eltype(values) <: Bool
end
