# representative_effects.jl
# Cleaned up to remove redundant functions

"""
    create_representative_value_grid(representative_values::Dict{Symbol, <:AbstractVector}) -> Vector{Tuple}

Create Cartesian product grid of representative value combinations.
This is a utility function that complements FormulaCompiler's scenario system.
"""
function create_representative_value_grid(representative_values::AbstractDict{Symbol,<:AbstractVector})
    if isempty(representative_values)
        return [()]  # Single empty combination
    end
    
    variables = collect(keys(representative_values))
    value_vectors = [collect(representative_values[var]) for var in variables]
    return collect(Iterators.product(value_vectors...))
end

"""
    validate_representative_values(representative_values::AbstractDict, data) -> Bool

Validate that representative values are appropriate for the data.
"""
function validate_representative_values(representative_values::AbstractDict, data)
    column_data = Tables.columntable(data)
    
    for (variable, values) in representative_values
        if !haskey(column_data, variable)
            throw(ArgumentError("Representative variable $variable not found in data"))
        end
        
        original_column = column_data[variable]
        
        # Check categorical variables have valid levels
        if original_column isa CategoricalArray
            valid_levels = levels(original_column)
            for value in values
                if value ∉ valid_levels
                    throw(ArgumentError("Representative value '$value' for $variable not in valid levels: $valid_levels"))
                end
            end
        end
        
        # Check continuous variables are reasonable
        if eltype(original_column) <: Real && !(original_column isa CategoricalArray)
            original_range = (minimum(original_column), maximum(original_column))
            for value in values
                if !(original_range[1] <= value <= original_range[2])
                    @warn "Representative value $value for $variable is outside data range $original_range"
                end
            end
        end
    end
    
    return true
end

"""
    compute_representative_value_effects!(effect_estimates, standard_errors, gradient_vectors,
                                         focal_variable_list, representative_values, factor_contrasts,
                                         workspace, coefficient_vector, cholesky_covariance,
                                         inverse_link_function, first_derivative, second_derivative)

Compute marginal effects at representative value combinations using workspace scenarios.
Note: This function is now primarily called from core.jl and uses workspace scenarios throughout.
"""
function compute_representative_value_effects!(
    effect_estimates, standard_errors, gradient_vectors,
    focal_variable_list, representative_values, factor_contrasts,
    workspace, coefficient_vector, cholesky_covariance,
    inverse_link_function, first_derivative, second_derivative
)
    # Validate representative values
    validate_representative_values(representative_values, workspace.column_data)
    
    # Extract representative variable information
    representative_variables = collect(keys(representative_values))
    value_combinations = create_representative_value_grid(representative_values)
    covariance_matrix = Matrix(cholesky_covariance)
    
    # Initialize storage for each focal variable
    for variable in focal_variable_list
        effect_estimates[variable] = Dict{Tuple,Float64}()
        standard_errors[variable] = Dict{Tuple,Float64}()
        gradient_vectors[variable] = Dict{Tuple,Vector{Float64}}()
    end
    
    # Compute effects for each representative value combination
    for value_combination in value_combinations
        # Create override dictionary for this combination using workspace scenarios
        variable_overrides = Dict{Symbol,Any}(
            var => val for (var, val) in zip(representative_variables, value_combination)
        )
        
        # Compute effects for each focal variable
        for focal_variable in focal_variable_list
            if is_continuous_variable(focal_variable, workspace.column_data)
                # Continuous focal variable using analytical derivatives + workspace scenarios
                effect, se, gradient = compute_single_continuous_effect(
                    focal_variable, workspace, coefficient_vector, cholesky_covariance,
                    first_derivative, second_derivative; variable_overrides=variable_overrides
                )
                
                effect_estimates[focal_variable][value_combination] = effect
                standard_errors[focal_variable][value_combination] = se
                gradient_vectors[focal_variable][value_combination] = gradient
                
            else
                # Categorical focal variable using workspace scenarios
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

# Helper functions that complement FormulaCompiler scenarios
function is_boolean_variable(variable::Symbol, column_data::NamedTuple)
    if !haskey(column_data, variable)
        return false
    end
    return eltype(column_data[variable]) <: Bool
end
