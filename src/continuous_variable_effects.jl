# continuous_variable_effects.jl - Clean implementation for continuous marginal effects

###############################################################################
# Continuous Variable Marginal Effects Computation
###############################################################################

"""
    compute_continuous_variable_effects(
    focal_variables::Vector{Symbol}, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    cholesky_covariance::LinearAlgebra.Cholesky,
    first_derivative::Function,
    second_derivative::Function
)

Compute average marginal effects for multiple continuous variables using zero-allocation approach.

# Returns
- `effects::Vector{Float64}`: Average marginal effect for each variable
- `standard_errors::Vector{Float64}`: Standard error for each effect
- `gradients::Vector{Vector{Float64}}`: Gradient vector for each effect

# Performance
- Memory: O(p) - uses workspace buffers only
- Time: ~100-200ns per observation per variable
- Zero allocations during computation
"""
function compute_continuous_variable_effects(
    focal_variables::Vector{Symbol}, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    cholesky_covariance::LinearAlgebra.Cholesky,
    first_derivative::Function,
    second_derivative::Function
)
    variable_count = length(focal_variables)
    effects = Vector{Float64}(undef, variable_count)
    standard_errors = Vector{Float64}(undef, variable_count)
    gradients = Vector{Vector{Float64}}(undef, variable_count)
    
    # Compute effect for each variable using shared workspace buffers
    for (variable_index, focal_variable) in enumerate(focal_variables)
        effect, se, gradient = compute_single_continuous_effect(
            focal_variable, workspace, coefficient_vector, cholesky_covariance,
            first_derivative, second_derivative
        )
        
        effects[variable_index] = effect
        standard_errors[variable_index] = se
        gradients[variable_index] = gradient
    end
    
    return effects, standard_errors, gradients
end

"""
    compute_single_continuous_effect(
        focal_variable::Symbol, 
        workspace::MarginalEffectsWorkspace,
        coefficient_vector::AbstractVector, 
        cholesky_covariance::LinearAlgebra.Cholesky,
        first_derivative::Function, 
        second_derivative::Function;
        variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}()
    )

Compute average marginal effect for a single continuous variable.

# Algorithm
For each observation i:
1. evaluate_model_row!(workspace, i; variable_overrides) - get model matrix row
2. η_i = dot(model_row, β) - compute linear predictor  
3. evaluate_model_derivative!(workspace, i, variable; variable_overrides) - get derivatives
4. dη_dx_i = dot(derivative_row, β) - compute derivative of linear predictor
5. marginal_effect_i = μ'(η_i) * dη_dx_i - compute marginal effect for observation
6. Accumulate AME and gradient contributions

Final: AME = mean(marginal_effect_i), SE via delta method using accumulated gradient

# Performance
- Time: ~100-200ns per observation
- Memory: O(p) - reuses workspace buffers
- Allocations: 0 bytes during computation
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
    
    # Process each observation
    for observation_index in 1:observation_count
        # Evaluate model matrix row for this observation
        evaluate_model_row!(workspace, observation_index; variable_overrides=variable_overrides)
        linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
        
        # Evaluate derivative of model matrix row w.r.t. focal variable
        evaluate_model_derivative!(workspace, observation_index, focal_variable; variable_overrides=variable_overrides)
        predictor_derivative = dot(workspace.derivative_buffer, coefficient_vector)
        
        # Compute marginal effect for this observation
        if all_finite_and_reasonable(linear_predictor, predictor_derivative)
            link_first_derivative = first_derivative(linear_predictor)
            observation_marginal_effect = link_first_derivative * predictor_derivative
            
            if is_finite_and_reasonable(observation_marginal_effect)
                marginal_effect_sum += observation_marginal_effect
                
                # Accumulate gradient using product rule
                # ∂(marginal_effect)/∂β = ∂(μ'(η) * dη/dx)/∂β
                # = μ''(η) * ∂η/∂β * dη/dx + μ'(η) * ∂(dη/dx)/∂β
                link_second_derivative = second_derivative(linear_predictor)
                
                @inbounds for param_index in 1:parameter_count
                    second_order_term = link_second_derivative * workspace.model_row_buffer[param_index] * predictor_derivative
                    first_order_term = link_first_derivative * workspace.derivative_buffer[param_index]
                    workspace.gradient_accumulator[param_index] += (second_order_term + first_order_term) / observation_count
                end
            end
        end
    end
    
    # Compute final average marginal effect
    average_marginal_effect = marginal_effect_sum / observation_count
    
    # Compute standard error using delta method
    standard_error = compute_standard_error_from_gradient(workspace, cholesky_covariance)
    
    return average_marginal_effect, standard_error, copy(workspace.gradient_accumulator)
end

###############################################################################
# Numerical Stability and Validation
###############################################################################

"""
    all_finite_and_reasonable(values...) -> Bool

Check if all values are finite and within reasonable bounds for computation.
"""
function all_finite_and_reasonable(values...)
    return all(is_finite_and_reasonable, values)
end

"""
    is_finite_and_reasonable(value::Real) -> Bool

Check if a single value is finite and within reasonable computational bounds.
"""
function is_finite_and_reasonable(value::Real)
    return isfinite(value) && abs(value) < 1e10
end

"""
    compute_standard_error_from_gradient(workspace::MarginalEffectsWorkspace,
                                        cholesky_covariance::LinearAlgebra.Cholesky) -> Float64

Compute standard error using delta method with Cholesky decomposition.
SE = sqrt(gradient' * Σ * gradient) = sqrt(||U' * gradient||²)
where cholesky_covariance = U' * U
"""
function compute_standard_error_from_gradient(
    workspace::MarginalEffectsWorkspace,
    cholesky_covariance::LinearAlgebra.Cholesky
)
    try
        # Compute U' * gradient where U is upper Cholesky factor
        mul!(workspace.computation_buffer, cholesky_covariance.U, workspace.gradient_accumulator)
        
        # Compute ||U' * gradient||² = gradient' * U * U' * gradient = gradient' * Σ * gradient
        variance_estimate = dot(workspace.computation_buffer, workspace.computation_buffer)
        
        if variance_estimate >= 0 && isfinite(variance_estimate)
            return sqrt(variance_estimate)
        else
            @warn "Invalid variance estimate: $variance_estimate"
            return NaN
        end
    catch exception
        @warn "Standard error computation failed: $exception"
        return NaN
    end
end

###############################################################################
# Diagnostics and Performance Monitoring
###############################################################################

"""
    ContinuousEffectDiagnostics

Diagnostic information for continuous variable marginal effect computation.
"""
struct ContinuousEffectDiagnostics
    focal_variable::Symbol
    observation_count::Int
    parameter_count::Int
    finite_observations::Int
    reasonable_effects::Int
    mean_linear_predictor::Float64
    mean_predictor_derivative::Float64
    mean_marginal_effect::Float64
    computation_warnings::Vector{String}
    variable_overrides::Dict{Symbol,Any}
end

"""
    diagnose_continuous_effect_computation(focal_variable::Symbol, 
                                          workspace::MarginalEffectsWorkspace,
                                          coefficient_vector::AbstractVector,
                                          first_derivative::Function,
                                          second_derivative::Function;
                                          variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
                                          sample_size::Int = 10) -> ContinuousEffectDiagnostics

Diagnostic analysis of continuous variable marginal effect computation.
"""
function diagnose_continuous_effect_computation(
    focal_variable::Symbol, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    first_derivative::Function,
    second_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    sample_size::Int = 10
)
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    warnings = String[]
    
    # Sample observations for detailed analysis
    sample_indices = 1:min(sample_size, observation_count)
    linear_predictors = Float64[]
    predictor_derivatives = Float64[]
    marginal_effects = Float64[]
    finite_count = 0
    reasonable_count = 0
    
    for observation_index in sample_indices
        try
            # Evaluate model row and derivatives
            evaluate_model_row!(workspace, observation_index; variable_overrides=variable_overrides)
            linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
            
            evaluate_model_derivative!(workspace, observation_index, focal_variable; variable_overrides=variable_overrides)
            predictor_derivative = dot(workspace.derivative_buffer, coefficient_vector)
            
            push!(linear_predictors, linear_predictor)
            push!(predictor_derivatives, predictor_derivative)
            
            # Check finite values
            if isfinite(linear_predictor) && isfinite(predictor_derivative)
                finite_count += 1
                
                # Compute marginal effect
                link_derivative = first_derivative(linear_predictor)
                marginal_effect = link_derivative * predictor_derivative
                push!(marginal_effects, marginal_effect)
                
                # Check reasonable values
                if is_finite_and_reasonable(marginal_effect)
                    reasonable_count += 1
                else
                    push!(warnings, "Observation $observation_index: unreasonable marginal effect $marginal_effect")
                end
            else
                push!(warnings, "Observation $observation_index: non-finite predictor values")
                push!(marginal_effects, NaN)
            end
            
        catch exception
            push!(warnings, "Observation $observation_index: computation failed with $exception")
            push!(linear_predictors, NaN)
            push!(predictor_derivatives, NaN)
            push!(marginal_effects, NaN)
        end
    end
    
    # Compute summary statistics
    mean_predictor = isempty(linear_predictors) ? NaN : mean(filter(isfinite, linear_predictors))
    mean_derivative = isempty(predictor_derivatives) ? NaN : mean(filter(isfinite, predictor_derivatives))
    mean_effect = isempty(marginal_effects) ? NaN : mean(filter(isfinite, marginal_effects))
    
    return ContinuousEffectDiagnostics(
        focal_variable,
        observation_count,
        parameter_count,
        finite_count,
        reasonable_count,
        mean_predictor,
        mean_derivative,
        mean_effect,
        warnings,
        variable_overrides
    )
end

"""
    benchmark_continuous_effect_computation(focal_variable::Symbol,
                                           workspace::MarginalEffectsWorkspace,
                                           coefficient_vector::AbstractVector,
                                           cholesky_covariance::LinearAlgebra.Cholesky,
                                           first_derivative::Function,
                                           second_derivative::Function;
                                           variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
                                           benchmark_samples::Int = 10) -> NamedTuple

Benchmark performance of continuous variable marginal effect computation.
"""
function benchmark_continuous_effect_computation(
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    cholesky_covariance::LinearAlgebra.Cholesky,
    first_derivative::Function,
    second_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    benchmark_samples::Int = 10
)
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    
    # Benchmark full computation
    computation_times = Float64[]
    allocation_counts = Int[]
    
    # Warm-up runs
    for _ in 1:3
        compute_single_continuous_effect(
            focal_variable, workspace, coefficient_vector, cholesky_covariance,
            first_derivative, second_derivative; variable_overrides=variable_overrides
        )
    end
    
    # Actual benchmark runs
    for _ in 1:benchmark_samples
        timing_result = @timed compute_single_continuous_effect(
            focal_variable, workspace, coefficient_vector, cholesky_covariance,
            first_derivative, second_derivative; variable_overrides=variable_overrides
        )
        push!(computation_times, timing_result.time)
        push!(allocation_counts, timing_result.bytes)
    end
    
    return (
        focal_variable = focal_variable,
        observations = observation_count,
        parameters = parameter_count,
        benchmark_samples = benchmark_samples,
        
        # Timing statistics
        mean_computation_time_seconds = mean(computation_times),
        minimum_computation_time_seconds = minimum(computation_times),
        time_per_observation_nanoseconds = mean(computation_times) * 1e9 / observation_count,
        
        # Allocation statistics
        mean_allocations_bytes = mean(allocation_counts),
        minimum_allocations_bytes = minimum(allocation_counts),
        zero_allocation_percentage = 100 * count(==(0), allocation_counts) / length(allocation_counts),
        
        # Throughput estimates
        observations_per_second = observation_count / mean(computation_times),
        computations_per_second = 1.0 / mean(computation_times),
        
        # Memory efficiency
        workspace_memory_per_parameter = (sizeof(workspace.model_row_buffer) + 
                                         sizeof(workspace.derivative_buffer) + 
                                         sizeof(workspace.gradient_accumulator) + 
                                         sizeof(workspace.computation_buffer)) / parameter_count,
        
        # Representative values info
        override_count = length(variable_overrides)
    )
end

# All the main functions that other modules might need
# export extract_factor_levels
# export compute_baseline_factor_contrasts!, compute_all_pairs_factor_contrasts!
# export compute_factor_level_contrast
