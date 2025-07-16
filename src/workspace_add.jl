###############################################################################
# Helper Functions (mostly unchanged)
###############################################################################

get_observation_count(workspace::MarginalEffectsWorkspace) = length(first(workspace.column_data))
get_parameter_count(workspace::MarginalEffectsWorkspace) = length(workspace.compiled_formula)
get_variable_names(workspace::MarginalEffectsWorkspace) = collect(keys(workspace.column_data))

function validate_data_structure(column_data::NamedTuple)
    if isempty(column_data)
        throw(ArgumentError("Data cannot be empty"))
    end
    
    reference_length = length(first(column_data))
    if reference_length == 0
        throw(ArgumentError("Data cannot have zero observations"))
    end
    
    for (variable_name, variable_values) in pairs(column_data)
        current_length = length(variable_values)
        if current_length != reference_length
            throw(DimensionMismatch(
                "Variable $variable_name has $current_length observations, " *
                "expected $reference_length observations"
            ))
        end
    end
    
    return true
end

function all_finite_and_reasonable(values...)
    return all(is_finite_and_reasonable, values)
end

function is_finite_and_reasonable(value::Real)
    return isfinite(value) && abs(value) < 1e10
end

function compute_standard_error_from_gradient(workspace::MarginalEffectsWorkspace,
                                            cholesky_covariance::LinearAlgebra.Cholesky)
    try
        mul!(workspace.computation_buffer, cholesky_covariance.U, workspace.gradient_accumulator)
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

function is_continuous_variable(variable::Symbol, column_data::NamedTuple)
    if !haskey(column_data, variable)
        return false
    end
    
    values = column_data[variable]
    element_type = eltype(values)
    
    if element_type <: Bool || values isa CategoricalArray
        return false
    end
    
    return element_type <: Real
end

###############################################################################
# Performance Benchmarking
###############################################################################

"""
    benchmark_analytical_derivatives(focal_variable::Symbol, 
                                    workspace::MarginalEffectsWorkspace,
                                    coefficient_vector::AbstractVector,
                                    cholesky_covariance::LinearAlgebra.Cholesky,
                                    first_derivative::Function,
                                    second_derivative::Function;
                                    samples::Int = 100) -> NamedTuple

Benchmark the performance improvement from analytical derivatives.
"""
function benchmark_analytical_derivatives(
    focal_variable::Symbol, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    cholesky_covariance::LinearAlgebra.Cholesky,
    first_derivative::Function,
    second_derivative::Function;
    samples::Int = 100
)
    observation_count = get_observation_count(workspace)
    
    # Benchmark analytical approach
    analytical_times = Float64[]
    analytical_allocations = Int[]
    
    # Warm up
    for _ in 1:10
        compute_single_continuous_effect(focal_variable, workspace, coefficient_vector, 
                                       cholesky_covariance, first_derivative, second_derivative)
    end
    
    # Benchmark analytical derivatives
    for _ in 1:samples
        timing_result = @timed compute_single_continuous_effect(
            focal_variable, workspace, coefficient_vector, 
            cholesky_covariance, first_derivative, second_derivative
        )
        push!(analytical_times, timing_result.time)
        push!(analytical_allocations, timing_result.bytes)
    end
    
    return (
        focal_variable = focal_variable,
        observations = observation_count,
        parameters = get_parameter_count(workspace),
        samples = samples,
        
        # Analytical performance
        mean_time_seconds = mean(analytical_times),
        min_time_seconds = minimum(analytical_times),
        time_per_observation_nanoseconds = mean(analytical_times) * 1e9 / observation_count,
        
        # Allocation analysis
        mean_allocations_bytes = mean(analytical_allocations),
        min_allocations_bytes = minimum(analytical_allocations),
        zero_allocation_percentage = 100 * count(==(0), analytical_allocations) / length(analytical_allocations),
        
        # Throughput
        observations_per_second = observation_count / mean(analytical_times),
        computations_per_second = 1.0 / mean(analytical_times)
    )
end

export benchmark_analytical_derivatives
