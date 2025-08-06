# workspace_add.jl
# Updated helper functions for FormulaCompiler integration

###############################################################################
# Helper Functions for MarginalEffectsWorkspace
###############################################################################

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

###############################################################################
# Performance Benchmarking with FormulaCompiler
###############################################################################

"""
    benchmark_analytical_derivatives(focal_variable::Symbol, 
                                    workspace::MarginalEffectsWorkspace,
                                    coefficient_vector::AbstractVector,
                                    cholesky_covariance::LinearAlgebra.Cholesky,
                                    first_derivative::Function,
                                    second_derivative::Function;
                                    samples::Int = 100) -> NamedTuple

Benchmark the performance improvement from FormulaCompiler.jl analytical derivatives.
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
    
    # Benchmark FormulaCompiler analytical approach
    analytical_times = Float64[]
    analytical_allocations = Int[]
    
    # Warm up
    for _ in 1:10
        for i in 1:min(10, observation_count)
            evaluate_model_derivative!(workspace, i, focal_variable)
        end
    end
    
    # Benchmark analytical derivatives from FormulaCompiler
    for _ in 1:samples
        timing_result = @timed begin
            for i in 1:min(100, observation_count)  # Sample subset for benchmarking
                evaluate_model_derivative!(workspace, i, focal_variable)
            end
        end
        push!(analytical_times, timing_result.time)
        push!(analytical_allocations, timing_result.bytes)
    end
    
    return (
        focal_variable = focal_variable,
        observations_tested = min(100, observation_count),
        total_observations = observation_count,
        parameters = get_parameter_count(workspace),
        benchmark_samples = samples,
        
        # FormulaCompiler analytical performance
        mean_time_seconds = mean(analytical_times),
        min_time_seconds = minimum(analytical_times),
        time_per_observation_nanoseconds = mean(analytical_times) * 1e9 / min(100, observation_count),
        
        # Allocation analysis (should be near zero)
        mean_allocations_bytes = mean(analytical_allocations),
        min_allocations_bytes = minimum(analytical_allocations),
        zero_allocation_percentage = 100 * count(==(0), analytical_allocations) / length(analytical_allocations),
        
        # Throughput
        observations_per_second = min(100, observation_count) / mean(analytical_times),
        derivative_computations_per_second = min(100, observation_count) / mean(analytical_times),
        
        # Integration info
        using_formulacompiler = true,
        method = "analytical_derivatives"
    )
end

export benchmark_analytical_derivatives
