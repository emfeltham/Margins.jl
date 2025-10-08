# Centralized statistical computation utilities for Margins.jl.
#
# This file consolidates the repeated delta-method SE computation patterns
# while maintaining **ZERO ALLOCATION** characteristics. All functions are
# designed to work with pre-allocated buffers and existing data structures.
#
# **CRITICAL**: No new allocations are introduced. All functions either:
# 1. Return scalar values (zero allocation)
# 2. Write to pre-allocated output buffers
# 3. Work with existing gradient vectors without copying

"""
    compute_delta_method_se(gradient::Vector{Float64}, Σ::Matrix{Float64}) -> Float64

Compute delta-method standard error from gradient and covariance matrix.

This is a zero-allocation wrapper around delta_method_se()
that provides a consistent API point for all SE computations in Margins.jl.

# Arguments
- `gradient::Vector{Float64}`: Pre-allocated gradient vector (not copied)
- `Σ::Matrix{Float64}`: Model covariance matrix

# Returns
- `Float64`: Standard error (zero allocation)

# Performance
- Zero allocations (direct delegation to FormulaCompiler)
- No gradient copying or modification
- Identical performance to direct FormulaCompiler calls
"""
@inline function compute_delta_method_se(gradient::AbstractVector{Float64}, Σ::Matrix{Float64})::Float64
    return delta_method_se(gradient, Σ)
end

"""
    compute_t_statistic(estimate::Float64, se::Float64) -> Float64

Compute t-statistic from estimate and standard error.
Zero allocation scalar computation.
"""
@inline function compute_t_statistic(estimate::Float64, se::Float64)::Float64
    return estimate / se
end

"""
    compute_p_value(t_stat::Float64) -> Float64

Compute two-tailed p-value from t-statistic.
Zero allocation scalar computation.
"""
@inline function compute_p_value(t_stat::Float64)::Float64
    return 2 * (1 - cdf(Normal(), abs(t_stat)))
end

"""
    compute_confidence_interval(estimate::Float64, se::Float64, α::Float64=0.05) -> (Float64, Float64)

Compute confidence interval bounds.
Zero allocation, returns tuple of scalars.

# Arguments
- `estimate::Float64`: Point estimate
- `se::Float64`: Standard error  
- `α::Float64`: Significance level (default 0.05 for 95% CI)

# Returns
- `(lower_bound, upper_bound)`: Tuple of confidence bounds
"""
@inline function compute_confidence_interval(estimate::Float64, se::Float64, α::Float64=0.05)
    critical_value = quantile(Normal(), 1 - α/2)
    margin = critical_value * se
    return (estimate - margin, estimate + margin)
end

"""
    compute_statistical_summary_scalars(estimate::Float64, gradient::Vector{Float64}, Σ::Matrix{Float64}, α::Float64=0.05) -> (Float64, Float64, Float64, Float64, Float64)

Compute complete statistical summary returning scalar tuple.

This is the **preferred** function for most use cases as it avoids any allocation
by returning a tuple of scalars rather than a struct.

# Arguments
- `estimate::Float64`: Point estimate
- `gradient::Vector{Float64}`: Pre-allocated gradient (not copied)
- `Σ::Matrix{Float64}`: Covariance matrix
- `α::Float64`: Significance level for CI

# Returns
`(se, t_stat, p_value, ci_lower, ci_upper)` - All Float64 scalars

# Performance
- Zero allocations (all scalar computations)
- No gradient copying
- Optimal for tight loops and performance-critical code
"""
@inline function compute_statistical_summary_scalars(
    estimate::Float64, gradient::Vector{Float64}, Σ::Matrix{Float64}, α::Float64=0.05
)
    se = compute_delta_method_se(gradient, Σ)
    t_stat = compute_t_statistic(estimate, se)
    p_value = compute_p_value(t_stat)
    ci_lower, ci_upper = compute_confidence_interval(estimate, se, α)
    
    return (se, t_stat, p_value, ci_lower, ci_upper)
end

"""
    compute_se_only(gradient::Vector{Float64}, Σ::Matrix{Float64}) -> Float64

Compute only the standard error (most common use case).
Identical to compute_delta_method_se but with clearer naming for single-purpose use.
"""
@inline function compute_se_only(gradient::AbstractVector{Float64}, Σ::Matrix{Float64})::Float64
    return delta_method_se(gradient, Σ)
end

# End of statistics.jl