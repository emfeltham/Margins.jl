# computation/result_formatting.jl - Result construction and formatting utilities

using DataFrames
using LinearAlgebra: dot

"""
    build_results_dataframe(total_rows::Int, n_obs::Int, n_params::Int) -> (DataFrame, Matrix{Float64})

Pre-allocate results DataFrame and gradient matrix for population marginal effects.
Eliminates allocation during result construction with proper capacity planning.

# Arguments
- `total_rows`: Total number of result rows needed
- `n_obs`: Number of observations in the dataset
- `n_params`: Number of model parameters (for gradient matrix width)

# Returns
- `DataFrame`: Pre-allocated results DataFrame with proper column types
- `Matrix{Float64}`: Pre-allocated gradient matrix (total_rows × n_params)

# DataFrame Schema
- `variable::String`: Variable name
- `contrast::String`: Contrast type ("dy/dx" for continuous, level names for categorical)
- `estimate::Float64`: Point estimate of marginal effect
- `se::Float64`: Standard error via delta method
- `n::Int`: Number of observations

# Performance Benefits
- Pre-allocation eliminates growth-related allocations during result construction
- Proper typing ensures type stability throughout result processing
- Gradient matrix pre-allocated for efficient standard error computation

# Special Cases
- **Empty results** (total_rows = 0): Returns empty DataFrame with correct schema
- **Large datasets**: Efficient pre-allocation scales well with dataset size
"""
function build_results_dataframe(total_rows::Int, n_obs::Int, n_params::Int)
    if total_rows == 0
        empty_df = DataFrame(
            variable = String[],
            contrast = String[],
            estimate = Float64[],
            se = Float64[],
            n = Int[]
        )
        return (empty_df, Matrix{Float64}(undef, 0, n_params))
    end

    results = DataFrame(
        variable = Vector{String}(undef, total_rows),
        contrast = Vector{String}(undef, total_rows),
        estimate = Vector{Float64}(undef, total_rows),
        se = Vector{Float64}(undef, total_rows),
        n = fill(n_obs, total_rows)
    )
    G = Matrix{Float64}(undef, total_rows, n_params)

    return (results, G)
end

"""
    build_profile_results_dataframe(total_terms::Int, n_params::Int) -> (DataFrame, Matrix{Float64})

Pre-allocate results DataFrame and gradient matrix for profile margins.
Eliminates allocation during result construction for profile-based analysis.

# Arguments
- `total_terms`: Total number of profile margin terms needed
- `n_params`: Number of model parameters (for gradient matrix width)

# Returns
- `DataFrame`: Empty results DataFrame with proper column types (grown via push!)
- `Matrix{Float64}`: Pre-allocated gradient matrix (total_terms × n_params)

# DataFrame Schema
- `variable::String`: Variable name
- `contrast::String`: Contrast type ("derivative" for continuous variables)
- `estimate::Float64`: Point estimate of marginal effect
- `se::Float64`: Standard error via delta method
- `profile_desc::NamedTuple`: Profile specification (variable values at which effect is computed)

# Implementation Notes
Profile DataFrames use push! rather than indexing because profile descriptions
vary in structure and cannot be easily pre-allocated. The gradient matrix is
still pre-allocated for efficient standard error computation.
"""
function build_profile_results_dataframe(total_terms::Int, n_params::Int)
    results = DataFrame(
        variable = String[],
        contrast = String[],
        estimate = Float64[],
        se = Float64[],
        profile_desc = NamedTuple[]
    )
    G = Matrix{Float64}(undef, total_terms, n_params)
    return (results, G)
end

"""
    store_continuous_result!(results::DataFrame, G::Matrix, row_idx::Int, var::Symbol,
                            ame_val::Float64, gradient::AbstractVector,
                            covariance::Matrix, n_obs::Int)

Store continuous variable result in pre-allocated DataFrame with standard error computation.
Zero additional allocations, in-place operations only.

# Arguments
- `results`: Pre-allocated results DataFrame (modified in-place)
- `G`: Pre-allocated gradient matrix (modified in-place)
- `row_idx`: Row index to store result
- `var`: Variable name
- `ame_val`: Average marginal effect value
- `gradient`: Parameter gradient vector
- `covariance`: Parameter covariance matrix
- `n_obs`: Number of observations

# Standard Error Computation
Uses delta method: SE = √(g'Σg) where:
- g is the parameter gradient vector
- Σ is the parameter covariance matrix
- Includes max(0.0, ·) for numerical stability

# Performance
- Zero allocations through pre-allocated DataFrame indexing
- Direct gradient storage with view operations
- Single standard error computation with dot product
"""
function store_continuous_result!(results::DataFrame, G::Matrix, row_idx::Int,
                                 var::Symbol, ame_val::Float64, gradient::AbstractVector,
                                 covariance::Matrix, n_obs::Int)
    # Store results
    results.variable[row_idx] = string(var)
    results.contrast[row_idx] = "dy/dx"
    results.estimate[row_idx] = ame_val
    results.n[row_idx] = n_obs

    # Store gradient for SE computation
    copyto!(view(G, row_idx, :), gradient)

    # Compute standard error using delta method: SE = √(g'Σg)
    se = sqrt(max(0.0, dot(gradient, covariance, gradient)))
    results.se[row_idx] = se

    return nothing
end

"""
    store_profile_result!(results::DataFrame, G::Matrix, row_idx::Int, var::Symbol,
                         effect_val::Float64, gradient::AbstractVector,
                         covariance::Matrix, profile::Dict)

Store profile margin result in DataFrame with standard error computation.
Uses push! for DataFrame growth and indexing for gradient matrix.

# Arguments
- `results`: Results DataFrame (modified via push!)
- `G`: Pre-allocated gradient matrix (modified in-place)
- `row_idx`: Row index for gradient matrix storage
- `var`: Variable name
- `effect_val`: Marginal effect value at the profile
- `gradient`: Parameter gradient vector
- `covariance`: Parameter covariance matrix
- `profile`: Profile specification as Dict

# Profile Description
The profile description is converted to NamedTuple for efficient storage
and consistent access patterns. Contains variable values at which the
marginal effect is computed.

# Performance
- DataFrame growth via push! (necessary for varying profile structures)
- Gradient matrix uses pre-allocated indexing for efficiency
- Single standard error computation per result
"""
function store_profile_result!(results::DataFrame, G::Matrix, row_idx::Int,
                              var::Symbol, effect_val::Float64, gradient::AbstractVector,
                              covariance::Matrix, profile::Dict)
    # Compute standard error
    se = sqrt(max(0.0, dot(gradient, covariance, gradient)))

    # Store results
    push!(results, (
        variable = string(var),
        contrast = "derivative",
        estimate = effect_val,
        se = se,
        profile_desc = NamedTuple(profile)
    ))

    # Store gradient
    G[row_idx, :] = gradient

    return nothing
end

# Reference grid construction and metadata utilities

"""
    _build_metadata(; kwargs...) -> Dict{Symbol, Any}

Build comprehensive metadata dictionary for results.
Provides standardized metadata structure for both population and profile margins results.

# Keyword Arguments
- `type::Symbol=:unknown`: Result type (:population, :profile, :custom)
- `vars::Vector{Symbol}=Symbol[]`: Variables included in analysis
- `scale::Symbol=:response`: Prediction scale (:response or :linear)
- `backend::Symbol=:ad`: Computation backend (:ad or :fd)
- `measure::Symbol=:effect`: Econometric measure (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
- `n_obs::Int=0`: Number of observations in dataset
- `model_type=nothing`: Statistical model type (for display)
- `timestamp=nothing`: Analysis timestamp (defaults to current time)
- `at_spec=nothing`: Profile specification for profile margins
- `has_contexts::Bool=false`: Whether results include contextual information

# Returns
- `Dict{Symbol, Any}`: Comprehensive metadata dictionary

# Metadata Structure
The returned dictionary contains standardized keys for result documentation:
- **Analysis specifications**: type, scale, backend, measure
- **Variable information**: vars, n_vars (computed)
- **Model information**: n_obs, model_type (stringified)
- **Timing information**: timestamp (ISO format)
- **Profile specifications**: at_spec, has_contexts

# Usage Contexts
- **Population margins**: type=:population, at_spec=nothing
- **Profile margins**: type=:profile, at_spec=profile_dict
- **Custom analyses**: type=:custom with appropriate specifications

# Timestamp Handling
- **Automatic**: Uses current time if timestamp=nothing
- **Manual**: Accepts any timestamp format for consistency
- **Format**: String representation for serialization compatibility

# Model Type Processing
Safely handles model type conversion:
- **Valid models**: Converts type to string representation
- **Nothing/missing**: Uses "unknown" as fallback
- **Type safety**: Prevents errors from invalid model objects

# Example
```julia
metadata = _build_metadata(
    type=:profile,
    vars=[:age, :education],
    scale=:response,
    backend=:ad,
    measure=:elasticity,
    n_obs=1000,
    model_type=LinearModel,
    at_spec=Dict(:age => 30.0, :education => "college")
)
# Returns comprehensive metadata dictionary for profile elasticity analysis
```
"""
function _build_metadata(;
    type=:unknown,
    vars=Symbol[],
    scale=:response,
    backend=:ad,
    measure=:effect,
    n_obs=0,
    model_type=nothing,
    timestamp=nothing,
    at_spec=nothing,
    has_contexts=false)

    # Generate timestamp if not provided
    ts = isnothing(timestamp) ? string(now()) : timestamp

    return Dict{Symbol, Any}(
        :type => type,
        :vars => vars,
        :n_vars => isnothing(vars) ? 0 : length(vars),
        :scale => scale,
        :backend => backend,
        :measure => measure,
        :n_obs => n_obs,
        :model_type => isnothing(model_type) ? "unknown" : string(typeof(model_type)),
        :timestamp => ts,
        :at_spec => at_spec,
        :has_contexts => has_contexts
    )
end
