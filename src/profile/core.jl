# profile/core.jl
# Main profile_margins() function with reference grid approach

using Distributions: Normal, cdf

"""
    profile_margins(model, data; at=:means, kwargs...) -> MarginsResult

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at representative points or user-specified scenarios,
implementing the "Profile" approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).
It provides marginal effects at the mean (MEM), marginal effects at representative values (MER),
or adjusted predictions at specific profiles (APM/APR).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)

# Keyword Arguments
- `at=:means`: Profile specification (required for profile analysis)
  - `:means` - Effects/predictions at sample means (MEM/APM)
  - `Dict` - Cartesian product specification: `Dict(:x1 => [0, 1], :x2 => [2, 3])`
  - `Vector{Dict}` - Explicit profiles: `[Dict(:x1 => 0, :x2 => 2), Dict(:x1 => 1, :x2 => 3)]`
  - `DataFrame` - Pre-built reference grid (most efficient for complex scenarios)
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Marginal Effects at profiles (MEM/MER): derivatives/contrasts at specific points
  - `:predictions` - Adjusted Predictions at profiles (APM/APR): fitted values at specific points
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or :all_continuous)
  - Only required when `type=:effects`
  - Defaults to all continuous variables (numeric types except Bool)
- `target::Symbol=:mu`: Target scale for computation
  - `:mu` - Response scale (default, applies inverse link function)  
  - `:eta` - Linear predictor scale (link scale)
- `backend::Symbol=:ad`: Computational backend
  - `:ad` - Automatic differentiation (higher accuracy, small memory cost)
  - `:fd` - Finite differences (zero allocation, production-ready)
- `measure::Symbol=:effect`: Effect measure (only for `type=:effects`)
  - `:effect` - Marginal effects (default, current behavior)
  - `:elasticity` - Elasticities (percent change in y for percent change in x)
  - `:semielasticity_x` - Semi-elasticities w.r.t. x (percent change in y for unit change in x)
  - `:semielasticity_y` - Semi-elasticities w.r.t. y (unit change in y for percent change in x)

# Returns
`MarginsResult` containing:
- Results DataFrame with estimates, standard errors, t-statistics, p-values
- Profile columns (at_varname) showing covariate values for each estimate
- Parameter gradients matrix for delta-method standard errors
- Analysis metadata (options used, model info, etc.)

# Statistical Notes
- Standard errors computed via delta method using full model covariance matrix
- Categorical variables use baseline contrasts vs reference levels at each profile
- Profile approach enables interpretation at specific, meaningful covariate combinations
- More efficient than population approach when analyzing specific scenarios

# Examples
```julia
# Effects at sample means (MEM) - most common case
result = profile_margins(model, data; at=:means, type=:effects, vars=[:x1, :x2])
DataFrame(result)  # Convert to DataFrame with profile information

# Elasticities at sample means (NEW in Phase 3)
result = profile_margins(model, data; at=:means, type=:effects, vars=[:x1], measure=:elasticity)

# Effects at specific scenarios (MER)
result = profile_margins(model, data; at=Dict(:x1 => [0, 1], :income => [25000, 50000]), 
                        type=:effects, vars=[:education])

# Semi-elasticities at specific profiles (NEW in Phase 3)
result = profile_margins(model, data; at=Dict(:x1 => [-1, 0, 1]), 
                        type=:effects, vars=[:x2], measure=:semielasticity_x)

# Predictions at the mean (APM)
result = profile_margins(model, data; at=:means, type=:predictions)

# Multiple explicit profiles for complex analysis
profiles = [
    Dict(:x1 => 0.0, :x2 => 1.0, :region => "North"),
    Dict(:x1 => 1.0, :x2 => 0.0, :region => "South")
]
result = profile_margins(model, data; at=profiles, type=:effects)

# High-performance with pre-built reference grid
reference_grid = DataFrame(x1=[0, 1, 2], x2=[10, 20, 30])
result = profile_margins(model, reference_grid; type=:predictions)

# Cartesian product for systematic exploration
result = profile_margins(model, data; 
                        at=Dict(:age => [25, 35, 45], :education => [12, 16]), 
                        type=:effects, vars=[:income], backend=:fd)
```

# Frequency-Weighted Categorical Defaults
Unspecified categorical variables use actual data composition:
```julia
# Your data: region = 75% Urban, 25% Rural
#           treated = 60% true, 40% false

# Effects "at means" now uses realistic population profile
result = profile_margins(model, data; at=:means, type=:effects)
# → income: sample mean
# → region: frequency-weighted (75% urban, 25% rural) 
# → treated: 0.6 (actual treatment rate)
# → Not arbitrary first levels!

# Override when needed for scenario analysis
result = profile_margins(model, data; 
    at=Dict(:treated => 1.0),  # 100% treatment scenario
    type=:effects)
```

See also: [`population_margins`](@ref) for population-averaged effects and predictions.
"""
# Single method to handle both cases: data with 'at' specification and explicit reference grids  
function profile_margins(model, data; at=:means, type::Symbol=:effects, vars=nothing, target::Symbol=:mu, backend::Symbol=:auto, measure::Symbol=:effect, kwargs...)
    # Convert to NamedTuple immediately to avoid DataFrame dispatch issues
    data_nt = Tables.columntable(data)
    
    # Call internal implementation with NamedTuple (no more DataFrame dispatch issues)
    return _profile_margins_impl(model, data_nt, at, type, vars, target, backend, measure, kwargs...)
end

# Internal implementation that works with NamedTuple to avoid dispatch confusion
function _profile_margins_impl(model, data_nt::NamedTuple, at, type::Symbol, vars, target::Symbol, backend::Symbol, measure::Symbol, kwargs...)
    # Input validation
    _validate_profile_inputs(model, data_nt, at, type, vars, target, backend, measure)
    
    # Build reference grid from at specification  
    reference_grid = _build_reference_grid(at, data_nt)
    at_spec = at
    
    # Handle vars parameter with improved validation - use same helper as population_margins
    if type === :effects
        vars = _process_vars_parameter(vars, data_nt)
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Proper backend selection:
    # Profile margins default to :ad for speed/accuracy at specific points
    recommended_backend = backend === :auto ? :ad : backend
    
    # Build zero-allocation engine with caching
    engine = _get_or_build_engine_for_profiles(model, data_nt, vars)
    
    if type === :effects
        # Convert reference grid to profiles for processing
        profiles = [Dict(pairs(row)) for row in eachrow(reference_grid)]
        df, G = _mem_continuous_and_categorical(engine, profiles; target, backend=recommended_backend, measure)  # → MEM/MER
        metadata = _build_metadata(; type, vars, target, backend=recommended_backend, measure, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec, kwargs...)
        return MarginsResult(df, G, metadata)
    else # :predictions  
        df, G = _profile_predictions(engine, reference_grid; target, kwargs...)  # → APM/APR
        metadata = _build_metadata(; type, vars=Symbol[], target, backend=recommended_backend, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec, kwargs...)
        return MarginsResult(df, G, metadata)
    end
end

"""
    profile_margins(model, reference_grid::DataFrame; kwargs...) -> MarginsResult

Compute profile margins using an explicit reference grid DataFrame.

This is the most efficient method when you have already constructed your desired reference grid.
No additional profile building is performed - the DataFrame is used directly, making this 
ideal for complex scenarios or when the same reference grid will be reused multiple times.

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `reference_grid::DataFrame`: Pre-built reference grid with specific covariate combinations

# Keyword Arguments
Same as the main `profile_margins()` method, except `at` parameter is not needed since
the reference grid is provided directly.

# Returns
`MarginsResult` with the same structure as the main method, containing results for
each row of the reference grid.

# Performance Notes
- Most efficient profile margins method (zero reference grid construction overhead)
- Ideal for complex scenarios requiring custom reference grids
- Perfect for repeated analysis with the same covariate combinations
- Enables maximum control over profile specification

# Examples
```julia
# Pre-built reference grid for systematic analysis
reference_grid = DataFrame(
    age = [25, 35, 45, 55],
    education = [12, 16, 12, 16], 
    income = [30000, 50000, 35000, 60000]
)
result = profile_margins(model, reference_grid; type=:effects, vars=[:experience])

# Complex factorial design
grid = expand.grid(
    treatment = ["A", "B", "C"],
    dose = [10, 20, 30],
    age_group = ["young", "old"]
)
result = profile_margins(model, grid; type=:predictions)

# Reuse same reference grid for multiple analyses
effects_result = profile_margins(model, reference_grid; type=:effects)
predictions_result = profile_margins(model, reference_grid; type=:predictions)
```
"""
# Note: DataFrame method removed to avoid dispatch ambiguity
# Both use cases (data with 'at' and explicit reference_grid) now handled by single method above

"""
    _get_or_build_engine_for_profiles(model, data_nt, vars)

Get or build engine with caching for profile computations. Uses unified caching system.
"""
function _get_or_build_engine_for_profiles(model, data_nt::NamedTuple, vars)
    # Use unified caching system from engine/caching.jl
    return get_or_build_engine(model, data_nt, vars === nothing ? Symbol[] : vars)
end

"""
    _profile_predictions(engine, reference_grid; target, kwargs...) -> (DataFrame, Matrix{Float64})

Compute adjusted predictions at profiles (APM/APR) with delta-method standard errors.

This function evaluates predictions at each row of the reference grid, providing
adjusted predictions at the mean (APM) or at representative values (APR).
"""
function _profile_predictions(engine::MarginsEngine, reference_grid::DataFrame; target=:mu, kwargs...)
    n_profiles = nrow(reference_grid)
    n_params = length(engine.β)
    
    # Reuse η_buf for predictions if possible
    predictions = length(engine.η_buf) >= n_profiles ? view(engine.η_buf, 1:n_profiles) : Vector{Float64}(undef, n_profiles)
    G = zeros(n_profiles, n_params)  # One row per profile
    
    # Convert reference grid to NamedTuple format for FormulaCompiler
    data_nt = Tables.columntable(reference_grid)
    
    # Single pass over profiles via helper that takes only concrete arguments
    _profile_predictions_impl!(predictions, G, engine.compiled, engine.row_buf,
                               engine.β, engine.link, data_nt, target)
    
    # Safely use g_buf for SE computation if large enough  
    if length(engine.g_buf) >= n_profiles
        se_vals = view(engine.g_buf, 1:n_profiles)  # Reuse g_buf if large enough
    else
        se_vals = Vector{Float64}(undef, n_profiles)  # Fall back to allocation
    end
    for i in 1:n_profiles
        se_vals[i] = sqrt((G[i:i, :] * engine.Σ * G[i:i, :]')[1, 1])
    end
    
    # Create results DataFrame with profile information
    results = DataFrame()
    results.term = ["APM/APR" for _ in 1:n_profiles]
    results.estimate = predictions
    results.se = se_vals
    results.t_stat = predictions ./ se_vals
    results.p_value = 2 .* (1 .- cdf.(Normal(), abs.(predictions ./ se_vals)))
    
    # Add profile columns to show which profile each prediction corresponds to
    for (col_name, col_data) in pairs(Tables.columns(reference_grid))
        results[!, Symbol("at_$(col_name)")] = col_data
    end
    
    return results, G
end

function _profile_predictions_impl!(predictions::AbstractVector{<:Float64},
                                    G::AbstractMatrix{<:Float64},
                                    compiled,
                                    row_buf::Vector{Float64},
                                    β::Vector{Float64},
                                    link,
                                    data_nt::NamedTuple,
                                    target::Symbol)
    n_profiles = length(predictions)
    if target === :mu
        for i in 1:n_profiles
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            μ = GLM.linkinv(link, η)
            dμ_dη = GLM.mueta(link, η)
            predictions[i] = μ
            @inbounds for j in 1:length(row_buf)
                G[i, j] = dμ_dη * row_buf[j]
            end
        end
    else
        for i in 1:n_profiles
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            predictions[i] = η
            @inbounds for j in 1:length(row_buf)
                G[i, j] = row_buf[j]
            end
        end
    end
    return nothing
end

"""
    _validate_profile_inputs(model, data, at, type, vars, target, backend, measure)

Validate inputs to profile_margins() with clear Julia-style error messages.
"""
function _validate_profile_inputs(model, data, at, type::Symbol, vars, target::Symbol, backend::Symbol, measure::Symbol)
    # Reuse population validation for common parameters
    _validate_population_inputs(model, data, type, vars, target, backend, nothing, nothing, measure)
    
    # Note: at parameter validation 
    # Since the function signature has at=:means as default, at===nothing should never occur
    # But we include this check for robustness if called directly with at=nothing
    
    if !(at === :means || at isa Dict || at isa Vector || at isa DataFrame)
        throw(ArgumentError("at parameter must be :means, Dict, Vector{Dict}, or DataFrame"))
    end
    
    # Additional validation for Dict specification
    if at isa Dict
        if isempty(at)
            throw(ArgumentError("at Dict cannot be empty - specify at least one variable and its values"))
        end
        for (k, v) in pairs(at)
            if !(k isa Symbol)
                throw(ArgumentError("at Dict keys must be Symbols (variable names), got $(typeof(k))"))
            end
        end
    end
    
    # Additional validation for Vector specification  
    if at isa Vector && !isempty(at)
        for (i, profile) in enumerate(at)
            if !(profile isa Dict || profile isa NamedTuple)
                throw(ArgumentError("at Vector elements must be Dict or NamedTuple (profiles), element $i is $(typeof(profile))"))
            end
        end
    end
end
