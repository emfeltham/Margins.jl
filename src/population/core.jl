# population/core.jl - Main population_margins() function with compilation caching

# Global cache for compiled formulas (MARGINS_GUIDE.md pattern)
# Unified caching system (see engine/caching.jl)
# Removed: const COMPILED_CACHE = Dict{UInt64, Any}()  # Now unified in engine/caching.jl

"""
    population_margins(model, data; kwargs...) -> MarginsResult

Compute population-level marginal effects or adjusted predictions.

This function averages effects/predictions across the observed sample distribution,
providing true population parameters for your sample. It implements the "Population" 
approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Average Marginal Effects (AME): population-averaged derivatives/contrasts
  - `:predictions` - Average Adjusted Predictions (AAP): population-averaged fitted values
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
- `at=nothing`: Counterfactual scenarios (Dict mapping variables to values)
  - Example: `Dict(:x1 => 0, :x2 => [1, 2])` creates scenarios for all combinations
- `over=nothing`: Subgroup analysis specification
  - NamedTuple: `(var1=values, var2=nothing)` for flexible grouping
  - Vector{Symbol}: `[:var1, :var2]` for all-categorical automatic grouping

# Returns
`MarginsResult` containing:
- Results DataFrame with estimates, standard errors, t-statistics, p-values
- Parameter gradients matrix for delta-method standard errors
- Analysis metadata (options used, model info, etc.)

# Statistical Notes
- Standard errors computed via delta method using full model covariance matrix
- Categorical variables use baseline contrasts vs reference levels
- All computations maintain statistical validity with zero tolerance for approximations

# Examples
```julia
# Average marginal effects for all continuous variables
result = population_margins(model, data)
DataFrame(result)  # Convert to DataFrame

# Specific variables with response-scale effects
result = population_margins(model, data; vars=[:x1, :x2], target=:mu)

# Average elasticities (NEW in Phase 3)
result = population_margins(model, data; vars=[:x1, :x2], measure=:elasticity)

# Semi-elasticities (NEW in Phase 3)
result = population_margins(model, data; vars=[:x1], measure=:semielasticity_x)

# Average adjusted predictions  
result = population_margins(model, data; type=:predictions)

# Counterfactual analysis: effects when x2 is set to 0 vs 1
result = population_margins(model, data; vars=[:x1], at=Dict(:x2 => [0, 1]))

# Subgroup analysis: effects by region and income categories
result = population_margins(model, data; vars=[:education], 
                          over=(region=nothing, income=[20000, 50000, 80000]))

# High-performance production use with finite differences
result = population_margins(model, data; backend=:fd, target=:eta)
```

# Frequency-Weighted Categorical Handling
Unspecified categorical variables automatically use population frequencies:
```julia
# Your data: education = 40% HS, 45% College, 15% Graduate
#           treated = 67% true, 33% false

result = population_margins(model, data; type=:effects)
# → Averages effects across actual population composition
# → Not arbitrary first levels or 50-50 assumptions
```

See also: [`profile_margins`](@ref) for effects at specific covariate combinations.
"""
function population_margins(model, data; type::Symbol=:effects, vars=nothing, target::Symbol=:mu, backend::Symbol=:auto, at=nothing, over=nothing, measure::Symbol=:effect, kwargs...)
    # Input validation
    _validate_population_inputs(model, data, type, vars, target, backend, at, over, measure)
    # Single data conversion (consistent format throughout)
    data_nt = Tables.columntable(data)
    
    # Handle vars parameter with improved validation
    if type === :effects
        vars = _process_vars_parameter(vars, data_nt)
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Proper backend selection
    # Population margins default to :fd for zero allocations across many rows
    recommended_backend = backend === :auto ? :fd : backend
    
    # Build zero-allocation engine with caching
    engine = _get_or_build_engine(model, data_nt, vars)
    
    # Handle at/over parameters for population contexts
    if at !== nothing || over !== nothing
        return _population_margins_with_contexts(engine, data_nt, vars, at, over; type, target, backend=recommended_backend, kwargs...)
    end
    
    if type === :effects
        df, G = _ame_continuous_and_categorical(engine, data_nt; target, backend=recommended_backend, measure, kwargs...)  # → AME (both continuous and categorical)
        metadata = _build_metadata(; type, vars, target, backend, measure, n_obs=length(first(data_nt)), model_type=typeof(model), kwargs...)
        return MarginsResult(df, G, metadata)
    else # :predictions  
        df, G = _population_predictions(engine, data_nt; target, kwargs...)  # → AAP
        metadata = _build_metadata(; type, vars=Symbol[], target, backend, n_obs=length(first(data_nt)), model_type=typeof(model), kwargs...)
        return MarginsResult(df, G, metadata)
    end
end

# Use unified caching system
function _get_or_build_engine(model, data_nt::NamedTuple, vars)
    # Use unified caching system from engine/caching.jl
    return get_or_build_engine(model, data_nt, vars === nothing ? Symbol[] : vars)
end

"""
    _get_continuous_variables(data_nt) -> Vector{Symbol}

Extract continuous variables from data, filtering out categorical types.
"""
function _get_continuous_variables(data_nt::NamedTuple)
    continuous_vars = Symbol[]
    for (name, col) in pairs(data_nt)
        # Continuous: numeric types except Bool
        if eltype(col) <: Real && !(eltype(col) <: Bool)
            push!(continuous_vars, name)
        end
    end
    return continuous_vars
end

"""
    _validate_population_inputs(model, data, type, vars, target, backend, at, over, measure)

Validate inputs to population_margins() with clear Julia-style error messages.
"""
function _validate_population_inputs(model, data, type::Symbol, vars, target::Symbol, backend::Symbol, at, over, measure::Symbol)
    # Validate required arguments
    if model === nothing
        throw(ArgumentError("model cannot be nothing"))
    end
    
    if data === nothing
        throw(ArgumentError("data cannot be nothing"))
    end
    
    # Validate type parameter
    if type ∉ (:effects, :predictions)
        throw(ArgumentError("type must be :effects or :predictions, got :$(type)"))
    end
    
    # Validate target parameter
    if target ∉ (:eta, :mu)
        throw(ArgumentError("target must be :eta or :mu, got :$(target)"))
    end
    
    # Validate backend parameter (including :auto for automatic selection)
    if backend ∉ (:ad, :fd, :auto)
        throw(ArgumentError("backend must be :ad, :fd, or :auto, got :$(backend)"))
    end
    
    # Validate measure parameter
    if measure ∉ (:effect, :elasticity, :semielasticity_x, :semielasticity_y)
        throw(ArgumentError("measure must be :effect, :elasticity, :semielasticity_x, or :semielasticity_y, got :$(measure)"))
    end
    
    # Measure parameter only applies to effects, not predictions
    if type === :predictions && measure !== :effect
        throw(ArgumentError("measure parameter only applies when type = :effects"))
    end
    
    # Validate vars parameter for effects
    if type === :effects && vars !== nothing
        if !(vars isa Symbol || vars isa Vector{Symbol} || vars === :all_continuous)
            throw(ArgumentError("vars must be Symbol, Vector{Symbol}, or :all_continuous for effects analysis"))
        end
    end
    
    # Validate at parameter
    if at !== nothing && !(at isa Dict)
        throw(ArgumentError("at parameter must be a Dict specifying counterfactual scenarios"))
    end
    
    # Validate over parameter
    if over !== nothing && !(over isa NamedTuple || over isa Vector{Symbol})
        throw(ArgumentError("over parameter must be a NamedTuple or Vector{Symbol} for grouping analysis"))
    end
    
    # Validate model has required methods
    try
        coef(model)
    catch e
        throw(ArgumentError("model must support coef() method (fitted statistical model required)"))
    end
    
    try
        vcov(model)
    catch e
        throw(ArgumentError("model must support vcov() method (covariance matrix required for standard errors)"))
    end
end

"""
    _process_vars_parameter(vars, data_nt) -> Vector{Symbol}

Process and validate the vars parameter with improved error handling.
"""
function _process_vars_parameter(vars, data_nt::NamedTuple)
    if vars === nothing || vars === :all_continuous
        continuous_vars = _get_continuous_variables(data_nt)
        if isempty(continuous_vars)
            throw(MarginsError("No continuous variables found in data for effects analysis. Available variables: $(collect(keys(data_nt)))"))
        end
        return continuous_vars
    elseif vars isa Symbol
        vars_vec = [vars]
        _validate_variables(data_nt, vars_vec)
        return vars_vec
    elseif vars isa Vector{Symbol}
        _validate_variables(data_nt, vars)
        return vars
    else
        throw(ArgumentError("vars must be Symbol, Vector{Symbol}, or :all_continuous"))
    end
end