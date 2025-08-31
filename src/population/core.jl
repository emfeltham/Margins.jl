# population/core.jl - Main population_margins() function with compilation caching

# Global cache for compiled formulas (MARGINS_GUIDE.md pattern)
const COMPILED_CACHE = Dict{UInt64, Any}()

"""
    population_margins(model, data; type=:effects, vars=nothing, target=:mu, backend=:ad, at=nothing, over=nothing, kwargs...)

Compute population-level marginal effects or adjusted predictions.

This function averages effects/predictions across the observed sample distribution,
providing true population parameters for your sample.

# Arguments
- `model`: Fitted statistical model (GLM, etc.)
- `data`: Data table (DataFrame, NamedTuple, etc.)
- `type::Symbol=:effects`: Type of analysis (`:effects` for AME, `:predictions` for AAP)
- `vars=nothing`: Variables for analysis (only needed for `:effects`; defaults to all continuous)
- `target::Symbol=:mu`: Target scale (`:mu` for response scale, `:eta` for linear predictor)
- `backend::Symbol=:ad`: Computational backend (`:ad` or `:fd`)
- `at=nothing`: Counterfactual scenarios (Dict specifying variable values)
- `over=nothing`: Subgroup analysis (NamedTuple or Vector specifying grouping variables)

# Returns
- `MarginsResult`: Container with results DataFrame, gradients matrix, and metadata

# Examples
```julia
# Average marginal effects (AME)
result = population_margins(model, data; type=:effects, vars=[:x1, :x2])

# Average adjusted predictions (AAP)  
result = population_margins(model, data; type=:predictions)

# Population effects with counterfactual scenarios
result = population_margins(model, data; type=:effects, vars=[:x1], 
                          at=Dict(:x2 => [0, 1]))

# Population effects by subgroups
result = population_margins(model, data; type=:effects, vars=[:x1], 
                          over=(region=nothing, income=[10000, 50000]))
```
"""
function population_margins(model, data; type::Symbol=:effects, vars=nothing, target::Symbol=:mu, backend::Symbol=:ad, at=nothing, over=nothing, kwargs...)
    # Single data conversion (consistent format throughout)
    data_nt = Tables.columntable(data)
    
    # Handle vars parameter (only needed for type=:effects)
    if type === :effects
        vars = vars === nothing ? :all_continuous : vars  # Default to all continuous variables
        if vars === :all_continuous
            vars = _get_continuous_variables(data_nt)
        elseif isa(vars, Symbol)
            vars = [vars]  # Convert single var to vector
        end
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Build zero-allocation engine with caching
    engine = _get_or_build_engine(model, data_nt, vars)
    
    # Handle at/over parameters for population contexts
    if at !== nothing || over !== nothing
        return _population_margins_with_contexts(engine, data_nt, vars, at, over; type, target, backend, kwargs...)
    end
    
    if type === :effects
        df, G = _ame_continuous_and_categorical(engine, data_nt; target, backend, kwargs...)  # → AME (both continuous and categorical)
        metadata = _build_metadata(; type, vars, target, backend, n_obs=length(first(data_nt)), model_type=typeof(model), kwargs...)
        return MarginsResult(df, G, metadata)
    else # :predictions  
        df, G = _population_predictions(engine, data_nt; target, kwargs...)  # → AAP
        metadata = _build_metadata(; type, vars=Symbol[], target, backend, n_obs=length(first(data_nt)), model_type=typeof(model), kwargs...)
        return MarginsResult(df, G, metadata)
    end
end

# Compilation caching (MARGINS_GUIDE.md pattern)
function _get_or_build_engine(model, data_nt::NamedTuple, vars)
    cache_key = hash((model, keys(data_nt), vars))  # Include vars in cache key
    if haskey(COMPILED_CACHE, cache_key)
        return COMPILED_CACHE[cache_key]
    else
        engine = build_engine(model, data_nt, vars === nothing ? Symbol[] : vars)
        COMPILED_CACHE[cache_key] = engine
        return engine
    end
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