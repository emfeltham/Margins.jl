# engine/caching.jl - Unified caching system for MarginsEngine instances

"""
    ENGINE_CACHE

Global cache for MarginsEngine instances to avoid recompilation.
Uses FormulaCompiler's recommended caching patterns.

Replaces the fragmented caching system that had separate COMPILED_CACHE 
instances in population/core.jl and profile/core.jl, plus TYPICAL_VALUES_CACHE.
"""
const ENGINE_CACHE = Dict{UInt64, MarginsEngine}()

"""
    get_or_build_engine(model, data_nt, vars) -> MarginsEngine

Get cached engine or build new one. Implements FormulaCompiler caching best practices.

This unified function replaces scattered cache logic throughout the codebase,
providing consistent caching behavior for both population and profile margins.

# Arguments
- `model`: Fitted statistical model (GLM.jl, etc.)
- `data_nt::NamedTuple`: Data in columntable format 
- `vars::Vector{Symbol}`: Variables for derivative analysis

# Returns
- `MarginsEngine`: Cached or newly built engine

# Cache Key Strategy
Creates comprehensive cache key including:
- Model object and structure
- Data structure (column names)
- Variables for derivatives
- Model type for dispatch

# Examples
```julia
# Get cached engine (or build new one)
engine = get_or_build_engine(model, data_nt, [:x1, :x2])

# Same call will return cached instance
engine2 = get_or_build_engine(model, data_nt, [:x1, :x2])  # Cache hit!
```
"""
function get_or_build_engine(model, data_nt::NamedTuple, vars::Vector{Symbol})
    # Create comprehensive cache key including all relevant factors
    cache_key = hash((
        model,                    # Model object (coefficients, structure, etc.)
        keys(data_nt),           # Data structure (column names)
        vars,                    # Variables for derivatives
        typeof(model),           # Model type for dispatch  
        fieldnames(typeof(model)) # Model structure fields
    ))
    
    return get!(ENGINE_CACHE, cache_key) do
        build_engine(model, data_nt, vars)
    end
end

"""
    clear_engine_cache!()

Clear the engine cache. Useful for memory management in long-running sessions.

# Examples
```julia
# Clear all cached engines
clear_engine_cache!()

# Subsequent calls will rebuild engines
engine = get_or_build_engine(model, data_nt, vars)  # Will rebuild
```
"""
function clear_engine_cache!()
    empty!(ENGINE_CACHE)
    return nothing
end

"""
    get_cache_stats() -> NamedTuple

Return cache statistics for monitoring and debugging.

# Returns
- `NamedTuple` with cache metrics:
  - `entries`: Number of cached engines
  - `memory_estimate`: Estimated memory usage in bytes
  - `keys`: Cache keys for debugging

# Examples
```julia
stats = get_cache_stats()
println("Cached engines: ", stats.entries)
println("Memory usage: ", stats.memory_estimate, " bytes")
```
"""
function get_cache_stats()
    entries = length(ENGINE_CACHE)
    memory_estimate = entries == 0 ? 0 : sum(sizeof(engine) for engine in values(ENGINE_CACHE))
    
    return (
        entries = entries,
        memory_estimate = memory_estimate,
        keys = collect(keys(ENGINE_CACHE))
    )
end