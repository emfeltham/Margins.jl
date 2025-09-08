# engine/caching.jl - Unified caching system for MarginsEngine instances

"""
    ENGINE_CACHE

Global cache for MarginsEngine instances to avoid recompilation.
Uses FormulaCompiler's recommended caching patterns.

Replaces the fragmented caching system that had separate COMPILED_CACHE 
instances in population/core.jl and profile/core.jl, plus TYPICAL_VALUES_CACHE.

Cache key includes usage type to ensure proper buffer sizing for different usage patterns.
"""
const ENGINE_CACHE = Dict{UInt64, MarginsEngine}()

"""
    get_or_build_engine(usage, deriv, model, data_nt, vars, vcov) -> MarginsEngine

Get cached engine or build new one with usage-specific optimization and derivative support.

This unified function replaces scattered cache logic throughout the codebase,
providing consistent caching behavior for both population and profile margins
with compile-time derivative support specialization.

# Arguments
- `usage::Type{<:MarginsUsage}`: Usage pattern (PopulationUsage or ProfileUsage)
- `deriv::Type{<:DerivativeSupport}`: Derivative support (HasDerivatives or NoDerivatives)
- `model`: Fitted statistical model (GLM.jl, etc.)
- `data_nt::NamedTuple`: Data in columntable format 
- `vars::Vector{Symbol}`: Variables for derivative analysis
- `vcov`: Covariance estimator (function or CovarianceMatrices estimator)

# Returns
- `MarginsEngine{L, U, D}`: Cached or newly built usage-optimized engine with derivative support

# Cache Key Strategy
Creates comprehensive cache key including:
- Usage type for buffer sizing strategy
- Derivative support for compile-time specialization
- Model object and structure  
- Data structure (column names)
- Variables for derivatives
- Model type for dispatch
- Covariance specification

# Examples
```julia
# Get cached population margins engine with derivatives
engine = get_or_build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:x1, :x2], GLM.vcov)

# Get cached profile margins engine without derivatives (different cache entry)
engine2 = get_or_build_engine(ProfileUsage, NoDerivatives, model, data_nt, [:category], GLM.vcov)

# Same call will return cached instance
engine3 = get_or_build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:x1, :x2], GLM.vcov)  # Cache hit!
```
"""
function get_or_build_engine(usage::Type{U}, deriv::Type{D}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov) where {U<:MarginsUsage, D<:DerivativeSupport}
    # Create comprehensive cache key including usage type, derivative support, and all relevant factors
    cache_key = hash((
        usage,                   # Usage type for buffer sizing strategy
        deriv,                   # Derivative support for compile-time specialization (NEW!)
        model,                   # Model object (coefficients, structure, etc.)
        keys(data_nt),          # Data structure (column names)
        vars,                   # Variables for derivatives
        typeof(model),          # Model type for dispatch  
        fieldnames(typeof(model)), # Model structure fields
        vcov                    # Covariance matrix specification (critical for caching!)
    ))
    
    return get!(ENGINE_CACHE, cache_key) do
        build_engine(usage, deriv, model, data_nt, vars, vcov)
    end
end

# Backward compatibility function without DerivativeSupport (auto-detects)
"""
    get_or_build_engine(usage, model, data_nt, vars, vcov) -> MarginsEngine

**Backward compatibility**: Auto-detect derivative support based on continuous variables in data.
New code should use the explicit DerivativeSupport version for better type safety.
"""
function get_or_build_engine(usage::Type{U}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov) where {U<:MarginsUsage}
    # Auto-detect derivative support based on data
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    
    # Auto-select DerivativeSupport based on presence of continuous variables
    deriv = isempty(continuous_vars) ? NoDerivatives : HasDerivatives
    
    return get_or_build_engine(usage, deriv, model, data_nt, vars, vcov)
end

"""
    clear_engine_cache!()

Clear the engine cache. Useful for memory management in long-running sessions.

# Examples
```julia
# Clear all cached engines
clear_engine_cache!()

# Subsequent calls will rebuild engines
engine = get_or_build_engine(PopulationUsage, model, data_nt, vars, vcov)  # Will rebuild
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