# engine/caching.jl - Unified caching system for MarginsEngine instances

"""
    ENGINE_CACHE

Global cache for MarginsEngine instances to avoid recompilation.
Uses FormulaCompiler's recommended caching patterns.

Replaces the fragmented caching system that had separate COMPILED_CACHE
instances in population/core.jl and profile/core.jl.

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

The key identifies the data only by its column *names*, so a hit is additionally
revalidated with `_engine_data_matches` — a cached engine is reused only when it is bound to
the same column objects the caller passed, and rebuilt otherwise. This is what makes it safe
to reuse one fitted model across data slices (subgroups, train/test splits).

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
function get_or_build_engine(usage::Type{U}, deriv::Type{D}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov, backend::Symbol) where {U<:MarginsUsage, D<:DerivativeSupport}
    # Create comprehensive cache key including usage type, derivative support, backend, and all relevant factors
    cache_key = hash((
        usage, # Usage type for buffer sizing strategy
        deriv, # Derivative support for compile-time specialization
        backend, # Backend for derivative computation (:ad or :fd)
        model, # Model object (coefficients, structure, etc.)
        keys(data_nt), # Data structure (column names)
        vars, # Variables for derivatives
        typeof(model), # Model type for dispatch
        fieldnames(typeof(model)), # Model structure fields
        vcov # Covariance matrix specification (critical for caching!)
    ))

    # The key above does NOT identify the data — only its column names. A MarginsEngine is
    # bound to the columns it was built from (its derivative and contrast evaluators hold
    # those arrays), so a cache hit must be revalidated against the caller's data before it
    # can be reused. Without this, two datasets sharing a schema and a fitted model collide
    # on one entry and the second caller silently gets the first caller's rows.
    cached = get(ENGINE_CACHE, cache_key, nothing)
    if !isnothing(cached) && _engine_data_matches(cached, data_nt)
        return cached
    end

    engine::MarginsEngine = build_engine(usage, deriv, model, data_nt, vars, vcov, backend)
    ENGINE_CACHE[cache_key] = engine
    return engine
end

"""
    _engine_data_matches(engine::MarginsEngine, data_nt::NamedTuple) -> Bool

Whether a cached engine is bound to exactly the data the caller passed.

Identity (`===`) is the correct test rather than value equality: the engine's derivative
evaluator (`build_engine`, `src/engine/core.jl`) stores the data columns and dereferences
them at call time, so *the same array object* is precisely the condition under which the
cached engine computes on the caller's rows. It is also collision-free — the cache holds
the engine, which holds the arrays, so a cached column's address cannot be recycled by a
later allocation.

Consequence by design: alternating between two same-schema datasets rebuilds the engine on
each call. `build_engine` costs ~0.03-0.13 ms (1-3% of a single margins call, measured on a
16-parameter logit with interactions at n=20,000), which is the right price for not
returning results computed on the wrong data. Note that `_convert_numeric_to_float64`
returns the original array for `Float64`, `Bool` and categorical columns, so ordinary data
still hits the cache; only `Int`/`Float32` columns are re-converted per call and therefore
force a rebuild. `Union{Missing,T}` columns without missing values are narrowed by
`_narrow_missing_eltype`, which memoizes the narrowed copy per source array, so repeated
calls on the same data still hit the cache.
"""
function _engine_data_matches(engine::MarginsEngine, data_nt::NamedTuple)
    cached_nt = engine.data_nt
    keys(cached_nt) == keys(data_nt) || return false
    for k in keys(data_nt)
        getfield(cached_nt, k) === getfield(data_nt, k) || return false
    end
    return true
end


"""
    clear_engine_cache!()

Clear the engine cache, and the memo of converted data columns (`Union{Missing,T}`
columns narrowed, model-categorical columns converted to `CategoricalArray`). Useful for
memory management in long-running sessions.

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
    empty!(_NARROWED_COLUMNS)  # memoized narrowed / categorized data columns
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
