# engine/caching.jl - Bounded, thread-safe pool of MarginsEngine instances

# A cache slot: the engines built for one cache key (usage, derivative support, backend,
# model, column names, vars, vcov). `idle` engines are free to be checked out; `checked_out`
# counts engines currently in exclusive use by a running call. Usually a slot holds a
# single engine; it holds more only while several calls on the same key run concurrently.
mutable struct _EngineSlot
    idle::Vector{MarginsEngine}
    checked_out::Int
    last_used::UInt64
end

"""
    ENGINE_CACHE

Process-wide pool of `MarginsEngine`s, keyed by configuration hash (see
`get_or_build_engine`). Each key maps to an `_EngineSlot` of idle engines. A Margins call
checks an engine out for its exclusive use (`_checkout_engine`) and returns it when done
(`_checkin_engine!`), so concurrent calls never share an engine's mutable buffers.

The pool is bounded: at most `engine_cache_capacity()` engines (idle + checked out) are
kept, evicting idle engines of the least recently used keys first; checked-out engines
are never evicted. All access goes through `_ENGINE_CACHE_LOCK`.
"""
const ENGINE_CACHE = Dict{UInt64, _EngineSlot}()
const _ENGINE_CACHE_LOCK = ReentrantLock()
const _DEFAULT_ENGINE_CACHE_CAPACITY = 8
const _ENGINE_CACHE_CAPACITY = Ref(_DEFAULT_ENGINE_CACHE_CAPACITY)
const _ENGINE_CACHE_TICK = Ref{UInt64}(0)

"""
    Margins.set_engine_cache_capacity!(n::Integer) -> Int

Set the maximum number of engines Margins keeps cached (default 8) and return the
previous value. Each cached engine holds references to the data columns it was built
from, so the capacity bounds how many datasets the cache can keep alive. Idle engines
beyond the new capacity are evicted immediately (least recently used first). `n = 0`
disables caching: every call builds a fresh engine.

The default comfortably covers the common pattern of repeated calls on a few
models/datasets (each `(model, data, type, vars, vcov, backend)` combination uses one
engine, plus one per additional concurrent call on it).
"""
function set_engine_cache_capacity!(n::Integer)
    n >= 0 || throw(ArgumentError("engine cache capacity must be ≥ 0, got $n"))
    lock(_ENGINE_CACHE_LOCK)
    try
        old = _ENGINE_CACHE_CAPACITY[]
        _ENGINE_CACHE_CAPACITY[] = Int(n)
        _evict_engines_over_capacity!()
        return old
    finally
        unlock(_ENGINE_CACHE_LOCK)
    end
end

"""
    Margins.engine_cache_capacity() -> Int

The current engine cache capacity (see `set_engine_cache_capacity!`).
"""
engine_cache_capacity() = _ENGINE_CACHE_CAPACITY[]

function _engine_cache_key(usage, deriv, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov, backend::Symbol)
    return hash((
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
end

# Number of engines the pool accounts for (idle + checked out). Caller holds the lock.
function _engine_count()
    n = 0
    for slot in values(ENGINE_CACHE)
        n += length(slot.idle) + slot.checked_out
    end
    return n
end

# Evict idle engines, least recently used slot first, until the pool is within capacity.
# Checked-out engines are never evicted (if they alone exceed the capacity, the excess is
# dropped when they are checked in). Caller holds the lock.
function _evict_engines_over_capacity!()
    cap = _ENGINE_CACHE_CAPACITY[]
    n = _engine_count()
    while n > cap
        victim_key = UInt64(0); victim_tick = typemax(UInt64); found = false
        for (k, slot) in ENGINE_CACHE
            if !isempty(slot.idle) && slot.last_used <= victim_tick
                victim_key = k; victim_tick = slot.last_used; found = true
            end
        end
        found || break
        slot = ENGINE_CACHE[victim_key]
        popfirst!(slot.idle)
        n -= 1
        isempty(slot.idle) && slot.checked_out == 0 && delete!(ENGINE_CACHE, victim_key)
    end
    return nothing
end

"""
    _checkout_engine(usage, deriv, model, data_nt, vars, vcov, backend) -> (engine, key)

Take an engine for exclusive use by the current call: an idle cached engine bound to
exactly `data_nt` (`_engine_data_matches`) if there is one, otherwise a newly built one.
The caller must return it with `_checkin_engine!(key, engine)` in a `finally` block.

Single-threaded, repeated calls on the same data reuse one engine; the only per-call
overhead is a lock acquire/release on check-out and check-in. The engine is built outside
the lock, so a slow build does not block other calls.
"""
function _checkout_engine(usage::Type{U}, deriv::Type{D}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov, backend::Symbol) where {U<:MarginsUsage, D<:DerivativeSupport}
    # The key does NOT identify the data — only its column names. A MarginsEngine is bound
    # to the columns it was built from (its derivative and contrast evaluators hold those
    # arrays), so an idle engine is reused only if `_engine_data_matches` the caller's data.
    # Without this, two datasets sharing a schema and a fitted model would collide and the
    # second caller would silently get the first caller's rows.
    key = _engine_cache_key(usage, deriv, model, data_nt, vars, vcov, backend)
    lock(_ENGINE_CACHE_LOCK)
    try
        tick = (_ENGINE_CACHE_TICK[] += 1)
        slot = get(ENGINE_CACHE, key, nothing)
        if !isnothing(slot)
            idle = slot.idle
            for i in length(idle):-1:1
                e = idle[i]
                if _engine_data_matches(e, data_nt)
                    deleteat!(idle, i)
                    slot.checked_out += 1
                    slot.last_used = tick
                    return (e, key)
                end
            end
        else
            slot = _EngineSlot(MarginsEngine[], 0, tick)
            ENGINE_CACHE[key] = slot
        end
        # Miss: reserve a place for the engine about to be built, and make room now so that
        # a stale engine (e.g. bound to the previous bootstrap replicate) is released before
        # the new one is built.
        slot.checked_out += 1
        slot.last_used = tick
        _evict_engines_over_capacity!()
    finally
        unlock(_ENGINE_CACHE_LOCK)
    end
    engine = try
        build_engine(usage, deriv, model, data_nt, vars, vcov, backend)
    catch
        _release_reservation!(key)
        rethrow()
    end
    return (engine::MarginsEngine, key)
end

function _release_reservation!(key::UInt64)
    lock(_ENGINE_CACHE_LOCK)
    try
        slot = get(ENGINE_CACHE, key, nothing)
        isnothing(slot) && return nothing
        slot.checked_out = max(slot.checked_out - 1, 0)
        isempty(slot.idle) && slot.checked_out == 0 && delete!(ENGINE_CACHE, key)
    finally
        unlock(_ENGINE_CACHE_LOCK)
    end
    return nothing
end

"""
    _checkin_engine!(key, engine)

Return an engine taken with `_checkout_engine` to the idle pool. Idle engines of the same
key bound to *different* data are dropped (a key keeps engines for the most recent data
only, so sequentially processing many same-schema datasets does not accumulate engines),
then the pool is trimmed to capacity. If the cache was cleared while the engine was
checked out, the engine is discarded.
"""
function _checkin_engine!(key::UInt64, engine::MarginsEngine)
    lock(_ENGINE_CACHE_LOCK)
    try
        slot = get(ENGINE_CACHE, key, nothing)
        isnothing(slot) && return nothing  # cleared meanwhile
        slot.checked_out = max(slot.checked_out - 1, 0)
        idle = slot.idle
        for i in length(idle):-1:1
            _engine_data_matches(idle[i], engine.data_nt) || deleteat!(idle, i)
        end
        push!(idle, engine)
        slot.last_used = (_ENGINE_CACHE_TICK[] += 1)
        _evict_engines_over_capacity!()
    finally
        unlock(_ENGINE_CACHE_LOCK)
    end
    return nothing
end

"""
    get_or_build_engine(usage, deriv, model, data_nt, vars, vcov, backend) -> MarginsEngine

Return a cached engine for this configuration and data, building (and caching) one if
needed. Internal / testing helper: the returned engine is NOT held exclusively, so it must
not be used while other tasks may run Margins calls. Margins' own entry points use
`_checkout_engine` / `_checkin_engine!` instead.

# Arguments
- `usage::Type{<:MarginsUsage}`: Usage pattern (PopulationUsage or ProfileUsage)
- `deriv::Type{<:DerivativeSupport}`: Derivative support (HasDerivatives or NoDerivatives)
- `model`: Fitted statistical model (GLM.jl, etc.)
- `data_nt::NamedTuple`: Data in columntable format
- `vars::Vector{Symbol}`: Variables for derivative analysis
- `vcov`: Covariance estimator (function or CovarianceMatrices estimator)
- `backend::Symbol`: `:ad` or `:fd`

# Cache Key Strategy
The key hashes the usage type, derivative support, backend, model, column names, vars,
model type and vcov. It identifies the data only by its column *names*, so a hit is
additionally revalidated with `_engine_data_matches` — a cached engine is reused only when
it is bound to the same column objects the caller passed, and rebuilt otherwise. This is
what makes it safe to reuse one fitted model across data slices (subgroups, train/test
splits).
"""
function get_or_build_engine(usage::Type{U}, deriv::Type{D}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov, backend::Symbol) where {U<:MarginsUsage, D<:DerivativeSupport}
    (engine, key) = _checkout_engine(usage, deriv, model, data_nt, vars, vcov, backend)
    _checkin_engine!(key, engine)
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
returning results computed on the wrong data. Data preparation keeps column identity stable
across calls on the same data: `Float64`, `Bool` and categorical columns are passed through,
and the copies it makes (`Int`/`Float32` → `Float64`, `Union{Missing,T}` narrowing,
model-categorical conversion) are memoized per source array (`_memoized_transform`), so
repeated calls on the same data hit the cache.
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

Empty the engine cache and the memo of converted data columns (`Union{Missing,T}` columns
narrowed, integer/Float32 columns converted to `Float64`, model-categorical columns
converted to `CategoricalArray`), releasing the data they reference. Engines checked out
by calls running at that moment finish normally and are then discarded. The cache is
bounded (`set_engine_cache_capacity!`), so this is only needed to release memory sooner.
Also clears FormulaCompiler's compiled-formula cache (`FormulaCompiler.clear_compiled_cache!()`).
"""
function clear_engine_cache!()
    lock(_ENGINE_CACHE_LOCK)
    try
        empty!(ENGINE_CACHE)
    finally
        unlock(_ENGINE_CACHE_LOCK)
    end
    lock(_STATE_LOCK)
    try
        empty!(_NARROWED_COLUMNS)  # memoized narrowed / converted / categorized data columns
    finally
        unlock(_STATE_LOCK)
    end
    # FormulaCompiler's compiled-formula cache also holds compiled models
    FormulaCompiler.clear_compiled_cache!()
    return nothing
end

"""
    get_cache_stats() -> NamedTuple

Return cache statistics for monitoring and debugging.

# Returns
- `NamedTuple` with cache metrics:
  - `entries`: Number of cached engines (idle + checked out)
  - `keys`: Number of distinct cache keys (configurations)
  - `capacity`: Current capacity (`engine_cache_capacity()`)
  - `memory_estimate`: Shallow size in bytes of the idle engine objects (not their buffers or data)

# Examples
```julia
stats = Margins.get_cache_stats()
println("Cached engines: ", stats.entries, " of ", stats.capacity)
```
"""
function get_cache_stats()
    lock(_ENGINE_CACHE_LOCK)
    try
        entries = _engine_count()
        memory_estimate = 0
        for slot in values(ENGINE_CACHE), e in slot.idle
            memory_estimate += sizeof(e)
        end
        return (
            entries = entries,
            keys = length(ENGINE_CACHE),
            capacity = _ENGINE_CACHE_CAPACITY[],
            memory_estimate = memory_estimate,
        )
    finally
        unlock(_ENGINE_CACHE_LOCK)
    end
end

"""
    _cached_engines() -> Vector{MarginsEngine}

All idle cached engines (testing/debugging helper).
"""
function _cached_engines()
    lock(_ENGINE_CACHE_LOCK)
    try
        return MarginsEngine[e for slot in values(ENGINE_CACHE) for e in slot.idle]
    finally
        unlock(_ENGINE_CACHE_LOCK)
    end
end
