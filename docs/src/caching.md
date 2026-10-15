# Caching System (Internal Architecture)

> **Note:** This document describes internal implementation details of the Margins.jl caching system. The caching mechanism is **fully automatic** and requires no user intervention. This documentation is provided for developers, contributors, and users interested in understanding performance characteristics.

## User Perspective: Automatic Caching

**TL;DR for Users:** Margins.jl automatically caches compiled formula evaluators. The first call to `population_margins()` or `profile_margins()` with a given model compiles the evaluator (~1-10ms overhead), and subsequent calls reuse the cached version (microsecond overhead). No user action required.

```julia
# First call: includes compilation overhead
@time result1 = population_margins(model, data; type=:effects)  # ~5ms

# Second call: uses cached evaluator
@time result2 = population_margins(model, data; type=:effects)  # ~0.3ms (15x faster!)

# Different parameters create new cache entry
@time result3 = population_margins(model, data; type=:predictions)  # ~5ms (new compilation)
```

---

## Implementation Details

### Introduction

The computation of marginal effects involves repeated evaluation of compiled formula representations and their derivatives. To minimize computational overhead, Margins.jl implements a caching mechanism that preserves compiled evaluators and their associated memory buffers across multiple invocations. This section describes the architecture and implementation of this caching system.

## System Architecture

### Cache Structure

The package maintains a global dictionary that maps configuration hashes to pre-compiled engine instances:

```julia
const ENGINE_CACHE = Dict{UInt64, MarginsEngine}()
```

Each `MarginsEngine` encapsulates a compiled formula evaluator from FormulaCompiler.jl along with pre-allocated buffers necessary for computation. The cache key uniquely identifies the computational context through a hash of relevant parameters.

### Configuration Identification

The cache key encompasses all parameters that affect the computational procedure:

```julia
cache_key = hash((
    usage,                      # Computational pattern (population or profile)
    deriv,                      # Derivative requirement specification
    model,                      # Statistical model with fitted parameters
    keys(data_nt),             # Data structure specification
    vars,                      # Variables requiring derivative computation
    typeof(model),             # Model type for method dispatch
    fieldnames(typeof(model)), # Model structure specification
    vcov                       # Variance-covariance specification
))
```

This key identifies the data only by its column *names* — not by which arrays those names refer to. Because an engine's derivative and contrast evaluators store the data columns and dereference them at call time, the key alone is not sufficient to establish that a cached engine computes on the caller's rows. Retrieval therefore adds an explicit revalidation step.

### Retrieval Mechanism

A cache hit is accepted only when the cached engine is bound to the same column objects the caller passed; otherwise the engine is rebuilt and the entry replaced:

```julia
function get_or_build_engine(usage, deriv, model, data_nt, vars, vcov, backend)
    cache_key = hash(...)

    cached = get(ENGINE_CACHE, cache_key, nothing)
    if !isnothing(cached) && _engine_data_matches(cached, data_nt)
        return cached
    end

    engine = build_engine(usage, deriv, model, data_nt, vars, vcov, backend)
    ENGINE_CACHE[cache_key] = engine
    return engine
end
```

`_engine_data_matches` compares columns by identity (`===`) rather than by value: the same array object is precisely the condition under which the cached evaluators read the caller's data. Identity comparison is also collision-free, since the cache holds the engine, which holds the arrays.

The tradeoff is deliberate. Alternating between two datasets that share a schema rebuilds the engine on each call — measured at 0.03–0.13 ms, or 1–3% of a single margins call on a 16-parameter logit with interactions at n = 20,000. That is the correct price for never returning results computed on the wrong rows. Note also that `_convert_numeric_to_float64` returns the original array for `Float64`, `Bool` and categorical columns, so ordinary data continues to hit the cache; only `Int` and `Float32` columns are re-converted per call and therefore force a rebuild.

When a configuration is encountered for the first time, the system constructs a new engine instance. Subsequent requests with an identical configuration *and* the same data retrieve the existing instance without recompilation.

## Computational Implications

### Compilation Overhead Reduction

The construction of formula evaluators involves several computationally intensive operations:

1. Formula parsing and algebraic expansion
2. Categorical variable encoding and level extraction
3. Interaction term construction and indexing
4. Derivative graph construction for automatic differentiation
5. Memory buffer allocation and sizing

Through caching, these operations occur once per unique configuration rather than for each marginal effects computation. Consider the following timing comparison:

```julia
# Initial computation requires compilation
@time result1 = population_margins(model, data; vars=[:x1, :x2])
# 0.003456 seconds

# Subsequent computation uses cached evaluator
@time result2 = population_margins(model, data; vars=[:x1, :x2])
# 0.000234 seconds
```

The order-of-magnitude reduction in computation time reflects the elimination of compilation overhead.

### Memory Buffer Management

Each cached engine maintains pre-allocated buffers sized according to its usage pattern:

```julia
struct MarginsEngine{L, U, D}
    compiled::FormulaCompiler.UnifiedCompiled
    de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}

    # Pre-allocated computation buffers
    g_buf::Vector{Float64}           # Gradient storage
    gβ_accumulator::Vector{Float64}  # Accumulator for averaged effects
    η_buf::Vector{Float64}           # Linear predictor values
    row_buf::Vector{Float64}         # Design matrix row storage

    # Model parameters and metadata
    model::Any
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple
end
```

The buffer sizing strategy differs between usage patterns:

- **Population analysis**: Buffers sized for single-row operations to minimize memory footprint
- **Profile analysis**: Larger buffers accommodate multiple profile evaluations simultaneously

### Type Specialization

The caching system leverages Julia's type system to eliminate runtime dispatch overhead. The `DerivativeSupport` type parameter enables compile-time specialization:

```julia
# Continuous variables requiring derivatives
engine = get_or_build_engine(PopulationUsage, HasDerivatives, ...)
# Returns: MarginsEngine{..., PopulationUsage, HasDerivatives}

# Categorical variables without derivatives
engine = get_or_build_engine(ProfileUsage, NoDerivatives, ...)
# Returns: MarginsEngine{..., ProfileUsage, NoDerivatives}
```

This parametric typing eliminates Union type overhead in computational kernels.

## Practical Considerations

### Cache Entry Differentiation

The caching system creates distinct entries for configurations that differ in any computationally relevant aspect:

1. **Model variation**: Different fitted models, even with identical formulas, maintain separate cache entries due to distinct coefficient values.

2. **Covariance specification**: Each variance-covariance estimator (e.g., robust, clustered) requires its own cache entry as it affects standard error computation.

3. **Variable selection**: Different sets of variables for marginal effects analysis necessitate distinct derivative evaluators.

4. **Usage patterns**: Population and profile analyses employ different buffer sizing strategies and thus maintain separate cache entries.

5. **The data itself**: a cached engine is reused only when it is bound to the same column objects supplied by the caller, and is rebuilt otherwise. One fitted model may therefore be applied to different data — subgroup slices, train/test splits, resampled frames — within a single session without manual cache management.

### Memory Management

The memory footprint of cached engines scales with the number of unique configurations encountered during a session. For a model with p predictors, each engine requires approximately O(p²) memory, dominated by the variance-covariance matrix storage.

Typical usage patterns exhibit bounded cache growth:

- Standard analysis sessions: 1-10 unique configurations
- Complex comparative studies: 10-100 configurations
- Long-running services: Periodic cache clearing may be warranted

#### Internal Cache Management (Not Exported)

For advanced users and developers, the package provides internal utilities for cache management. These are **not exported** and require qualified access:

```julia
# Query cache statistics (internal function)
stats = Margins.get_cache_stats()
# Returns: (entries = n, memory_estimate = bytes, keys = [...])

# Clear cache to reclaim memory (internal function)
Margins.clear_engine_cache!()
```

**Note:** These functions are primarily for debugging and development. Most users will never need to call them directly, as Julia's garbage collection handles memory management automatically when MarginsEngine instances are no longer referenced.

### Performance Characteristics

Empirical measurements demonstrate the performance impact of caching for a generalized linear model with 10 predictors and 100,000 observations:

| Computation Type | Initial (ms) | Cached (ms) | Ratio |
|-----------------|--------------|-------------|-------|
| Population margins | 45 | 3 | 15:1 |
| Profile margins (10 profiles) | 8 | 0.5 | 16:1 |
| Bootstrap standard errors (1000 iterations) | 45,000 | 3,000 | 15:1 |

These measurements reflect the elimination of compilation overhead while preserving computational correctness.

## Implementation Details

### Thread Safety Considerations

The current implementation does not provide thread-safe cache access. Concurrent access from multiple threads may result in race conditions. For parallel computation, two strategies are available:

1. Pre-populate the cache before initiating parallel computation
2. Implement thread-local caches (not currently supported)

### Cache Invalidation

Passing different data requires no action: retrieval revalidates the cached engine's data binding and rebuilds when it does not match (see "Retrieval Mechanism").

The cache does not, however, detect changes to model coefficients that occur outside the standard fitting procedures. Manual coefficient modification requires explicit cache clearing:

```julia
# Fit model and compute margins
model = lm(@formula(y ~ x), data)
result1 = population_margins(model, data)

# Manual coefficient update (atypical scenario)
model.pp.beta0 .= new_coefficients

# Clear cache to ensure consistency (internal function)
Margins.clear_engine_cache!()
result2 = population_margins(model, data)
```

### Backward Compatibility

The system provides automatic derivative support detection for legacy code:

```julia
# Explicit specification (recommended)
get_or_build_engine(PopulationUsage, HasDerivatives, model, data_nt, vars, vcov)

# Automatic detection (backward compatible)
get_or_build_engine(PopulationUsage, model, data_nt, vars, vcov)
```

The automatic detection examines the data types to determine whether derivative computation is required.

## Theoretical Foundation

The caching system exploits the mathematical property that marginal effects computation for a given model configuration involves deterministic transformations of the design matrix and coefficient vector. For a model with linear predictor η = Xβ, the marginal effect computation:

∂E[y|X]/∂xⱼ = g'(η) × βⱼ

depends only on the model structure, coefficients, and link function. By caching the compiled representation of these transformations, the system avoids redundant symbolic and numerical preprocessing while preserving exact numerical equivalence.

## Future Directions

Several enhancements to the caching system are under consideration:

1. **Least-recently-used eviction**: Automatic cache size management through LRU policies
2. **Thread-local storage**: Safe concurrent access through thread-specific caches
3. **Weak references**: Enable garbage collection of unused engines while preserving active ones
4. **Persistent caching**: Serialization of compiled engines for cross-session reuse

These enhancements would extend the applicability of the caching system to more diverse computational contexts while maintaining the current guarantees of numerical correctness.

## Related Documentation

- [Computational Architecture](computational_architecture.md): FormulaCompiler.jl integration and evaluation strategies
- [Performance Guide](performance.md): Computational complexity analysis and optimization strategies
- [API Reference](api.md): Cache management function specifications