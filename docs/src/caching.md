# Caching and Per-Call State (Internal Architecture)

> **Note:** This document describes internal implementation details. Nothing here requires user action. It is provided for developers, contributors, and users interested in performance characteristics and thread safety.

## User Perspective

**TL;DR for Users:** Margins.jl keeps no cache of its own. Every call to `population_margins()`, `profile_margins()` or `second_differences()` builds its own engine, prepares the data columns it needs, validates its inputs and evaluates `vcov`, so each result reflects the model, data and covariance as they are *at that call*. The expensive step, compiling the model's formula, is cached by FormulaCompiler.jl, so only the first call on a model pays for it.

```julia
# First call on a model: compiles the formula (cached by FormulaCompiler) and Julia code
@time result1 = population_margins(model, data; type=:effects)

# Later calls reuse the compiled formula; everything else is rebuilt (a fraction of a ms)
@time result2 = population_margins(model, data; type=:effects)

# Refitting in place, editing a vcov matrix in place, or passing other data: always picked up
refit!(lmm, new_y)                          # MixedModels
result3 = population_margins(lmm, data)     # uses the new coefficients
```

---

## What Is Cached, and What Is Not

### FormulaCompiler's compiled-formula cache (the only shared state)

`FormulaCompiler.get_or_compile_formula(model, data)` keeps a bounded, process-wide cache of compiled formulas, one small entry per model and column-type signature (the column *names* and a semantic type per column, not the values). The compiled operations depend only on the formula's structure (terms, contrasts, categorical levels), never on the coefficients or the data values, so a cached entry is valid for every call on that model. The cache has its own lock, and compiles *outside* the lock so a slow compilation does not block other callers.

Margins reads the shared compiled object only for read-only inspection (its operations, continuous variables and categorical levels). Anything that evaluates rows uses a private copy with its own scratch buffer (`_private_compiled`), because evaluation writes into the compiled object's scratch vector.

### Built on every call

- **The engine** (`MarginsEngine`, `build_engine`): the private compiled formula, the derivative and contrast evaluators bound to this call's data columns, the coefficient vector `coef(model)`, the covariance matrix Σ and the pre-allocated buffers. Building it costs a fraction of a millisecond once the formula is compiled.
- **The data columns**: `Union{Missing,T}` columns without missing values are narrowed to `T`, `Int`/`Float32` columns are converted to `Float64`, and columns the model treats as categorical are converted to `CategoricalArray`. `Float64`, `Bool` and categorical columns are passed through without copying. The copies are made per call, so in-place edits of the source are always seen.
- **Validation** of the model, the parameters and `vcov`.
- **The covariance matrix Σ**: `vcov` is evaluated exactly once per entry-point call, when the call's options are normalized into a request (`_normalize_request`, via `validate_vcov_parameter`), and the matrix is passed to `build_engine`. For a CovarianceMatrices estimator (`HC1()`, `CR1(cluster)`, ...) this is the estimator's own cost on every call. `second_differences(model, data, ...)`, which computes one AME per profile point or shift, builds one request for the whole call, so it too evaluates `vcov` once. (The degrees of freedom do not depend on which `vcov` is passed: they are a property of the model class, see the API reference.)

### Why there is no engine cache

Through the 3.0 development cycle (and in 2.x) Margins kept a process-wide cache of engines keyed by a hash of the model, column names, variables and `vcov`, plus a memo of converted data columns and a cache of validated models. Measured warm, the cache saved at most about 0.1 ms per call (FormulaCompiler's cache already held the expensive state), and its keys were wrong in ways that returned stale results:

- the key used the model object, so after MixedModels `refit!` (which mutates the model) the old estimates came back;
- a `vcov` matrix was keyed by `hash(V)`, which samples the entries of a large array, so an in-place edit of an unsampled entry returned the old standard errors (most single-entry edits at p = 201);
- the validated-models cache used the same sampled hash, so an edited matrix that was no longer valid skipped validation.

Removing the cache fixed all three and deleted the locks, the engine pool and the column memo that existed only to serve it.

## Engine Structure

Each engine holds pre-allocated buffers sized for its usage pattern:

```julia
struct MarginsEngine{L, U, C, E, CE}
    compiled::C                      # private copy of the compiled formula (own scratch)
    de::E                            # derivative evaluator, or nothing
    contrast::CE                     # categorical contrast evaluator, or nothing

    # Pre-allocated computation buffers
    g_buf::Vector{Float64}           # Gradient storage
    gβ_accumulator::Vector{Float64}  # Accumulator for averaged effects
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

- **Population analysis**: buffers sized for single-row operations, reused across the rows of the call, so the per-row loops do not allocate.
- **Profile analysis**: larger buffers that accommodate multiple profile evaluations.

The evaluator types are type parameters (`E`, `CE`), so the kernels are compiled for the exact evaluators and a warm AME call makes no dynamic dispatch through them (the continuous-effects accumulator allocates 0 bytes per call, for every measure and both backends). Only population engines carry evaluators over the data rows; profile engines evaluate on the reference grid, and the scenarios/groups path builds evaluators per context, so their engines have `de = contrast = nothing`:

```julia
engine = build_engine(PopulationUsage, model, data_nt, vars, Σ, :ad)
# MarginsEngine{..., PopulationUsage, ..., ADEvaluator{...}, ContrastEvaluator{...}}

engine = build_engine(ProfileUsage, model, data_nt, vars, Σ, :ad)
# MarginsEngine{..., ProfileUsage, ..., Nothing, Nothing}
```

`build_engine` also accepts a function of the model or a CovarianceMatrices estimator in place of Σ and evaluates it; the entry points always pass the matrix.

## Memory

An engine lives only for the call that built it, so Margins keeps no dataset alive between calls. FormulaCompiler's compiled cache holds references to the models it was compiled for (and, through a model's frame, its estimation data), bounded by `FormulaCompiler.compiled_cache_capacity()`. A loop that fits a new model per iteration (bootstrap, simulation, cross-validation) retains at most that many models.

```julia
# Empty FormulaCompiler's compiled-formula cache (Margins has no cache of its own)
Margins.clear_engine_cache!()
```

`clear_engine_cache!()` keeps its 2.x name so that existing code (for example benchmarks that clear caches before every sample) keeps working; ordinary use never needs it. The 2.x/early-3.0 cache controls `get_cache_stats`, `set_engine_cache_capacity!`, `engine_cache_capacity`, `get_or_build_engine`, `ENGINE_CACHE` and `VALIDATED_MODELS_CACHE` were removed (see the CHANGELOG for the migration).

## Thread Safety

Concurrent Margins calls are safe. `population_margins`, `profile_margins` and the
functions built on them (second differences, contrasts) may be called from several tasks
or threads at once, on the same model and data or on different ones, and each returns
exactly what it returns when called alone.

This holds because no two running calls share mutable state:

- Each call builds its own `MarginsEngine`, whose mutable scratch buffers (gradient
  accumulators, model-row and Jacobian buffers, counterfactual state in its FormulaCompiler
  evaluators) no other call can see.
- FormulaCompiler's cached compiled formula is shared per model and schema and holds a
  scratch buffer. Margins never evaluates through the shared object: every engine, and
  every reference-grid evaluation, uses a copy of the compiled operations with its own
  scratch buffer. The operations' only mutable state, FormulaCompiler's categorical level
  cache, is published atomically (FormulaCompiler ≥ 1.5.0).
- The only process-wide table that holds computation state, FormulaCompiler's
  compiled-formula cache, is guarded by its own lock. (Margins' own process-wide state is the
  display-digit settings and a locked memo of model-type names for result metadata, keyed by
  type, which cannot go stale.)

```julia
results = Vector{Any}(undef, 200)
Threads.@threads for b in 1:200
    d = bootstrap_sample(data, b)              # your resampling function
    m = glm(formula, d, Binomial(), LogitLink())
    results[b] = DataFrame(population_margins(m, d; vars=[:x]))
end
```

Objects obtained from FormulaCompiler directly (`compile_formula`,
`derivativeevaluator`, `contrastevaluator`, ...) and engines from the internal
`Margins.build_engine` are *not* thread-safe: they carry their own scratch buffers, so
create one per task when using them in threaded code.

## Related Documentation

- [Computational Architecture](computational_architecture.md): FormulaCompiler.jl integration and evaluation strategies
- [Performance Guide](performance.md): Computational complexity analysis and optimization strategies
- [API Reference](api.md): Function specifications
