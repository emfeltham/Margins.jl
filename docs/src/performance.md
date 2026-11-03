# Performance Guide

*Computational characteristics and optimization strategies*

## Conceptual Framework

### Performance Design Principles

Margins.jl achieves computational efficiency through architectural design that respects the fundamental mathematical structure of marginal effects analysis. Performance optimization preserves statistical correctness while exploiting the distinct computational requirements of population versus profile analysis.

### Algorithmic Complexity Characteristics

- **Profile Analysis**: cost dominated by the reference-grid size, plus a small per-call O(n) part
- **Population Analysis**: O(n) linear scaling with optimized per-observation computational cost
- **Statistical Integrity**: All performance optimizations maintain mathematical validity

## Implementation Performance

## Performance Characteristics

### Profile Analysis: Cost Dominated by the Grid

Profile margins evaluate the model only at the reference-grid rows, so their cost is dominated by the grid size. Each call also does a small amount of O(n) work on the data: integer columns are copied to `Float64` (about 8 bytes per row per column), the derived-column check scans the columns, and variables the grid leaves out are filled in with their typical values over the estimation sample (weighted when the model has weights). For FixedEffectModels the data is also subset to the estimation sample.

Measured on a 65-coefficient logit (Apple M4 Pro, Margins 3.0):

| Rows | Profile predictions (4 grid rows) | Profile effects (4 grid rows) | AME, all variables |
|------|-----------------------------------|-------------------------------|--------------------|
| 5,000 | 0.27 ms, 0.78 MiB | 0.69 ms | 18 ms, 0.95 MiB |
| 50,000 | 0.59 ms, 3.2 MiB | 0.91 ms | 182 ms, 3.4 MiB |
| 500,000 | 1.8 ms, 27 MiB | 3.4 ms | 1.8 s, 27 MiB |

```julia
using BenchmarkTools, Margins

# Cost grows with the number of grid rows; the per-call O(n) part is small
@btime profile_margins($model, $data_1k, means_grid($data_1k); type=:effects)
@btime profile_margins($model, $data_100k, means_grid($data_100k); type=:effects)

scenarios = cartesian_grid(x1=[0,1,2], x2=[10,20,30], group=["A","B"])  # 18 profiles
@btime profile_margins($model, $huge_data, $scenarios)
```

**Why this matters**: Profile analysis stays fast on large datasets: going from 5,000 to 500,000 rows multiplies the data by 100 and the profile time by about 5 to 7 in the measurements above.

### Population Analysis: Optimized O(n) Scaling

Population margins scale linearly with optimized per-row costs:

```julia
# Linear scaling with low per-row computational cost
@btime population_margins($model, $data_1k)    # scales with dataset size
@btime population_margins($model, $data_10k)   # with efficient per-row processing  
@btime population_margins($model, $data_100k)  # minimal allocation overhead

# The per-row loop allocates nothing; with Float64 data a call allocates a constant amount
# (the result, Σ, the engine's buffers), the same at every n (both backends; tens of KB for a
# small model). Integer columns add a per-call Float64 copy (about 8 bytes per row per column),
# as do the columns a scenario replaces.
@allocated population_margins(model, data_1k; backend=:fd)    # c bytes
@allocated population_margins(model, data_10k; backend=:fd)   # the same c
@allocated population_margins(model, data_100k; backend=:fd)  # the same c

@allocated population_margins(model, data_1k; backend=:ad)    # c′ bytes
@allocated population_margins(model, data_100k; backend=:ad)  # the same c′
```

**Why this matters**: Apart from those column copies, population analysis has a **constant allocation footprint** (it does not grow with the number of rows) while delivering consistent per-row performance. Categorical variables follow the same rule: their observed levels come from the reference codes, not from a per-row pass that allocates.

## Dataset Size Guidelines

### Performance Expectations by Scale

| Dataset Size | Population Margins | Profile Margins | Recommended Workflow |
|--------------|-------------------|-----------------|---------------------|
| **< 1k** | Fast | Fast (grid-dominated) | Use either approach freely |
| **1k-10k** | Fast | Fast (grid-dominated) | Profile preferred for scenarios |
| **10k-100k** | Scales linearly | Fast (grid-dominated) | Profile for exploration, population for final analysis |
| **100k-1M** | Scales appropriately | Small O(n) part becomes visible (ms) | Profile strongly preferred |
| **> 1M** | Scales with dataset size | Small O(n) part becomes visible | Profile analysis, selective population |

### Backend Selection by Use Case

For detailed backend selection guidance including domain-sensitive functions and reliability considerations, see **[Backend Selection Guide](backend_selection.md)**.

**Quick summary:**
- **`:ad`** - Recommended default; machine-precision accuracy, zero allocation, handles all functions
- **`:fd`** - Alternative; zero allocation, numerical approximation, good for simple formulas

```julia
# Recommended configuration
population_margins(model, data; backend=:ad, scale=:response)

# Profile analysis (also uses :ad by default)
profile_margins(model, data, means_grid(data); backend=:ad, scale=:response)

# Domain-sensitive functions (log, sqrt) - AD recommended
population_margins(model, data; backend=:ad)  # Recommended for log(x), sqrt(x)
```

## Optimization Principles

### Core Performance Philosophy

**Statistical Correctness First**: Performance optimizations maintain statistical validity
- Delta-method standard errors use full covariance matrix
- All gradient computations maintain mathematical precision
- Bootstrap validation ensures statistical accuracy
- Never change estimators, gradients, or SE math to "optimize"

**Zero-Allocation Patterns**: Eliminate unnecessary memory allocations
- Pre-allocated buffers reused across computations
- FormulaCompiler.jl provides zero-allocation evaluation primitives
- Row loops allocate nothing, whatever the dataset size
- Per-call allocations do not grow with n, apart from per-call column copies (integer columns converted to Float64, columns a scenario replaces)

**Computational Efficiency**: Optimize hot paths without changing methodology
- Compiled formula evaluation with caching
- Efficient gradient accumulation patterns
- Scalar operations over broadcast temporaries
- Zero dynamic growth: avoid `push!` in hot paths; size outputs up-front

### Backend Performance Characteristics

#### Automatic Differentiation (`:ad`) - **RECOMMENDED DEFAULT**
```julia
# Zero allocation per row after warmup (a call allocates a constant amount)
population_margins(model, data; backend=:ad)
```

**Advantages**:
- Zero allocation per row after warmup
- Machine precision accuracy (exact derivatives)
- Robust domain handling (handles log, sqrt, 1/x safely)
- Suitable for complex formulas

**Use cases**: Most applications - provides good performance and reliability

#### Finite Differences (`:fd`)
```julia
# Zero allocation per row after warmup (a call allocates a constant amount)
population_margins(model, data; backend=:fd)
```

**Advantages**:
- Zero allocation per row
- Simple numerical implementation
- Good accuracy for well-conditioned functions

**Use cases**: Simple linear formulas where marginal speed differences matter

## Memory Management

### Allocation Patterns

The per-row kernels allocate nothing. Apart from per-call copies of columns that need conversion (integer columns copied to `Float64`, columns a scenario replaces), each call allocates an amount that does not grow with the data:

```julia
# Profile margins: small, but not constant: the column copies and typical-value
# computations are O(n) (0.78, 3.2, 27 MiB at 5K, 50K, 500K rows in the table above)
@allocated profile_margins(model, small_data, means_grid(small_data))
@allocated profile_margins(model, large_data, means_grid(large_data))

# Population margins on Float64 data: the same constant at every n after warmup (both backends)
@allocated population_margins(model, data_1k; backend=:fd)   # c bytes
@allocated population_margins(model, data_10k; backend=:fd)  # the same c
@allocated population_margins(model, data_1k; backend=:ad)   # c′ bytes
@allocated population_margins(model, data_10k; backend=:ad)  # the same c′
```

### Memory Efficiency Best Practices

#### For Large Datasets
```julia
# Use profile analysis for exploration (cost dominated by the grid)
scenarios = cartesian_grid(x1=[-1, 0, 1], treatment=[0, 1])
results = profile_margins(model, large_data, scenarios)

# Use population analysis with zero-allocation backends
key_effects = population_margins(model, large_data; vars=[:treatment], backend=:ad)  # Recommended
# OR
key_effects = population_margins(model, large_data; vars=[:treatment], backend=:fd)  # Also zero allocation
```

#### For Batch Processing
```julia
# Process multiple models (allocation-free row loops)
models = [model1, model2, model3]
results = []

for model in models
    # The row loops allocate nothing (either backend)
    result = population_margins(model, data; backend=:ad)  # Recommended
    push!(results, DataFrame(result))
end
```

## Performance Best Practices

### High-Performance Usage Patterns

For optimal performance in production environments, follow these proven patterns:

#### Compilation and Caching
```julia
# FormulaCompiler caches the compiled formula (keyed by model and data schema), so repeated
# calls do not recompile. Margins keeps no cache of its own: each call builds its engine and
# evaluators (cheap), so results always reflect the current model, data and vcov.
result1 = population_margins(model, data; type=:effects)     
result2 = profile_margins(model, data, means_grid(data); type=:effects)

# Avoid: Forcing recompilation in loops
for subset in data_subsets
    # Each call may recompile unnecessarily
    result = population_margins(fit_model(subset), subset)  
end
```

#### Memory-Efficient Data Processing
```julia
# Good: Pre-allocate result structures for known sizes
n_effects = length(vars) * length(scenarios)
result_buffer = DataFrame(
    term = Vector{String}(undef, n_effects),
    estimate = Vector{Float64}(undef, n_effects),
    se = Vector{Float64}(undef, n_effects)
)

# Good: Use scalar operations in hot paths
for i in eachindex(estimates)
    μ = GLM.linkinv(link, η[i])      # Scalar operation
    se[i] = sqrt(gradients[i]' * Σ * gradients[i])
end

# Avoid: Growing DataFrames with push! in loops
results = DataFrame()
for scenario in scenarios
    result = population_margins(model, scenario_data)
    push!(results, DataFrame(result))  # Expensive growth
end
```

#### FormulaCompiler Integration Patterns
```julia
# Good: Let FormulaCompiler handle the optimization
# Use built-in primitives for zero-allocation paths
population_margins(model, data; backend=:fd)  # Uses optimized accumulation

# Good: Cache compiled objects for batch processing
models = [model1, model2, model3]

for model in models
    # Compilation is cached automatically by model signature
    result = population_margins(model, data; backend=:ad)
end
```

### Performance Validation

#### Checking Allocation Patterns
```julia
# Verify zero-allocation performance
using BenchmarkTools

# Both backends: a constant per call after warmup, the same for any number of rows
@allocated population_margins(model, data; backend=:ad)
@allocated population_margins(model, data; backend=:fd)

# Profile margins allocate little, but the per-call column work is O(n)
@allocated profile_margins(model, small_data, means_grid(small_data))  # Small
@allocated profile_margins(model, large_data, means_grid(large_data))  # Larger, still small
```

#### Performance Monitoring
```julia
# Production monitoring pattern
function monitored_margins(model, data; max_alloc_kb=10, kwargs...)
    alloc_before = Base.gc_num().poolalloc
    
    result = population_margins(model, data; kwargs...)
    
    alloc_after = Base.gc_num().poolalloc
    alloc_kb = (alloc_after - alloc_before) / 1024
    
    if alloc_kb > max_alloc_kb
        @warn "Excessive allocation detected: $(alloc_kb)KB"
    end
    
    return result
end
```

## Troubleshooting Performance Issues

### Diagnostic Tools

#### Memory Allocation Checking
```julia
using BenchmarkTools

# Check allocation patterns
@allocated population_margins(model, data)  # Constant across dataset sizes for Float64 data

# Benchmark performance
@btime population_margins($model, $data)     # Timing analysis
```

#### Performance Profiling
```julia
# Profile hot paths (advanced)
using Profile

@profile for i in 1:100
    population_margins(model, data; backend=:fd)
end
Profile.print()
```

### Common Issues and Solutions

#### Issue: Profile margins slower than expected
**Diagnosis**: Reference grid specification or DataFrame dispatch issues
**Solution**: Use proper reference grid builders
```julia
# Correct: cost dominated by the grid size
profile_margins(model, data, means_grid(data); type=:effects)
profile_margins(model, data, cartesian_grid(x=[0,1,2]); type=:effects)

# Avoid: Improper reference grid specification
# Always use reference grid builders or explicit DataFrames
```

#### Issue: Population margins allocating excessively
**Diagnosis**: Hot loop allocation patterns or data format issues
**Solution**: Check backend and use efficient data formats
```julia
# Efficient: Both backends should be zero allocation
result = population_margins(model, data; backend=:ad)  # Recommended
result = population_margins(model, data; backend=:fd)  # Also zero allocation

# Efficient data format
data_nt = Tables.columntable(data)  # Convert once for multiple analyses
result = population_margins(model, data_nt; backend=:ad)
```

#### Issue: Inconsistent performance across runs
**Diagnosis**: Compilation effects, memory pressure, or GC interference
**Solution**: Proper warmup and consistent configuration
```julia
# Proper benchmarking protocol
# 1. Warmup run
population_margins(model, small_sample)  

# 2. Clear compilation effects  
GC.gc()

# 3. Consistent benchmark
@btime population_margins($model, $data; backend=:ad)  
```

#### Issue: Memory allocation growing with dataset size
**Diagnosis**: O(n) allocation pattern indicating performance regression
**Solution**: Verify zero-allocation backends and check for loops
```julia
# Expected (Float64 data): constant allocation across dataset sizes
@allocated population_margins(model, data_1k; backend=:ad)    # c bytes
@allocated population_margins(model, data_10k; backend=:ad)   # should be the same c
@allocated population_margins(model, data_100k; backend=:ad)  # should be the same c

# If allocations grow with n:
# 1. Integer columns are copied to Float64 on every call (8 bytes per row per column):
#    convert them once beforehand if this matters
# 2. Check backend selection (:ad and :fd both have zero-allocation row loops)
# 3. Verify data format (Tables.jl-compatible)
# 4. Check for custom vcov functions that may allocate
```

#### Issue: Slow compilation on first run
**Diagnosis**: Normal FormulaCompiler compilation overhead
**Solution**: Accept first-run cost, subsequent runs benefit from caching
```julia
# Expected pattern:
@time population_margins(model, data)        # Slower (compilation)
@time population_margins(model, data)        # Faster (cached)

# For production: accept compilation cost or precompile key models
```

## FormulaCompiler.jl Integration

### Zero-Allocation Foundations

Margins.jl achieves performance through tight integration with FormulaCompiler.jl:

#### Compiled Formula Evaluation
```julia
# Single compilation, multiple evaluations
compiled = FormulaCompiler.compile_formula(model, Tables.columntable(data))  # Once
# Reused across all margin computations - zero allocation per evaluation
```

#### Derivative Computation
```julia
# Pre-built derivative evaluators
de = FormulaCompiler.derivativeevaluator(:ad, compiled, Tables.columntable(data), vars)  # Once per call
# Reused for every row of the call - zero allocation per derivative
```

#### Buffer Management
```julia
# Pre-allocated buffers prevent runtime allocation
row_buf = Vector{Float64}(undef, n_coef)        # Model-row buffer
g_buf = Vector{Float64}(undef, n_vars)          # Gradient buffer  
gβ_accumulator = Vector{Float64}(undef, n_coef) # Parameter gradient buffer
```

### Advanced Performance Patterns

#### Caching Strategies
```julia
# FormulaCompiler artifacts are cached automatically
# Multiple margin calls on same model/data reuse compilation
result1 = population_margins(model, data; type=:effects)      # Compiles
result2 = profile_margins(model, data, means_grid(data); type=:effects)  # Reuses compilation
```

#### Batch Processing Optimization
```julia
# Process multiple scenarios efficiently
scenarios = cartesian_grid(x1=[0,1,2], group=["A","B","C"])  # 9 profiles

# Single compilation, multiple scenario evaluations
results = profile_margins(model, data, scenarios; type=:effects)  # Efficient
```

## Production Deployment Guidelines

### Recommended Configuration
```julia
# High-performance production settings
result = population_margins(
    model, data;
    backend = :ad,           # Recommended: zero allocation, exact derivatives
    scale = :response,       # Response scale
    type = :effects          # Core functionality
)
```

### Monitoring and Validation
```julia
# Performance monitoring in production
function production_margins(model, data; kwargs...)
    # Allocation monitoring
    alloc_before = Base.gc_num().poolalloc
    
    result = population_margins(model, data; backend=:ad, kwargs...)
    
    alloc_after = Base.gc_num().poolalloc
    alloc_diff = alloc_after - alloc_before
    
    # Log excessive allocations
    if alloc_diff > 10000  # 10KB threshold
        @warn "Excessive allocation detected" alloc_diff
    end
    
    return result
end
```

### Error Handling
```julia
# Robust production wrapper
# Note: No implicit backend fallbacks. Select `backend` explicitly.
```

---

*This performance guide ensures you can leverage Margins.jl's full computational potential while maintaining statistical rigor in production environments. For conceptual background on why Population vs Profile matters for performance, see [Mathematical Foundation](mathematical_foundation.md). For comprehensive API usage, see [API Reference](api.md).*
