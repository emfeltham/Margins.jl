# Performance Guide

*Optimizing marginal effects analysis for computational efficiency*

## Performance Overview

Margins.jl achieves efficient computation through careful optimization while maintaining statistical correctness. The package provides:

- **Profile margins**: O(1) constant time regardless of dataset size
- **Population margins**: O(n) scaling with low per-row computational cost and constant memory footprint  
  *See [Mathematical Foundation](mathematical_foundation.md) for conceptual understanding of when to choose population vs profile approaches*  
- **Zero-allocation core**: FormulaCompiler.jl foundation eliminates unnecessary allocations
- **Statistical integrity**: Performance optimizations maintain statistical validity

## Performance Characteristics

### Profile Analysis: O(1) Constant Time

Profile margins achieve **constant-time performance** regardless of dataset size:

```julia
using BenchmarkTools, Margins

# Performance is independent of dataset size
@btime profile_margins($model, $data_1k; at=:means, type=:effects)     # constant time
@btime profile_margins($model, $data_100k; at=:means, type=:effects)   # same complexity
@btime profile_margins($model, data_1M; at=:means, type=:effects)       # same complexity

# Complex scenarios also O(1)
scenarios = Dict(:x1 => [0,1,2], :x2 => [10,20,30], :group => ["A","B"])  # 18 profiles
@btime profile_margins($model, $huge_data; at=scenarios)                    # still constant time
```

**Why this matters**: Profile analysis cost is **independent of sample size**, making it efficient for large-scale econometric analysis.

### Population Analysis: Optimized O(n) Scaling

Population margins scale linearly with optimized per-row costs:

```julia
# Linear scaling with low per-row computational cost
@btime population_margins($model, $data_1k)    # scales with dataset size
@btime population_margins($model, $data_10k)   # with efficient per-row processing  
@btime population_margins($model, $data_100k)  # minimal allocation overhead

# Memory footprint remains zero (both backends)
@allocated population_margins(model, data_1k; backend=:fd)    # 0 bytes
@allocated population_margins(model, data_10k; backend=:fd)   # 0 bytes
@allocated population_margins(model, data_100k; backend=:fd)  # 0 bytes

@allocated population_margins(model, data_1k; backend=:ad)    # 0 bytes
@allocated population_margins(model, data_10k; backend=:ad)   # 0 bytes
@allocated population_margins(model, data_100k; backend=:ad)  # 0 bytes
```

**Why this matters**: Population analysis maintains **constant allocation footprint** while delivering consistent per-row performance.

## Dataset Size Guidelines

### Performance Expectations by Scale

| Dataset Size | Population Margins | Profile Margins | Recommended Workflow |
|--------------|-------------------|-----------------|---------------------|
| **< 1k** | Fast | Constant time | Use either approach freely |
| **1k-10k** | Fast | Constant time | Profile preferred for scenarios |
| **10k-100k** | Scales linearly | Constant time | Profile for exploration, population for final analysis |
| **100k-1M** | Scales appropriately | Constant time | Profile strongly preferred |
| **> 1M** | Scales with dataset size | Constant time | Profile analysis, selective population |

### Backend Selection by Use Case

For detailed backend selection guidance including domain-sensitive functions and reliability considerations, see **[Backend Selection Guide](backend_selection.md)**.

**Quick summary:**
- **`:ad`** - Required for log(), sqrt(), 1/x functions; higher reliability
- **`:fd`** - Zero allocation, optimal for production and large datasets

```julia
# Production configuration (memory-optimized)
population_margins(model, data; backend=:fd, scale=:link)

# Development/high-reliability configuration  
profile_margins(model, data, means_grid(data); backend=:ad, scale=:response)

# Domain-sensitive functions (log, sqrt) - AD required
population_margins(model, data; backend=:ad)  # Required for log(x), sqrt(x)
```

## Optimization Principles

### Core Performance Philosophy

**Statistical Correctness First**: Performance optimizations maintain statistical validity
- Delta-method standard errors use full covariance matrix
- All gradient computations maintain mathematical precision
- Bootstrap validation ensures statistical accuracy

**Zero-Allocation Patterns**: Eliminate unnecessary memory allocations
- Pre-allocated buffers reused across computations
- FormulaCompiler.jl provides zero-allocation evaluation primitives
- Constant memory footprint regardless of dataset size

**Computational Efficiency**: Optimize hot paths without changing methodology
- Compiled formula evaluation with caching
- Efficient gradient accumulation patterns
- Scalar operations over broadcast temporaries

### Backend Performance Characteristics

#### Automatic Differentiation (`:ad`) - **RECOMMENDED DEFAULT**
```julia
# Zero allocation after warmup
population_margins(model, data; backend=:ad)  # 0 bytes allocated
```

**Advantages**:
- Zero allocation after warmup
- Machine precision accuracy (exact derivatives)
- Robust domain handling (handles log, sqrt, 1/x safely)
- Suitable for complex formulas

**Use cases**: Most applications - provides good performance and reliability

#### Finite Differences (`:fd`)
```julia
# Zero allocation after warmup
population_margins(model, data; backend=:fd)  # 0 bytes allocated
```

**Advantages**:
- Zero allocation in production paths
- Simple numerical implementation
- Good accuracy for well-conditioned functions

**Use cases**: Simple linear formulas where marginal speed differences matter

## Memory Management

### Allocation Patterns

Margins.jl achieves zero-allocation performance for computational workflows:

```julia
# Profile margins: constant allocation regardless of data size
@allocated profile_margins(model, small_data; at=:means)  # small constant allocation
@allocated profile_margins(model, large_data; at=:means)  # same allocation pattern

# Population margins: zero allocation after warmup (both backends)
@allocated population_margins(model, data_1k; backend=:fd)   # 0 bytes
@allocated population_margins(model, data_10k; backend=:fd)  # 0 bytes
@allocated population_margins(model, data_1k; backend=:ad)   # 0 bytes
@allocated population_margins(model, data_10k; backend=:ad)  # 0 bytes
```

### Memory Efficiency Best Practices

#### For Large Datasets
```julia
# Use profile analysis for exploration (O(1) memory)
scenarios = Dict(:x1 => [-1, 0, 1], :treatment => [0, 1])
results = profile_margins(model, large_data; at=scenarios)

# Use population analysis with zero-allocation backends
key_effects = population_margins(model, large_data; vars=[:treatment], backend=:ad)  # Recommended
# OR
key_effects = population_margins(model, large_data; vars=[:treatment], backend=:fd)  # Also zero allocation
```

#### For Batch Processing
```julia
# Process multiple models with zero allocation
models = [model1, model2, model3]
results = []

for model in models
    # Each call has zero allocation (either backend)
    result = population_margins(model, data; backend=:ad)  # Recommended: zero allocation
    push!(results, DataFrame(result))
end
```

## Troubleshooting Performance Issues

### Diagnostic Tools

#### Memory Allocation Checking
```julia
using BenchmarkTools

# Check allocation patterns
@allocated population_margins(model, data)  # Should be constant across dataset sizes

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
**Diagnosis**: Likely using wrong dispatch or DataFrame confusion
**Solution**: Ensure proper at parameter specification
```julia
# Correct: O(1) performance
profile_margins(model, data; at=:means, type=:effects)

# Avoid: DataFrame method confusion  
# Use Dict or NamedTuple for 'at' parameter
```

#### Issue: Population margins allocating excessively
**Diagnosis**: Potential struct field access in hot loops
**Solution**: Verify backend selection and data format
```julia
# Efficient approach
data_nt = Tables.columntable(data)  # Convert once
result = population_margins(model, data_nt; backend=:fd)
```

#### Issue: Inconsistent performance across runs
**Diagnosis**: Compilation effects or memory pressure
**Solution**: Warmup runs and consistent backend selection
```julia
# Warmup for consistent benchmarking
population_margins(model, small_data)  # Warmup
@btime population_margins($model, $data)  # Benchmark
```

## FormulaCompiler.jl Integration

### Zero-Allocation Foundations

Margins.jl achieves performance through tight integration with FormulaCompiler.jl:

#### Compiled Formula Evaluation
```julia
# Single compilation, multiple evaluations
compiled = FormulaCompiler.compile_formula(model, data)  # Once
# Reused across all margin computations - zero allocation per evaluation
```

#### Derivative Computation
```julia
# Pre-built derivative evaluators
de = FormulaCompiler.build_derivative_evaluator(compiled, data; vars=vars)  # Once  
# Reused for all marginal effects - zero allocation per derivative
```

#### Buffer Management
```julia
# Pre-allocated buffers prevent runtime allocation
η_buf = Vector{Float64}(undef, n_profiles)      # Linear predictor buffer
g_buf = Vector{Float64}(undef, n_vars)          # Gradient buffer  
gβ_accumulator = Vector{Float64}(undef, n_coef) # Parameter gradient buffer
```

### Advanced Performance Patterns

#### Caching Strategies
```julia
# FormulaCompiler artifacts are cached automatically
# Multiple margin calls on same model/data reuse compilation
result1 = population_margins(model, data; type=:effects)      # Compiles
result2 = profile_margins(model, data; at=:means, type=:effects)  # Reuses compilation
```

#### Batch Processing Optimization
```julia
# Process multiple scenarios efficiently
scenarios = [
    Dict(:x1 => 0, :group => "A"),
    Dict(:x1 => 1, :group => "B"), 
    Dict(:x1 => 2, :group => "C")
]

# Single compilation, multiple scenario evaluations
results = profile_margins(model, data; at=scenarios, type=:effects)  # Efficient
```

## Production Deployment Guidelines

### Recommended Configuration
```julia
# High-performance production settings
result = population_margins(
    model, data;
    backend = :fd,           # Zero allocation
    target = :eta,           # Often faster than :mu for GLMs
    type = :effects          # Core functionality
)
```

### Monitoring and Validation
```julia
# Performance monitoring in production
function production_margins(model, data; kwargs...)
    # Allocation monitoring
    alloc_before = Base.gc_num().poolalloc
    
    result = population_margins(model, data; backend=:fd, kwargs...)
    
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
function robust_margins(model, data; fallback_backend=:fd, kwargs...)
    try
        return population_margins(model, data; backend=:ad, kwargs...)
    catch e
        @warn "AD backend failed, falling back to FD" exception=e
        return population_margins(model, data; backend=fallback_backend, kwargs...)
    end
end
```

---

*This performance guide ensures you can leverage Margins.jl's full computational potential while maintaining statistical rigor in production environments. For conceptual background on why Population vs Profile matters for performance, see [Mathematical Foundation](mathematical_foundation.md). For comprehensive API usage, see [API Reference](api.md).*