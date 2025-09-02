# Performance Guide

*Optimizing marginal effects analysis for production workflows*

## Performance Overview

Margins.jl achieves **production-grade performance** through careful optimization while maintaining absolute statistical correctness. The package delivers:

- **Profile margins**: O(1) constant time (~100-200μs) regardless of dataset size
- **Population margins**: Optimized O(n) scaling (~150ns per row) with constant memory footprint  
  *See [Mathematical Foundation](mathematical_foundation.md) for conceptual understanding of when to choose population vs profile approaches*  
- **Zero-allocation core**: FormulaCompiler.jl foundation eliminates unnecessary allocations
- **Statistical integrity**: Performance optimizations never compromise statistical validity

## Performance Characteristics

### Profile Analysis: O(1) Constant Time

Profile margins achieve **constant-time performance** regardless of dataset size:

```julia
using BenchmarkTools, Margins

# Performance is independent of dataset size
@btime profile_margins($model, $data_1k; at=:means, type=:effects)     # ~200μs
@btime profile_margins($model, $data_100k; at=:means, type=:effects)   # ~200μs (same!)
@btime profile_margins($model, data_1M; at=:means, type=:effects)       # ~200μs (same!)

# Complex scenarios also O(1)
scenarios = Dict(:x1 => [0,1,2], :x2 => [10,20,30], :group => ["A","B"])  # 18 profiles
@btime profile_margins($model, $huge_data; at=scenarios)                    # ~400μs
```

**Why this matters**: Profile analysis cost is **independent of sample size**, making it efficient for large-scale econometric analysis.

### Population Analysis: Optimized O(n) Scaling

Population margins scale linearly with optimized per-row costs:

```julia
# Consistent per-row performance across dataset sizes
@btime population_margins($model, $data_1k)    # ~0.2ms  (~150ns/row)
@btime population_margins($model, $data_10k)   # ~1.5ms  (~150ns/row)  
@btime population_margins($model, $data_100k)  # ~15ms   (~150ns/row)

# Memory footprint remains constant
@allocated population_margins(model, data_1k)    # ~6KB
@allocated population_margins(model, data_10k)   # ~6KB (same!)
@allocated population_margins(model, data_100k)  # ~6KB (same!)
```

**Why this matters**: Population analysis maintains **constant allocation footprint** while delivering consistent per-row performance.

## Dataset Size Guidelines

### Performance Expectations by Scale

| Dataset Size | Population Margins | Profile Margins | Recommended Workflow |
|--------------|-------------------|-----------------|---------------------|
| **< 1k** | Excellent (<1ms) | Excellent (~200μs) | Use either approach freely |
| **1k-10k** | Excellent (1-10ms) | Excellent (~200μs) | Profile preferred for scenarios |
| **10k-100k** | Good (10-100ms) | Excellent (~200μs) | Profile for exploration, population for final analysis |
| **100k-1M** | Acceptable (0.1-1s) | Excellent (~200μs) | Profile strongly preferred |
| **> 1M** | Scales appropriately | Excellent (~200μs) | Profile analysis, selective population |

### Backend Selection by Use Case

For detailed backend selection guidance including domain-sensitive functions and reliability considerations, see **[Backend Selection Guide](backend_selection.md)**.

**Quick summary:**
- **`:ad`** - Required for log(), sqrt(), 1/x functions; higher reliability
- **`:fd`** - Zero allocation, optimal for production and large datasets

```julia
# Production configuration (memory-optimized)
population_margins(model, data; backend=:fd, target=:eta)

# Development/high-reliability configuration  
profile_margins(model, data; at=:means, backend=:ad, target=:mu)

# Domain-sensitive functions (log, sqrt) - AD required
population_margins(model, data; backend=:ad)  # Required for log(x), sqrt(x)
```

## Optimization Principles

### Core Performance Philosophy

**Statistical Correctness First**: Performance optimizations never compromise statistical validity
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

#### Finite Differences (`:fd`)
```julia
# Zero allocation after warmup
population_margins(model, data; backend=:fd)  # 0 bytes allocated
```

**Advantages**:
- Zero allocation in production paths
- Consistent performance across dataset sizes
- Robust numerical behavior

**Use cases**: Production workflows, large datasets, memory-constrained environments

#### Automatic Differentiation (`:ad`)  
```julia
# Small allocation cost for higher accuracy
population_margins(model, data; backend=:ad)  # ~400-500 bytes per call
```

**Advantages**:
- Higher numerical accuracy
- Faster convergence for complex models
- Better handling of extreme values

**Use cases**: Development, verification, high-precision requirements

## Memory Management

### Allocation Patterns

Margins.jl follows strict allocation patterns to ensure predictable memory usage:

```julia
# Profile margins: constant allocation regardless of data size
@allocated profile_margins(model, small_data; at=:means)  # ~2KB
@allocated profile_margins(model, large_data; at=:means)  # ~2KB (same!)

# Population margins: constant base allocation + O(1) per computation
@allocated population_margins(model, data_1k)   # ~6KB  
@allocated population_margins(model, data_10k)  # ~6KB (constant!)
```

### Memory Efficiency Best Practices

#### For Large Datasets
```julia
# Use profile analysis for exploration (O(1) memory)
scenarios = Dict(:x1 => [-1, 0, 1], :treatment => [0, 1])
results = profile_margins(model, large_data; at=scenarios, backend=:fd)

# Use population analysis selectively  
key_effects = population_margins(model, large_data; vars=[:treatment], backend=:fd)
```

#### For Batch Processing
```julia
# Process multiple models with consistent memory usage
models = [model1, model2, model3]
results = []

for model in models
    # Each call has same memory footprint
    result = population_margins(model, data; backend=:fd)
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