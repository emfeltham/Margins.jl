# Backend Selection Guide

*Choosing between AD and FD backends for optimal reliability and performance*

## Overview

Margins.jl offers two computational backends for derivative calculation, each optimized for different use cases:

- **`:ad`** (Automatic Differentiation) - Zero allocation, higher reliability, handles domain-sensitive functions
- **`:fd`** (Finite Differences) - Zero allocation, efficient for simple formulas

**N.B.**: For the most part, Finite Differences essentially has legacy status at this point, developed before efficient AD was implemented. The AD backend now provides superior performance, reliability, and maintains zero allocation.

**Summary**: Use **`:ad`** for most applications. Zero allocation with comprehensive type support and robust domain handling.

## Quick Decision Tree

```
For most applications:
└── Use backend=:ad (recommended default)
    ├── Zero allocation performance
    ├── Machine precision accuracy
    ├── All numeric types supported
    └── Robust domain handling

Use backend=:fd for:
├── Specific compatibility requirements
└── Simple linear formulas
```

The AD backend supports all integer types (Int8, Int16, UInt32, etc.) with zero allocation performance.

## Critical Reliability Differences

### **Domain-Sensitive Functions: Always Use AD**

**Functions that require `backend=:ad`:**

```julia
# Log transformations - FD can push values below zero
model = lm(@formula(y ~ log(x)), data)
population_margins(model, data; backend=:ad)  # Required

# Square root functions - FD can push values negative  
model = lm(@formula(y ~ sqrt(x)), data)  
population_margins(model, data; backend=:ad)  # Required

# Inverse functions near zero - FD can create division issues
model = lm(@formula(y ~ 1/x), data)
population_margins(model, data; backend=:ad)  # Recommended

# Fractional powers - Similar domain sensitivity as sqrt
model = lm(@formula(y ~ x^(1/3)), data)
population_margins(model, data; backend=:ad)  # Recommended
```

**Why FD fails**: Finite difference computation `f(x+h) - f(x)` can push arguments outside valid domains:
- `log(x+h)` where `x+h < 0` → `DomainError`  
- `sqrt(x+h)` where `x+h < 0` → `DomainError`
- `1/(x+h)` where `x+h ≈ 0` → numerical instability

**Why AD succeeds**: Automatic differentiation computes exact derivatives without domain-violating function evaluations.

### **Functions Safe for Either Backend**

```julia
# Linear relationships - both backends equivalent
model = lm(@formula(y ~ x + z), data)  
population_margins(model, data; backend=:fd)   # Efficient performance
population_margins(model, data; backend=:ad)   # Equivalent results

# Polynomial functions - both work well
model = lm(@formula(y ~ x + x^2), data)
population_margins(model, data; backend=:fd)   # Choose based on performance needs
population_margins(model, data; backend=:ad)   # Same statistical results

# Simple transformations - no domain issues
model = lm(@formula(y ~ x/10 + z*2), data)
population_margins(model, data; backend=:fd)   # Zero allocation
```

## Performance Characteristics

### Memory Usage Analysis

**Both backends now achieve zero allocation performance:**

```julia
# FD: Zero allocation after warmup
@allocated population_margins(model, data_100; backend=:fd)    # 0 bytes
@allocated population_margins(model, data_1000; backend=:fd)   # 0 bytes  
@allocated population_margins(model, data_5000; backend=:fd)   # 0 bytes

# AD: Zero allocation after warmup
@allocated population_margins(model, data_100; backend=:ad)    # 0 bytes
@allocated population_margins(model, data_1000; backend=:ad)   # 0 bytes
@allocated population_margins(model, data_5000; backend=:ad)   # 0 bytes
```

**Memory Usage Decision:**
- **All dataset sizes**: Both backends achieve zero allocation performance
- **Choice based on reliability and accuracy**: AD provides superior domain handling
- **Construction cost**: AD requires slightly more memory during evaluator setup (amortized over many evaluations)

### Speed Performance

Both backends achieve excellent performance, with AD providing 3-5x improvements:

```julia
# Typical performance ranges (varies by system and model complexity)
# Small problems (n=100-1000)  
@btime population_margins($model, $data; backend=:fd)  # 0.1-10ms (baseline)
@btime population_margins($model, $data; backend=:ad)  # 0.05-5ms (3-5x faster!)

# Large problems (n=10000+)
@btime population_margins($model, $large_data; backend=:fd)  # Scales linearly with n
@btime population_margins($model, $large_data; backend=:ad)  # Scales linearly, but with better constant factors
```

**Key insight**: With zero-allocation AD, the performance differences now favor AD in most cases, while maintaining superior numerical properties.

## Numerical Accuracy

### Both Backends Provide Equivalent Accuracy

For well-conditioned problems, both backends produce statistically equivalent results:

```julia
# Linear models - identical to machine precision
fd_result = population_margins(model, data; backend=:fd)
ad_result = population_margins(model, data; backend=:ad)

DataFrame(fd_result).estimate ≈ DataFrame(ad_result).estimate  # rtol=1e-12 ✓

# GLM models - equivalent within appropriate tolerances  
fd_glm = population_margins(glm_model, data; backend=:fd)
ad_glm = population_margins(glm_model, data; backend=:ad)

DataFrame(fd_glm).estimate ≈ DataFrame(ad_glm).estimate  # rtol=1e-10 ✓
```

### AD May Be More Accurate For

- Complex function compositions
- Functions with steep gradients  
- Near-boundary evaluations
- Models with numerical conditioning issues

## Production Recommendations

### Automatic Backend Selection Pattern

```julia
# Safe automatic selection function
function safe_margins(model, data; formula_has_log=false, formula_has_sqrt=false, kwargs...)
    # Always use AD for domain-sensitive functions
    backend = (formula_has_log || formula_has_sqrt) ? :ad : :fd
    return population_margins(model, data; backend=backend, kwargs...)
end

# Usage examples
safe_margins(model, data; formula_has_log=true)   # Uses :ad automatically
safe_margins(model, data)                         # Uses :fd for performance
```

### Backend Selection by Use Case

| **Use Case** | **Backend** | **Rationale** |
|--------------|-------------|---------------|
| **Domain-sensitive functions** | `:ad` | Required for log(), sqrt(), 1/x |
| **General production workflows** | `:ad` | Zero allocation + higher reliability + faster |
| **Large datasets (>10k)** | `:ad` | Zero allocation + superior performance |
| **Memory-constrained systems** | Either | Both achieve zero allocation |
| **Development/testing** | `:ad` | Higher reliability, now zero allocation |
| **High-precision requirements** | `:ad` | Machine precision + zero allocation |
| **Simple linear formulas** | `:fd` | May be slightly faster for basic operations |

### Production Configuration Examples

```julia
# Safe production default (handles most cases)
function production_margins(model, data; kwargs...)
    # Start with AD for safety, fallback to FD if needed
    try
        return population_margins(model, data; backend=:ad, kwargs...)
    catch e
        if e isa DomainError
            rethrow(e)  # Don't fallback for domain errors - formula needs fixing
        else
            @warn "AD backend failed, using FD fallback" exception=e
            return population_margins(model, data; backend=:fd, kwargs...)
        end
    end
end

# Memory-optimized production (for well-conditioned functions only)
function memory_optimized_margins(model, data; kwargs...)
    return population_margins(model, data; backend=:fd, kwargs...)
end

# High-reliability production (when memory is not critical)  
function high_reliability_margins(model, data; kwargs...)
    return population_margins(model, data; backend=:ad, kwargs...)
end
```

## Troubleshooting Backend Issues

### Common Error Patterns

#### DomainError with FD Backend
```julia
# Error: DomainError with -1.23e-6: log was called with a negative real number
result = population_margins(model, data; backend=:fd)  # ❌ Fails

# Solution: Use AD backend for log functions
result = population_margins(model, data; backend=:ad)  # ✓ Works
```

#### Memory Pressure with AD Backend
```julia
# Large dataset causing memory issues with AD
huge_result = population_margins(model, huge_data; backend=:ad)  # ⚠ May run out of memory

# Solution: Use FD backend for memory efficiency
huge_result = population_margins(model, huge_data; backend=:fd)  # ✓ Constant memory
```

### Backend Validation Testing

```julia
# Test both backends for new functions
function test_backend_compatibility(model, data)
    try_fd = try population_margins(model, data; backend=:fd) catch nothing end
    try_ad = try population_margins(model, data; backend=:ad) catch nothing end
    
    if try_fd === nothing && try_ad !== nothing
        @warn "Function requires AD backend - FD fails with domain error"
        return :ad_required
    elseif try_fd !== nothing && try_ad !== nothing
        # Compare results for consistency
        fd_est = DataFrame(try_fd).estimate
        ad_est = DataFrame(try_ad).estimate
        
        if fd_est ≈ ad_est rtol=1e-10
            @info "Both backends produce consistent results"
            return :either_ok
        else
            @warn "Backends produce different results - investigate numerical issues"
            return :inconsistent
        end
    else
        @error "Both backends failed"
        return :both_failed
    end
end

# Usage
compatibility = test_backend_compatibility(model, data)
```

## Advanced Topics

### FormulaCompiler Integration

Both backends leverage FormulaCompiler.jl's optimized evaluation:

```julia
# FD: Uses finite difference approximation with compiled evaluators
# - Zero allocation after warmup
# - Reuses pre-allocated buffers
# - Scalar operations avoid broadcast allocations

# AD: Uses dual number arithmetic with compiled evaluators (OPTIMIZED)
# - Zero allocation after warmup via pre-conversion strategy
# - Exact derivative computation with machine precision
# - 3-5x performance improvement over previous AD implementation
# - Composition via chain rule with type homogeneity
```

### Custom Tolerance Settings

For functions near domain boundaries, you may need custom tolerances:

```julia
# Custom finite difference step size (advanced)
# Note: This is a FormulaCompiler.jl setting, not directly exposed in Margins.jl
# Contact maintainers if you need custom FD step sizes for specific functions
```

## Summary Guidelines

### **Default Strategy (Recommended):**

1. **For domain-sensitive functions (log, sqrt, 1/x):**
   - **Always use AD** (`:ad`) - FD will fail with DomainErrors

2. **For all other functions:**
   - **Use AD as default** (`:ad`) - Zero allocation + faster + more reliable
   - **Use FD only for:** Simple linear formulas where marginal speed differences matter

3. **When in doubt:** Use `:ad` - it now provides the best of all worlds (zero allocation, speed, reliability)

### **Statistical Guarantees:**

Both backends maintain statistical correctness when they succeed:
- Same delta-method standard errors (when computed successfully)
- Same marginal effect estimates (when numerically stable)  
- Same confidence intervals and hypothesis tests

**The reliability difference is in computational robustness, not statistical validity.**

---

*For performance optimization details, see [Performance Guide](performance.md). For mathematical background, see [Mathematical Foundation](mathematical_foundation.md).*