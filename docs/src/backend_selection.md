# Backend Selection Guide

*Choosing between AD and FD backends for optimal reliability and performance*

## Overview

Margins.jl offers two computational backends for derivative calculation, each optimized for different use cases:

- **`:ad`** (Automatic Differentiation) - Higher reliability, handles domain-sensitive functions
- **`:fd`** (Finite Differences) - Zero allocation, optimal for production environments

**TL;DR**: Use **`:ad`** for functions like `log()`, `sqrt()`, `1/x`. Use **`:fd`** for memory-critical production workflows with well-conditioned functions.

## Quick Decision Tree

```
Does your formula contain log(), sqrt(), 1/x, or x^(fractional)?
├── YES → Use backend=:ad (REQUIRED for reliability)
└── NO → Choose based on priorities:
    ├── Memory critical or large dataset? → Use backend=:fd (O(1) memory)
    └── Reliability/simplicity priority? → Use backend=:ad (handles more cases)
```

## Critical Reliability Differences

### 🚨 **Domain-Sensitive Functions: Always Use AD**

**Functions that require `backend=:ad`:**

```julia
# Log transformations - FD can push values below zero
model = lm(@formula(y ~ log(x)), data)
population_margins(model, data; backend=:ad)  # ✓ Required

# Square root functions - FD can push values negative  
model = lm(@formula(y ~ sqrt(x)), data)  
population_margins(model, data; backend=:ad)  # ✓ Required

# Inverse functions near zero - FD can create division issues
model = lm(@formula(y ~ 1/x), data)
population_margins(model, data; backend=:ad)  # ✓ Recommended

# Fractional powers - Similar domain sensitivity as sqrt
model = lm(@formula(y ~ x^(1/3)), data)
population_margins(model, data; backend=:ad)  # ✓ Recommended
```

**Why FD fails**: Finite difference computation `f(x+h) - f(x)` can push arguments outside valid domains:
- `log(x+h)` where `x+h < 0` → `DomainError`  
- `sqrt(x+h)` where `x+h < 0` → `DomainError`
- `1/(x+h)` where `x+h ≈ 0` → numerical instability

**Why AD succeeds**: Automatic differentiation computes exact derivatives without domain-violating function evaluations.

### ✅ **Functions Safe for Either Backend**

```julia
# Linear relationships - both backends equivalent
model = lm(@formula(y ~ x + z), data)  
population_margins(model, data; backend=:fd)   # ✓ Excellent performance
population_margins(model, data; backend=:ad)   # ✓ Equivalent results

# Polynomial functions - both work well
model = lm(@formula(y ~ x + x^2), data)
population_margins(model, data; backend=:fd)   # ✓ Choose based on performance needs
population_margins(model, data; backend=:ad)   # ✓ Same statistical results

# Simple transformations - no domain issues
model = lm(@formula(y ~ x/10 + z*2), data)
population_margins(model, data; backend=:fd)   # ✓ Zero allocation preferred
```

## Performance Characteristics

### Memory Usage Analysis

```julia
# FD: Constant memory regardless of dataset size
@allocated population_margins(model, data_100; backend=:fd)    # ~6KB
@allocated population_margins(model, data_1000; backend=:fd)   # ~6KB (same!)  
@allocated population_margins(model, data_5000; backend=:fd)   # ~6KB (same!)

# AD: Memory scales with dataset size
@allocated population_margins(model, data_100; backend=:ad)    # ~281KB
@allocated population_margins(model, data_1000; backend=:ad)   # ~2.8MB
@allocated population_margins(model, data_5000; backend=:ad)   # ~14MB+
```

**Memory Usage Decision:**
- **Dataset < 1000**: Either backend fine (memory difference minimal)
- **Dataset 1000-10000**: Consider FD for memory savings (6KB vs MB)
- **Dataset > 10000**: FD recommended for memory efficiency (6KB vs 10s of MB)

### Speed Performance

Performance varies by problem complexity and system:

```julia
# Typical performance ranges (varies by system and model complexity)
# Small problems (n=100-1000)
@btime population_margins($model, $data; backend=:fd)  # 0.1-10ms
@btime population_margins($model, $data; backend=:ad)  # 1-20ms

# Large problems (n=10000+)  
@btime population_margins($model, $large_data; backend=:fd)  # Scales with n
@btime population_margins($model, $large_data; backend=:ad)  # Scales with n + memory pressure
```

**Key insight**: FD's O(1) memory usage vs AD's O(n) memory becomes critical for large datasets.

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
| **Production workflows** | `:fd` | Zero allocation, predictable memory |
| **Large datasets (>10k)** | `:fd` | O(1) memory vs O(n) memory critical |
| **Memory-constrained systems** | `:fd` | Constant 6KB footprint |
| **Development/testing** | `:ad` | Higher reliability, allocation cost acceptable |
| **High-precision requirements** | `:ad` | Potentially more accurate for complex functions |
| **Batch processing** | `:fd` | Consistent memory usage across runs |

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

# AD: Uses dual number arithmetic with compiled evaluators  
# - Small allocation for dual number storage
# - Exact derivative computation
# - Composition via chain rule
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

2. **For other functions, choose based on priorities:**
   - **Choose AD when:** Reliability is paramount, memory usage is not a constraint, function complexity is high
   - **Choose FD when:** Memory usage is critical, large datasets (>10k observations), production environments requiring predictable memory footprint

3. **When in doubt:** Start with `:ad` for development, optimize to `:fd` for production if no domain sensitivity issues

### **Statistical Guarantees:**

Both backends maintain statistical correctness when they succeed:
- Same delta-method standard errors (when computed successfully)
- Same marginal effect estimates (when numerically stable)  
- Same confidence intervals and hypothesis tests

**The reliability difference is in computational robustness, not statistical validity.**

---

*For performance optimization details, see [Performance Guide](performance.md). For mathematical background, see [Mathematical Foundation](mathematical_foundation.md).*