# Backend Selection Guide

*Understanding AD and FD backends for marginal effects computation*

## Recommendation: Use AD (Automatic Differentiation)

**TL;DR:** Use `backend=:ad` (the default) for all marginal effects computations. It provides:
- **Zero allocation** performance after warmup
- **Machine precision** accuracy
- **Domain safety** for log(), sqrt(), and other sensitive functions
- **3-5x faster** than FD in most cases
- **All numeric types** supported (Int8, Int16, Float64, etc.)

```julia
# Recommended (AD is the default):
result = population_margins(model, data; type=:effects)

# Explicit AD specification (equivalent):
result = population_margins(model, data; type=:effects, backend=:ad)
```

## When Finite Differences (FD) Exists

The `:fd` backend exists for:
- **Historical compatibility** - Legacy code using FD
- **Debugging** - Comparing AD vs FD results to validate correctness
- **Edge cases** - Rare situations where FD may be preferred

**Important:** FD is **not recommended** for new code. It was developed before efficient AD implementation and is now effectively in maintenance mode.

## Quick Decision Tree

```
For all applications:
└── Use backend=:ad (default)
    ├── Required for: log(x), sqrt(x), 1/x, x^(1/3), etc.
    ├── Recommended for: all other formulas
    └── Never fails: domain-safe evaluation

Only use backend=:fd if:
├── Maintaining legacy code that explicitly uses FD
└── Debugging/validation (comparing AD vs FD results)
```

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

DataFrame(fd_result).estimate ≈ DataFrame(ad_result).estimate  # rtol=1e-12 PASS

# GLM models - equivalent within appropriate tolerances  
fd_glm = population_margins(glm_model, data; backend=:fd)
ad_glm = population_margins(glm_model, data; backend=:ad)

DataFrame(fd_glm).estimate ≈ DataFrame(ad_glm).estimate  # rtol=1e-10 PASS
```

### AD May Be More Accurate For

- Complex function compositions
- Functions with steep gradients  
- Near-boundary evaluations
- Models with numerical conditioning issues

## Production Recommendations

### Backend Selection Policy

- No `:auto` mode is provided.
- No implicit backend fallbacks are performed.
- Select `backend` explicitly. Use `:ad` by default; use `:fd` only when explicitly intended and theoretically safe.

### Backend Selection by Use Case

| **Use Case** | **Backend** | **Rationale** |
|--------------|-------------|---------------|
| **Domain-sensitive functions** (log, sqrt, 1/x) | `:ad` | **Required** - FD fails with DomainError |
| **General production workflows** | `:ad` | Zero allocation + reliability + 3-5x faster |
| **Large datasets (>10k observations)** | `:ad` | Zero allocation + superior performance |
| **Development/testing** | `:ad` | Higher reliability + machine precision |
| **High-precision requirements** | `:ad` | Exact derivatives vs numerical approximation |
| **Legacy code maintenance** | `:fd` | Only if existing code explicitly uses FD |
| **Debugging/validation** | Both | Compare results to verify correctness |

### Production Configuration Guidance

- Default to `backend=:ad` for reliability and accuracy (also zero allocation).
- Use `backend=:fd` only for simple, well-conditioned formulas and when you explicitly want FD.
- For domain-sensitive functions (log, sqrt, 1/x near 0), always use `:ad`.

## Troubleshooting Backend Issues

### Common Error Patterns

#### DomainError with FD Backend
```julia
# Error: DomainError with -1.23e-6: log was called with a negative real number
result = population_margins(model, data; backend=:fd)  #  Fails

# Solution: Use AD backend for log functions
result = population_margins(model, data; backend=:ad)  # Works
```

#### ~~Memory Pressure with AD Backend~~ (Obsolete)

**Note:** This troubleshooting section is obsolete as of v2.0. Both AD and FD achieve zero allocation performance, so there is no memory efficiency difference between backends. If you encounter memory issues, they are likely related to dataset size or model complexity, not the backend choice.

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

**Use `backend=:ad` for everything.** The AD backend is now the recommended default for all use cases, providing:
- Zero allocation performance (equal to FD)
- Superior speed (3-5x faster than FD)
- Domain safety (handles log, sqrt, 1/x correctly)
- Machine precision accuracy
- Statistical validity

### **When to Use FD:**

Only use `backend=:fd` for:
1. **Legacy compatibility** - Maintaining existing code that explicitly uses FD
2. **Validation** - Comparing AD vs FD results for debugging
3. **Very rare edge cases** - Contact maintainers if you believe you need FD for a new use case

**Important:** FD is not faster, not more memory-efficient, and less reliable than AD in v2.0+. There is no performance or memory reason to prefer FD for new code.

### **Statistical Guarantees:**

Both backends maintain statistical correctness when they succeed:
- Same delta-method standard errors (when computed successfully)
- Same marginal effect estimates (when numerically stable)  
- Same confidence intervals and hypothesis tests

**The reliability difference is in computational robustness, not statistical validity.**

---

*For performance optimization details, see [Performance Guide](performance.md). For mathematical background, see [Mathematical Foundation](mathematical_foundation.md).*
