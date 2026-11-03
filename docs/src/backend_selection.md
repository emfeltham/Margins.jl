# Backend Selection Guide

*Understanding AD and FD backends for marginal effects computation*

## Recommendation: Use AD (Automatic Differentiation)

**TL;DR:** Use `backend=:ad` (the default) for all marginal effects computations. It provides:
- **Zero per-row allocation** after warmup
- **Exact derivatives** (up to floating-point rounding), with no step size to choose
- **Domain safety** for log(), sqrt(), and other sensitive functions: no evaluation outside the data
- **Faster** than FD (about 1.6-2x in the measurements below)
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
    └── Evaluates only at the observed values (no domain errors from stepping)

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

With `log(w)` or `sqrt(w)` and a few values of `w` near zero (1e-12 to 1e-7), `backend=:fd` stops with a `DomainError` (the finite-difference step, about 6e-6 here, crosses zero), while `backend=:ad` returns the estimate.

**Why AD succeeds**: Automatic differentiation computes exact derivatives without evaluating the function away from the observed values. It cannot repair a derivative that is itself infinite: with an observation at exactly `w = 0`, the AD estimate for `log(w)` is `-Inf`.

### **Functions Safe for Either Backend**

```julia
# Linear relationships - both backends equivalent
model = lm(@formula(y ~ x + z), data)  
population_margins(model, data; backend=:fd)   # Works
population_margins(model, data; backend=:ad)   # Same estimates (to about 1e-8)

# Polynomial functions - both work well
model = lm(@formula(y ~ x + x^2), data)
population_margins(model, data; backend=:fd)   # Works
population_margins(model, data; backend=:ad)   # Same estimates (to about 1e-8)

# Simple transformations - no domain issues
model = lm(@formula(y ~ x/10 + z*2), data)
population_margins(model, data; backend=:fd)   # No per-row allocation
```

## Performance Characteristics

### Memory Usage Analysis

**Both backends evaluate rows without allocating:** the per-row derivative loops allocate nothing, on either backend. A whole `population_margins` call allocates a fixed amount for its setup (engine, buffers, result) that does not grow with the number of rows, plus a `Float64` copy of each integer-typed model column (8 bytes per row per column):

```julia
# Allocation per warm call does not grow with n (all-Float64 data):
@allocated population_margins(model, data_1000; backend=:fd)   # about the same as for data_50000
@allocated population_margins(model, data_1000; backend=:ad)   # about the same as for data_50000
```

**Memory Usage Decision:**
- **All dataset sizes**: Neither backend allocates per row
- **Choice based on reliability and accuracy**: AD provides superior domain handling
- **Construction cost**: the per-call allocation is nearly the same for both (about 52 KiB for AD and 51 KiB for FD in the measurements below)

### Speed Performance

Both backends scale linearly with n; AD has the smaller constant. Measured with BenchmarkTools (median of 30 warm calls; Apple M4 Pro, Julia 1.13.3, one thread, Margins 3.0.0), for the AME of three continuous variables in `y ~ x + z + x & z + log(w) + g` (`g` a three-level factor; 9 coefficients), with the same precomputed `vcov` passed to every call:

| Model | n | `:ad` | `:fd` | FD / AD |
|-------|---|-------|-------|---------|
| `lm` | 1,000 | 0.11 ms | 0.17 ms | 1.6 |
| `lm` | 10,000 | 0.74 ms | 1.5 ms | 2.0 |
| `lm` | 100,000 | 7.1 ms | 15 ms | 2.1 |
| logit `glm` | 1,000 | 0.13 ms | 0.19 ms | 1.5 |
| logit `glm` | 10,000 | 0.86 ms | 1.7 ms | 2.0 |
| logit `glm` | 100,000 | 8.5 ms | 17 ms | 2.0 |

Each call allocated about 44 KiB (n = 1,000) to 52 KiB (n = 10,000 and 100,000) on either backend, with no growth from 10,000 to 100,000 rows. Timings depend on the machine and the model; the ratio is the more portable figure.

## Numerical Accuracy

### Both Backends Agree Closely

For well-conditioned problems, both backends produce statistically equivalent results:

```julia
fd_result = population_margins(model, data; backend=:fd)
ad_result = population_margins(model, data; backend=:ad)

isapprox(DataFrame(fd_result).estimate, DataFrame(ad_result).estimate; rtol=1e-6)  # true
```

In the benchmark models above (`lm` and logit), the FD estimates and standard errors differed from the AD ones by at most about 1e-8 in relative terms: the error of the finite-difference approximation, which AD does not have.

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
| **General production workflows** | `:ad` | No per-row allocation, reliability, about 2x faster |
| **Large datasets (>10k observations)** | `:ad` | No per-row allocation, faster |
| **Development/testing** | `:ad` | Higher reliability, exact derivatives |
| **High-precision requirements** | `:ad` | Exact derivatives vs numerical approximation |
| **Legacy code maintenance** | `:fd` | Only if existing code explicitly uses FD |
| **Debugging/validation** | Both | Compare results to verify correctness |

### Production Configuration Guidance

- Default to `backend=:ad` for reliability and accuracy (also no per-row allocation).
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
        
        if isapprox(fd_est, ad_est; rtol=1e-6)
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
# - Zero allocation per row after warmup
# - Reuses pre-allocated buffers
# - Scalar operations avoid broadcast allocations

# AD: Uses dual number arithmetic with compiled evaluators (OPTIMIZED)
# - Zero allocation per row after warmup (columns are read as Float64 and converted to duals on read)
# - Exact derivative computation (up to floating-point rounding)
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
- No per-row allocation (as FD)
- Faster than FD (about 1.6-2x in the measurements above)
- Domain safety (no DomainError from stepping outside the domain of log, sqrt, 1/x)
- Exact derivatives (up to floating-point rounding)
- Statistical validity

### **When to Use FD:**

Only use `backend=:fd` for:
1. **Legacy compatibility** - Maintaining existing code that explicitly uses FD
2. **Validation** - Comparing AD vs FD results for debugging
3. **Very rare edge cases** - Contact maintainers if you believe you need FD for a new use case

**Important:** FD is not faster, not more memory-efficient, and less reliable than AD. There is no performance or memory reason to prefer FD for new code.

### **Statistical Guarantees:**

Both backends maintain statistical correctness when they succeed:
- Same delta-method standard errors (when computed successfully)
- Same marginal effect estimates (when numerically stable)  
- Same confidence intervals and hypothesis tests

**The reliability difference is in computational robustness, not statistical validity.**

---

*For performance optimization details, see [Performance Guide](performance.md). For mathematical background, see [Mathematical Foundation](mathematical_foundation.md).*
