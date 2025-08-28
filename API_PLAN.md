# Margins.jl API Plan (Profile/Population Framework)

This document finalizes the public API for Margins.jl based on the conceptual framework outlined in `statistical_framework.md`. The API is organized around the fundamental distinction between **profile** and **population** approaches, with a secondary choice between **effects** and **predictions**.

## Goals

- Clear conceptual separation based on inferential target (profile vs population)
- Simple API surface with two primary functions  
- Flexible type system that handles effects vs predictions naturally
- Preserve performance guarantees by building on FormulaCompiler
- Align with established statistical framework

## Conceptual Foundation

The API directly maps to the statistical framework:

|                    | **Effects**        | **Predictions**        |
|--------------------|-------------------|------------------------|
| **Profile**        | **MEM**           | **APM**                |
| **Population**     | **AME/APE**       | **Average Predictions**|

**Key insight**: The primary distinction is **where** you make inference (typical case vs population), not the historical naming conventions.

## Primary API (public)

Two main functions organized by evaluation strategy:

### Population-based Analysis
```julia
population_margins(model, data; 
    type = :effects,        # :effects or :predictions
    vars = :continuous,     # variables for effects (ignored for predictions)
    target = :mu,           # :mu or :eta for effects  
    scale = :response,      # :response or :link for predictions
    weights = nothing,      # user weights
    balance = :none,        # :none | :all | Vector{Symbol}
    over = nothing,         # grouping variables
    within = nothing,       # nested grouping
    by = nothing,           # stratification
    vcov = :model          # covariance specification
)
```

### Profile-based Analysis  
```julia
profile_margins(model, data;
    at = :means,            # :means | Dict | Vector{Dict}
    type = :effects,        # :effects or :predictions  
    vars = :continuous,     # variables for effects (ignored for predictions)
    target = :mu,           # :mu or :eta for effects
    scale = :response,      # :response or :link for predictions
    average = false,        # collapse profiles to single summary
    over = nothing,         # grouping variables
    by = nothing,           # stratification  
    vcov = :model          # covariance specification
)

```

## Parameter Reference

### Core Parameters
- **`type`**: `:effects` (derivatives/slopes) or `:predictions` (levels/values)
- **`vars`**: variables for marginal effects; `:continuous` auto-detects (ignored for predictions)
- **`target`**: `:mu` (response scale) or `:eta` (link scale) for effects
- **`scale`**: `:response` or `:link` for predictions (follows GLM conventions)
- **`at`**: `:means` or `Dict`/`Vector{Dict}` for explicit profiles (profile_margins only)

### Inference and Grouping  
- **`vcov`**: `:model` | `AbstractMatrix` | `Function` | `Estimator` (CovarianceMatrices-aware)
- **`weights`**: user-provided observation weights
- **`balance`**: `:none` | `:all` | `Vector{Symbol}` - balance factor distributions
- **`over`**: grouping variables for within-group analysis
- **`within`**: nested grouping structure
- **`by`**: stratification variables for separate analyses
- **`average`**: collapse profile grid to summary (profile_margins only)

## Implementation Strategy

The two functions map to distinct computational paths:

### Population Path (population_margins)
- **Row-wise computation**: Effects/predictions calculated per observation, then averaged
- **Supports**: User weights, balanced sampling, complex grouping structures  
- **Performance**: Zero-allocation FD backend for large-sample AME
- **Use case**: True population parameters across heterogeneous samples

### Profile Path (profile_margins)  
- **Profile evaluation**: Effects/predictions at specific covariate combinations
- **Supports**: Representative value grids, profile averaging, MER/APR extensions
- **Performance**: AD backend for accuracy at explicit profiles
- **Use case**: Concrete interpretable cases, focal variable heterogeneity

## Examples

### Basic Usage
```julia
# Population average marginal effects (AME/APE)
population_margins(m, df; type=:effects, vars=:continuous, target=:mu)

# Population average predictions  
population_margins(m, df; type=:predictions, scale=:response)

# Effects at sample means (MEM)
profile_margins(m, df; at=:means, type=:effects, vars=[:x, :z], target=:mu)

# Predictions at specific profiles (APR-style)
profile_margins(m, df; at=Dict(:x=>[-1,0,1], :group=>["A","B"]), 
                type=:predictions, scale=:response)
```

### Advanced Features
```julia
# Population effects with grouping and weights
population_margins(m, df; type=:effects, vars=[:education, :experience], 
                   over=:region, weights=:survey_weight, balance=:all)

# Profile grid with averaging (MER-style focal variable analysis)
profile_margins(m, df; at=Dict(:education=>[8,12,16,20], :experience=>[mean]),
                type=:effects, average=false)  # One row per education level

# Robust standard errors via CovarianceMatrices
population_margins(m, df; type=:effects, vcov=HC1())
```

### Mapping from Traditional Names
```julia
# Traditional AME → population_margins with type=:effects
# Traditional APE → population_margins with type=:effects  
# Traditional MEM → profile_margins with at=:means, type=:effects
# Traditional MER → profile_margins with at=Dict(...), type=:effects
# Traditional APM → profile_margins with at=:means, type=:predictions
# Traditional APR → profile_margins with at=Dict(...), type=:predictions
```

## Migration Strategy

### From Current API
1. **Deprecate `margins()`**: Provide deprecation warnings pointing to new functions
2. **Transition period**: Both APIs coexist with clear migration guidance
3. **Legacy compatibility**: Optional wrapper functions for historical names if needed

### Key Changes
- **`dydx` → `vars`**: More intuitive parameter name
- **`mode` eliminated**: Replaced by `type` parameter within each function
- **`asbalanced` → `balance`**: Cleaner naming
- **Conceptual clarity**: Users choose evaluation strategy first, then output type

## Documentation Updates

### Primary Documentation  
- Lead with profile vs population conceptual framework
- Emphasize the fundamental choice: "Where do you want to make inference?"
- Position traditional names (AME/MEM/etc.) as historical context

### Migration Guide
- Clear mapping from old API to new API
- Side-by-side examples showing equivalent calls
- Performance implications of each approach

## Testing Strategy

- **Correctness**: Verify new API produces identical results to current implementation
- **Coverage**: Test all combinations of type/target/scale parameters  
- **Edge cases**: Profile specifications, grouping interactions, robust SEs
- **Performance**: Maintain zero-allocation guarantees for population path

## Benefits of This Approach

1. **Conceptual clarity**: Aligns API with statistical framework
2. **Reduced complexity**: 2 functions instead of 6, fewer parameters per function
3. **Future extensibility**: Natural place for new profile types or effect measures
4. **Performance optimization**: Clear separation enables targeted optimizations  
5. **Pedagogical value**: Teaches fundamental statistical concepts
