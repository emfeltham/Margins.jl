# Reference Grid Specification

Reference grids define the covariate scenarios where marginal effects and predictions are evaluated in profile-based analysis. Margins.jl provides a flexible system for specifying reference grids through builder functions and direct DataFrame specification.

## Core Approach

The fundamental interface uses explicit reference grids:

```julia
profile_margins(model, data, reference_grid; type=:effects, ...)
```

Where `reference_grid` is a DataFrame specifying the covariate combinations for analysis.

## Reference Grid Builders

### 1. Sample Means - `means_grid(data)`

Creates reference grid with sample means for continuous variables and frequency-weighted mixtures for categorical variables:

```julia
# Build grid with realistic defaults
grid = means_grid(data)
result = profile_margins(model, data, grid; type=:effects)

# Custom typical value function (default is mean)
grid = means_grid(data; typical=median)
result = profile_margins(model, data, grid; type=:effects)
```

**Output structure:**
- **Continuous variables**: Sample mean (or custom typical function)
- **Categorical variables**: Frequency-weighted mixture based on actual data distribution
- **Bool variables**: Probability of true (proportion of true values)

### 2. Cartesian Product - `cartesian_grid(data; vars...)`

Creates all combinations of specified values across variables:

```julia
# 3×2 = 6 scenarios: all combinations of x and education values
grid = cartesian_grid(data; x=[-1, 0, 1], education=["High School", "College"])
result = profile_margins(model, data, grid; type=:effects)

# Single variable varying, others at typical values
grid = cartesian_grid(data; age=20:10:70)
result = profile_margins(model, data, grid; type=:predictions)

# Complex scenarios with multiple variables
grid = cartesian_grid(data; 
    income=[25000, 50000, 75000],
    education=["HS", "College"],
    region=["North", "South"]
)  # Creates 3×2×2 = 12 scenarios
result = profile_margins(model, data, grid; type=:effects)
```

### 3. Balanced Factorial - `balanced_grid(data; vars...)`

Creates balanced (equal-weight) mixtures for categorical variables, useful for orthogonal factorial designs:

```julia
# Balanced factorial for categorical variables
grid = balanced_grid(data; education=:all, region=:all)
result = profile_margins(model, data, grid; type=:effects)

# Mixed specification
grid = balanced_grid(data; 
    education=:all,           # All levels with equal weight
    income=mean(data.income)  # Fixed at mean
)
result = profile_margins(model, data, grid; type=:effects)
```

### 4. Quantile-Based - `quantile_grid(data; vars...)`

Uses quantiles of continuous variables:

```julia
# Effects at income quartiles
grid = quantile_grid(data; income=[0.25, 0.5, 0.75])
result = profile_margins(model, data, grid; type=:effects)

# Multiple quantile specifications
grid = quantile_grid(data; 
    income=[0.1, 0.5, 0.9],
    age=[0.25, 0.75]
)  # Creates 3×2 = 6 scenarios
result = profile_margins(model, data, grid; type=:effects)
```

## Direct DataFrame Specification

For maximum control, create reference grids directly:

```julia
# Simple custom grid
reference_grid = DataFrame(
    age=[25, 35, 45], 
    education=["High School", "College", "Graduate"],
    experience=[2, 8, 15],
    treated=[true, false, true]
)
result = profile_margins(model, data, reference_grid; type=:effects)

# Grid with categorical mixtures
using Margins: mix

policy_grid = DataFrame(
    age=[35, 45, 55],
    education=[
        mix("HS" => 0.4, "College" => 0.6),        # Current composition
        mix("HS" => 0.2, "College" => 0.8),        # Policy scenario 1
        mix("HS" => 0.1, "College" => 0.9)         # Policy scenario 2
    ]
)
result = profile_margins(model, data, policy_grid; type=:predictions)
```

## Advanced Patterns

### Frequency-Weighted Defaults

When variables are unspecified in builder functions, they use actual data composition:

```julia
# Your data composition:
# - education: 40% HS, 45% College, 15% Graduate  
# - region: 75% Urban, 25% Rural
# - treated: 60% true, 40% false

# Builder uses realistic defaults
grid = cartesian_grid(data; income=[30000, 50000, 70000])
# → income varies as specified
# → education: mix("HS" => 0.4, "College" => 0.45, "Graduate" => 0.15)
# → region: mix("Urban" => 0.75, "Rural" => 0.25)  
# → treated: 0.6 (probability of true)
```

### Scenario Comparison

Compare different policy scenarios:

```julia
# Current scenario (status quo)
current_grid = means_grid(data)
current = profile_margins(model, data, current_grid; type=:predictions)

# Policy scenario (increased education)
policy_grid = DataFrame(
    age=mean(data.age),
    income=mean(data.income),
    education=mix("HS" => 0.2, "College" => 0.5, "Graduate" => 0.3)  # Policy target
)
future = profile_margins(model, data, policy_grid; type=:predictions)

# Compare outcomes
current_pred = DataFrame(current).estimate[1]
future_pred = DataFrame(future).estimate[1]
policy_impact = future_pred - current_pred
```

### Sequential Analysis

Analyze effects along ranges of key variables:

```julia
# Effects across age ranges
age_grid = cartesian_grid(data; age=25:5:65)
age_effects = profile_margins(model, data, age_grid; type=:effects, vars=[:education])

# Plot age-varying effects
using Plots
plot(25:5:65, DataFrame(age_effects).estimate, 
     xlabel="Age", ylabel="Education Effect", 
     title="Age-Varying Education Effects")
```

## Performance Considerations

### Grid Size and Efficiency

Reference grid size affects performance linearly, but is independent of dataset size:

```julia
# Small grid: 3 scenarios
small_grid = cartesian_grid(data; x=[0, 1, 2])
@time profile_margins(model, huge_data, small_grid)  # ~150μs

# Large grid: 27 scenarios  
large_grid = cartesian_grid(data; x=[0,1,2], y=[0,1,2], z=[0,1,2])
@time profile_margins(model, huge_data, large_grid)  # ~400μs

# Dataset size doesn't matter
@time profile_margins(model, small_data, large_grid)  # Still ~400μs
```

### Memory Management

Builder functions are optimized for memory efficiency:

```julia
# Efficient: builders avoid unnecessary allocations
grid = means_grid(large_data)  # O(1) memory for typical values

# Less efficient: explicit grids require full materialization  
explicit_grid = DataFrame(
    x1=fill(mean(large_data.x1), 1000),  # O(n) memory
    x2=fill(mean(large_data.x2), 1000)
)
```

## Validation and Error Handling

Reference grids are validated automatically:

```julia
# Error: Missing model variables
incomplete_grid = DataFrame(x1=[0, 1])  # Missing x2 from model
profile_margins(model, data, incomplete_grid)  
# → ArgumentError: Missing model variables: x2

# Error: Invalid categorical levels
invalid_grid = DataFrame(
    x1=[0, 1], 
    group=["InvalidLevel", "AnotherInvalid"]  # Not in original data
)
profile_margins(model, data, invalid_grid)
# → ArgumentError: Invalid levels for categorical variable 'group'

# Warning: Large grid size
huge_grid = cartesian_grid(data; x=1:100, y=1:100)  # 10,000 scenarios
profile_margins(model, data, huge_grid)
# → Warning: Large reference grid (10000 scenarios) may impact performance
```

## Statistical Properties

### Delta-Method Standard Errors

Standard errors are computed consistently across all reference grid types:

```julia
# Same statistical rigor regardless of grid construction method
grid1 = means_grid(data)
grid2 = DataFrame(age=mean(data.age), education=mode(data.education))
grid3 = cartesian_grid(data; age=[mean(data.age)])

# All use identical delta-method computation
result1 = profile_margins(model, data, grid1; type=:effects)
result2 = profile_margins(model, data, grid2; type=:effects)  
result3 = profile_margins(model, data, grid3; type=:effects)

# Standard errors are mathematically equivalent
all(DataFrame(result1).se .≈ DataFrame(result2).se .≈ DataFrame(result3).se)  # true
```

### Categorical Mixture Handling

Categorical mixtures are handled natively throughout the system:

```julia
# Fractional specifications work seamlessly
mixed_grid = DataFrame(
    age=[35, 45],
    treated=[0.3, mix(0 => 0.6, 1 => 0.4)]  # Mix of scalar and mixture
)
result = profile_margins(model, data, mixed_grid; type=:predictions)

# Standard errors account for mixture uncertainty automatically
DataFrame(result)  # Includes proper SEs for mixed scenarios
```

## Migration Guide

### From Old `at` Parameter Syntax

```julia
# OLD (deprecated):
profile_margins(model, data; at=:means)
profile_margins(model, data; at=Dict(:x => [0,1,2]))
profile_margins(model, data; at=[Dict(:x => 0), Dict(:x => 1)])

# NEW (current):
profile_margins(model, data, means_grid(data))
profile_margins(model, data, cartesian_grid(data; x=[0,1,2]))

explicit_grid = DataFrame(x=[0, 1])
profile_margins(model, data, explicit_grid)
```

### Builder Function Evolution

```julia
# OLD (deprecated internal names):
refgrid_means(data)
refgrid_cartesian(specs, data)

# NEW (exported public API):
means_grid(data)
cartesian_grid(data; vars...)
balanced_grid(data; vars...)
quantile_grid(data; vars...)
```

## Best Practices

1. **Start with `means_grid()`** for basic analysis
2. **Use `cartesian_grid()`** for systematic exploration
3. **Use `balanced_grid()`** for orthogonal factorial designs
4. **Use `quantile_grid()`** for distributional analysis
5. **Use explicit DataFrame** for complex custom scenarios
6. **Validate grids** with small examples before scaling up
7. **Consider grid size** vs computational requirements
8. **Leverage frequency weighting** for realistic defaults

See also: [`profile_margins`](@ref) for the main function interface.