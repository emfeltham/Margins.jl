# Profile Margins

Profile margins compute marginal effects or predictions at specific covariate profiles (combinations of variable values). This is ideal for understanding effects at representative scenarios like "typical cases" or specific policy counterfactuals.

## Reference Grid Approach

Margins.jl uses an explicit reference grid approach for maximum flexibility and clarity. The core function signature is:

```julia
profile_margins(model, data, reference_grid; type=:effects, vars=nothing, ...)
```

## Reference Grid Builders

The package provides several builder functions to create reference grids easily:

### 1. Sample Means - `means_grid()`

The most common case: effects/predictions at sample means for continuous variables and frequency-weighted mixtures for categorical variables.

```julia
# Effects at sample means (MEM)
result = profile_margins(model, data, means_grid(data); type=:effects)

# Predictions at sample means (APM)
result = profile_margins(model, data, means_grid(data); type=:predictions)
```

### 2. Cartesian Product - `cartesian_grid()`

Create all combinations of specified values across variables:

```julia
# 3×2 = 6 scenarios: all combinations of x and education values
result = profile_margins(model, data, 
    cartesian_grid(data; x=[-1, 0, 1], education=["High School", "College"]); 
    type=:effects)

# Single variable varying, others at typical values
result = profile_margins(model, data,
    cartesian_grid(data; age=20:10:70); 
    type=:predictions)
```

### 3. Balanced Factorial - `balanced_grid()`

Create balanced (equal-weight) mixtures for categorical variables:

```julia
# Balanced factorial design for categorical variables
result = profile_margins(model, data,
    balanced_grid(data; education=:all, region=:all); 
    type=:effects)
```

### 4. Quantile-Based - `quantile_grid()`

Use quantiles of continuous variables:

```julia
# Effects at income quartiles
result = profile_margins(model, data,
    quantile_grid(data; income=[0.25, 0.5, 0.75]); 
    type=:effects)
```

## DataFrame Input (Maximum Control)

For complete control, provide a DataFrame directly:

```julia
# Custom reference grid
reference_grid = DataFrame(
    age=[25, 35, 45], 
    education=["High School", "College", "Graduate"],
    experience=[2, 8, 15]
)
result = profile_margins(model, data, reference_grid; type=:effects)
```

## Key Features

### Frequency-Weighted Categorical Defaults

When categorical variables are unspecified, they use actual population composition rather than arbitrary defaults:

```julia
# Your data: education = 40% HS, 45% College, 15% Graduate
#           region = 75% Urban, 25% Rural

# Sample means with realistic categorical composition
result = profile_margins(model, data, means_grid(data); type=:effects)
# → age: sample mean
# → education: frequency-weighted mixture (40% HS, 45% College, 15% Graduate)  
# → region: frequency-weighted mixture (75% Urban, 25% Rural)
```

### Categorical Mixtures

Use `mix()` for fractional specifications in reference grids:

```julia
using Margins: mix

# Policy scenario with specific treatment rates
reference_grid = DataFrame(
    age=[35, 45, 55],
    treated=[mix(0 => 0.3, 1 => 0.7)]  # 70% treatment rate
)
result = profile_margins(model, data, reference_grid; type=:predictions)
```

### Elasticity Analysis

Compute elasticities at specific profiles:

```julia
# Elasticities at sample means
result = profile_margins(model, data, means_grid(data); 
    type=:effects, measure=:elasticity)

# Semi-elasticities at specific scenarios
result = profile_margins(model, data,
    cartesian_grid(data; income=[25000, 50000, 75000]); 
    type=:effects, measure=:semielasticity_dyex)
```

## Performance Characteristics

Profile margins achieve **O(1) constant-time complexity** - execution time is independent of dataset size:

```julia
# Same computational cost regardless of data size
@time profile_margins(model, small_data, means_grid(small_data))   # ~100μs
@time profile_margins(model, large_data, means_grid(large_data))   # ~100μs

# Complex scenarios also maintain O(1) scaling
scenarios = cartesian_grid(data; x1=[0,1,2], x2=[10,20,30], group=["A","B"])  # 18 profiles
@time profile_margins(model, huge_data, scenarios)  # Still ~100μs
```

## Migration from Old API

If you have code using the deprecated `at` parameter:

```julia
# OLD (deprecated):
profile_margins(model, data; at=:means, type=:effects)
profile_margins(model, data; at=Dict(:x => [0,1,2]), type=:effects)

# NEW (current):
profile_margins(model, data, means_grid(data); type=:effects)  
profile_margins(model, data, cartesian_grid(data; x=[0,1,2]); type=:effects)
```

## Statistical Notes

- **Standard errors**: Computed via delta method using full model covariance matrix
- **Categorical effects**: Use baseline contrasts vs reference levels at each profile
- **Profile interpretation**: More concrete than population averages, ideal for policy communication
- **Computational efficiency**: Single compilation per analysis, reused across all profiles

## Examples

### Basic Workflow

```julia
using DataFrames, GLM, Margins

# Fit model
model = lm(@formula(y ~ x1 + x2 + group), data)

# Effects at sample means
mem_results = profile_margins(model, data, means_grid(data); type=:effects)
DataFrame(mem_results)

# Predictions at specific scenarios
scenarios = cartesian_grid(data; x1=[0, 1, 2], group=["A", "B"])
predictions = profile_margins(model, data, scenarios; type=:predictions)
DataFrame(predictions)
```

### Policy Analysis

```julia
# Current scenario: actual data composition
current = profile_margins(model, data, means_grid(data); type=:predictions)

# Policy scenario: increased education levels
policy_grid = DataFrame(
    x1=mean(data.x1),
    education=mix("High School" => 0.2, "College" => 0.5, "Graduate" => 0.3)
)
future = profile_margins(model, data, policy_grid; type=:predictions)

# Compare scenarios
policy_effect = DataFrame(future).estimate[1] - DataFrame(current).estimate[1]
```

See also: [`population_margins`](@ref) for population-averaged analysis.