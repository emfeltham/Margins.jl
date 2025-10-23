# Profile-Specific Marginal Effects Analysis

Profile-specific marginal effects analysis evaluates marginal quantities at predetermined points within the covariate space, providing inference for representative scenarios and policy counterfactuals. This methodological approach facilitates concrete interpretation of marginal effects through evaluation at theoretically motivated or practically relevant covariate combinations.

## Reference Grid Methodology

The implementation employs an explicit reference grid specification to ensure methodological transparency and computational flexibility. The core analytical function utilizes the following signature specification:

```julia
profile_margins(model, data, reference_grid; type=:effects, vars=nothing, ...)
```

## Reference Grid Construction Framework

The package implements systematic reference grid construction through specialized builder functions that accommodate diverse analytical requirements:

### 1. Sample Means - `means_grid()`

The canonical approach evaluates marginal quantities at empirical sample means for continuous variables while incorporating frequency-weighted probability mixtures for categorical variables.

```julia
# Marginal effects at sample means (MEM)
result = profile_margins(model, data, means_grid(data); type=:effects)

# Adjusted predictions at sample means (APM)
result = profile_margins(model, data, means_grid(data); type=:predictions)
```

### 2. Cartesian Product - `cartesian_grid()`

Systematic construction of reference grids through Cartesian product enumeration of specified covariate values across multiple dimensions:

```julia
# Complete factorial design: 3×2 = 6 scenarios
result = profile_margins(model, data, 
    cartesian_grid(x=[-1, 0, 1], education=["High School", "College"]); 
    type=:effects)

# Single-variable sensitivity analysis with typical values for remaining covariates
result = profile_margins(model, data,
    cartesian_grid(age=20:10:70);
    type=:predictions)
```

**Completing Partial Grids with `complete_reference_grid()`**

When creating reference grids with `cartesian_grid()`, you may specify only a subset of model variables. The `complete_reference_grid()` function automatically fills in typical values for any missing variables:

```julia
# Create partial grid focusing on variables of interest
partial_grid = cartesian_grid(
    x1=[0, 1, 2],      # Only specify x1 and x2
    x2=[10, 20]
)
# Result: 3×2 = 6 rows, but missing x3, x4, etc. from model

# Complete the grid with typical values for missing variables
complete_grid = complete_reference_grid(partial_grid, model, data)
# Result: 6 rows with x1, x2 specified AND x3=mean(data.x3), x4=mean(data.x4), etc.

# Use directly with profile_margins
result = profile_margins(model, data, complete_grid; type=:effects)

# Customize the typical value function (default is mean)
complete_grid = complete_reference_grid(partial_grid, model, data; typical=median)
# Now uses median instead of mean for missing variables
```

**Use Cases:**
- **Focused Analysis**: Specify only key variables of interest, let the function handle the rest
- **Sensitivity Analysis**: Vary 1-2 variables while holding others at representative values
- **Presentation**: Simplify grid specification for cleaner, more maintainable code
- **Custom Typical Values**: Use median, mode, or custom functions for missing variables

**Important:** This function only works with partial grids. If your grid already specifies all model variables, `complete_reference_grid()` simply returns it unchanged.

### 3. Balanced Factorial - `balanced_grid()`

Construction of balanced factorial designs utilizing equal-weight probability mixtures for categorical variable specifications:

```julia
# Balanced factorial design for comprehensive categorical analysis
result = profile_margins(model, data,
    balanced_grid(data; education=:all, region=:all); 
    type=:effects)
```

### 4. Quantile-Based - `quantile_grid()`

Reference grid construction based on empirical quantiles of continuous variable distributions:

```julia
# Marginal effects evaluated at income distribution quartiles
result = profile_margins(model, data,
    quantile_grid(data; income=[0.25, 0.5, 0.75]); 
    type=:effects)
```

### 5. Hierarchical Grammar - `hierarchical_grid()`

Systematic reference grid construction using the group nesting grammar (`=>` operator) for complex multi-dimensional covariate scenario analysis:

```julia
# Complex hierarchical specification with multiple representative types
reference_spec = :region => [
    (:income, :quartiles),  # Income quartiles within each region
    (:age, :mean),          # Mean age within each region
    :education              # All education levels within each region
]
result = profile_margins(model, data, 
    hierarchical_grid(data, reference_spec); 
    type=:effects)

# Deep hierarchical nesting for comprehensive policy analysis
policy_spec = :country => (:region => (:education => [(:income, :quintiles), (:age, :mean)]))
result = profile_margins(model, data,
    hierarchical_grid(data, policy_spec; max_depth=4); 
    type=:predictions)
```

## Direct DataFrame Specification (Complete Analytical Control)

Maximum analytical flexibility is achieved through direct DataFrame specification of reference grid points:

```julia
# Custom reference grid with explicit covariate specifications
reference_grid = DataFrame(
    age=[25, 35, 45], 
    education=["High School", "College", "Graduate"],
    experience=[2, 8, 15]
)
result = profile_margins(model, data, reference_grid; type=:effects)
```

## Advanced Methodological Features

### Population-Representative Categorical Composition

The implementation addresses the methodological limitation of arbitrary baseline category selection through empirical frequency-weighted categorical mixtures that reflect actual population composition:

```julia
# Data characteristics: education = 40% HS, 45% College, 15% Graduate
#                      region = 75% Urban, 25% Rural

# Sample means incorporating realistic categorical composition
result = profile_margins(model, data, means_grid(data); type=:effects)
# → age: empirical sample mean
# → education: frequency-weighted mixture (40% HS, 45% College, 15% Graduate)  
# → region: frequency-weighted mixture (75% Urban, 25% Rural)
```

### Fractional Categorical Specifications

Policy counterfactual analysis utilizes fractional categorical specifications through the categorical mixture interface:

```julia
using Margins: mix

# Policy scenario incorporating specific treatment probability distributions
reference_grid = DataFrame(
    age=[35, 45, 55],
    treated=[mix(0 => 0.3, 1 => 0.7)]  # 70% treatment probability
)
result = profile_margins(model, data, reference_grid; type=:predictions)
```

### Profile-Specific Elasticity Analysis

Elasticity computation at predetermined covariate profiles enables sensitivity analysis across representative scenarios:

```julia
# Elasticities evaluated at sample means
result = profile_margins(model, data, means_grid(data); 
    type=:effects, measure=:elasticity)

# Semi-elasticities across income distribution quantiles
result = profile_margins(model, data,
    cartesian_grid(income=[25000, 50000, 75000]); 
    type=:effects, measure=:semielasticity_dyex)
```

## Computational Performance Analysis

Profile-specific marginal effects analysis exhibits constant-time computational complexity with execution time independent of dataset dimensionality:

```julia
# Computational cost remains invariant to dataset size
@time profile_margins(model, small_data, means_grid(small_data))   # ~100μs
@time profile_margins(model, large_data, means_grid(large_data))   # ~100μs

# Complex factorial designs maintain constant-time scaling properties
scenarios = cartesian_grid(x1=[0,1,2], x2=[10,20,30], group=["A","B"])  # 18 profiles
@time profile_margins(model, huge_data, scenarios)  # Maintains ~100μs complexity
```

## Migration from Old API

If you have code using the deprecated `at` parameter:

```julia
# OLD (deprecated):
profile_margins(model, data; at=:means, type=:effects)
profile_margins(model, data; at=Dict(:x => [0,1,2]), type=:effects)

# NEW (current):
profile_margins(model, data, means_grid(data); type=:effects)  
profile_margins(model, data, cartesian_grid(x=[0,1,2]); type=:effects)
```

## Statistical Notes

- **Standard errors**: Computed via delta method using full model covariance matrix
- **Categorical effects**: Use baseline contrasts vs reference levels at each profile
- **Profile interpretation**: More concrete than population averages, ideal for policy communication
- **Computational efficiency**: Single compilation per analysis, reused across all profiles

## Output Structure and Result Multiplicity

### Result Row Generation Pattern
The number of output rows depends on variable types and categorical levels:

- **Continuous variables**: 1 reference grid row → 1 output row (derivative)
- **Categorical variables**: 1 reference grid row → Multiple output rows (baseline contrasts)

### Example Output Multiplicity
```julia
# Reference grid with 2 profiles
grid = DataFrame(
    age = [30, 40],                    # Continuous variable
    education = ["A", "B", "C"]        # Categorical with 3 levels
)

# Results: 2 profiles × (1 continuous + 2 categorical contrasts) = 6 total rows:
result = profile_margins(model, data, grid; type=:effects, vars=[:age, :education])

# Output structure:
# Row 1: age at (age=30, education="A,B,C")
# Row 2: education: B - A at (age=30, education="A,B,C")
# Row 3: education: C - A at (age=30, education="A,B,C")
# Row 4: age at (age=40, education="A,B,C")
# Row 5: education: B - A at (age=40, education="A,B,C")
# Row 6: education: C - A at (age=40, education="A,B,C")
```

### Categorical Baseline Contrasts
For a categorical variable with K levels, each reference grid row generates:
- **K-1 contrast rows** comparing each non-baseline level to the baseline
- Term names follow pattern: `"variable: level - baseline"`
- Example: 4-level categorical produces 3 contrasts (B - A, C - A, D - A)

This multiplicative pattern means that reference grids with multiple categorical variables can generate substantial numbers of result rows, enabling comprehensive categorical effect analysis at each profile.

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
scenarios = cartesian_grid(x1=[0, 1, 2], group=["A", "B"])
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