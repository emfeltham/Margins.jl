# Reference Grid Methodology and Implementation

**Reference Grids** 
A reference grid is simply a table that says "compute effects for people with these specific characteristics." For example:
- Age: 30, Education: College → "What's the effect for 30-year-old college graduates?"
- Age: 40, Education: High School → "What's the effect for 40-year-old high school graduates?"

The package provides helper functions to create these tables automatically.

Reference grid specification constitutes the methodological foundation for covariate scenario definition in profile-based marginal effects analysis. The implementation provides a systematic framework for scenario specification through structured builder functions and direct tabular specification interfaces.

## Methodological Foundation

The analytical framework employs explicit reference grid specification to ensure transparency and computational precision:

```julia
profile_margins(model, data, reference_grid; type=:effects, ...)
```

The `reference_grid` parameter accepts DataFrame specifications that enumerate the covariate combinations where marginal effects are computed.

## Typical values

Builders fill variables with **typical values**, computed by one function (`Margins.typical_value`) everywhere: `means_grid`, `balanced_grid` (columns it does not balance), `quantile_grid` (other columns), `hierarchical_grid` (`(:x, :mean)`, `(:x, :median)`, `:mix_proportional` and the variables a combination leaves out), `complete_reference_grid` and the completion inside `profile_margins`, and the second differences' `at = :mean` / `:median`.

| column | typical value |
|:--|:--|
| numeric | `typical(col)`: `mean` (default), `median`, or any function |
| `Bool` | share of `true` (a number in [0, 1]) |
| categorical, string | frequency mixture of the observed values, e.g. `mix("a" => 0.3, "b" => 0.7)` |

**Weighted typical values.** With `weights = :w` (a column) or a vector, the builders compute:

- the **weighted mean** `Σ wᵢxᵢ / Σ wᵢ`;
- the **weighted median**: sort the positively weighted values, with cumulative weights Cₖ and total W; the median is the smallest value with Cₖ ≥ W/2, and when Cₖ equals W/2 exactly, the midpoint of that value and the next. With equal weights this is `Statistics.median` (including the average of the two middle values for an even count); with integer weights it is the median of the data with each row repeated wᵢ times; rescaling the weights does not change it;
- **weighted shares** for `Bool` columns and categorical mixtures (the weight of each value over the total);
- **weighted quantiles** (`quantile_grid`, `hierarchical_grid` quantile specifications): the weighted analogue of `Statistics.quantile` (type 7) that weighted `groups = (:x, n)` bins use. It equals `Statistics.quantile` with equal weights and interpolates, so at probability 0.5 it can differ from the weighted median.

Weights are checked as in `population_margins`: real, finite, non‑negative, not all zero. A row with weight 0 contributes nothing (the result is the one without that row). With all weights equal, the result is the unweighted one.

**Custom `typical` functions cannot be weighted.** Only `mean` and `median` have weighted versions; `weights` with any other `typical` is an `ArgumentError` rather than silently ignoring the weights. Compute such a value yourself and put it in the grid (`cartesian_grid(x = [my_stat(df.x, df.w)])`).

**Weighted MEM for a survey-weighted model.** `profile_margins` evaluates the grid it is given and takes no `weights`. Build the grid with weights:

```julia
m = glm(@formula(y ~ x + z + group), df, Binomial(), LogitLink(); wts = df.w)
profile_margins(m, df, means_grid(df; weights = :w))   # x, z at weighted means; group a weighted mixture

# Partial grid: complete it with weighted values before profile_margins (which would fill unweighted)
grid = complete_reference_grid(cartesian_grid(x = [0, 1]), m, df; weights = :model)
profile_margins(m, df, grid)
```

`weights = :model` (the model's estimation weights) is accepted by `complete_reference_grid` only, the one builder that takes a model. A `Symbol` names a column of the data; a vector is used as given.

## Row and column order

Grids have a documented, deterministic order:

- **Columns:** `cartesian_grid` follows the keyword order (a `Dict` argument, which has no order, is taken alphabetically); `means_grid`, `balanced_grid`, `quantile_grid` and `hierarchical_grid`, which return every data column, follow the data's column order; `complete_reference_grid` keeps the grid's columns in their order and appends the filled-in variables in the data's column order.
- **Rows:** a Cartesian product varies the **first variable slowest** and the last fastest (`cartesian_grid(a = [1, 2], b = [3, 4])` gives `(1,3), (1,4), (2,3), (2,4)`), each variable's values in the order given. Categorical levels (`hierarchical_grid(data, :g)`) are in level order (`levels(col)`); `Bool` and string columns ascending.
- `profile_margins` results keep the grid's row order and column order.

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

# Weighted means, medians and shares (e.g. survey weights in column :w)
grid = means_grid(data; weights=:w)
```

**Output structure:**
- **Continuous variables**: Sample mean (or custom typical function)
- **Categorical variables**: Frequency-weighted mixture based on actual data distribution
- **Bool variables**: Probability of true (proportion of true values)

### 2. Cartesian Product - `cartesian_grid(vars...)`

Creates all combinations of specified values across variables:

```julia
# 3×2 = 6 scenarios: all combinations of x and education values
grid = cartesian_grid(x=[-1, 0, 1], education=["High School", "College"])
result = profile_margins(model, data, grid; type=:effects)

# Single variable varying, others at typical values
grid = cartesian_grid(age=20:10:70)
result = profile_margins(model, data, grid; type=:predictions)

# Complex scenarios with multiple variables
grid = cartesian_grid(
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

### 5. Hierarchical Grammar - `hierarchical_grid(data, spec)`

Creates systematic reference grids using the group nesting grammar (`=>` operator) for complex multi-dimensional covariate scenario construction:

```julia
# Simple hierarchical: region-specific education representatives
spec = :region => :education
grid = hierarchical_grid(data, spec)
result = profile_margins(model, data, grid; type=:effects)

# Complex hierarchy with multiple representative types
spec = :region => [
    (:income, :quartiles),  # Income quartiles within each region
    (:age, :mean),          # Mean age within each region  
    :education              # All education levels within each region
]
grid = hierarchical_grid(data, spec)
result = profile_margins(model, data, grid; type=:effects)

# Deep nesting (3+ levels) with automatic safety validation
spec = :country => (
    :region => (
        :education => [(:income, :quartiles), (:age, :mean)]
    )
)
grid = hierarchical_grid(data, spec; max_depth=4, warn_large=true)
result = profile_margins(model, data, grid; type=:effects)
```

**Safety Parameters:**

`hierarchical_grid()` includes built-in safety features to prevent accidental creation of excessively large grids:

- **`max_depth::Int=5`** - Maximum allowed nesting depth
  - Default: 5 levels
  - Prevents runaway nesting that could create enormous grids
  - Error thrown if specification exceeds this depth
  - Example: `hierarchical_grid(data, deep_spec; max_depth=10)` allows up to 10 levels

- **`warn_large::Bool=true`** - Grid size warnings
  - Default: enabled (shows warnings)
  - Estimates total grid size before construction
  - Warns if grid will exceed 10,000 rows
  - Helps catch specification errors before expensive computation
  - Example: `hierarchical_grid(data, spec; warn_large=false)` disables warnings

```julia
# Example: deeply nested specification with safety overrides
very_deep_spec = :country => (
    :state => (
        :county => (
            :city => [:education, (:income, :quartiles)]
        )
    )
)

# This will error without increasing max_depth (default is 5)
# grid = hierarchical_grid(data, very_deep_spec)  # Error: exceeds max_depth

# Allow deeper nesting explicitly
grid = hierarchical_grid(data, very_deep_spec; max_depth=10, warn_large=true)
```

**Advanced Representative Types:**
```julia
# Statistical representatives within hierarchical groups
spec = :region => [
    (:income, :mean),           # Mean income per region
    (:income, :median),         # Median income per region
    (:income, :quartiles),      # Q1, Q2, Q3, Q4 per region
    (:income, :quintiles),      # Quintiles per region
    (:income, :deciles),        # Deciles per region
    (:income, [0.1, 0.5, 0.9]), # Custom percentiles per region
    (:age, [25, 45, 65]),       # Fixed representative ages
    (:score, (:range, 5))       # 5 evenly spaced points from min to max
]
grid = hierarchical_grid(data, spec)
```

**Mixture Integration:**
```julia
# Population-proportion mixtures for realistic scenarios
spec = :region => [
    (:education, :mix_proportional),  # Use actual data proportions
    (:income, :quartiles),
    (:age, :mean)
]
grid = hierarchical_grid(data, spec)

# Custom mixtures for policy analysis
using Margins: mix
spec = :region => [
    (:education, mix("HS" => 0.3, "College" => 0.7)),  # Policy scenario
    (:income, :median)
]
grid = hierarchical_grid(data, spec)
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

# Completion (here by profile_margins) uses realistic defaults, unweighted
grid = cartesian_grid(income=[30000, 50000, 70000])
# → income varies as specified
# → education: mix("HS" => 0.4, "College" => 0.45, "Graduate" => 0.15)
# → region: mix("Urban" => 0.75, "Rural" => 0.25)  
# → treated: 0.6 (probability of true)
```

### Hierarchical Policy Analysis

Systematic multi-dimensional policy evaluation using hierarchical grids:

```julia
# Complex policy analysis across administrative levels
policy_spec = :state => (
    :county => [
        (:education, :mix_proportional),     # Actual education composition per county
        (:income, :quintiles),               # Income distribution per county
        (:age, [25, 45, 65]),               # Key demographic groups
        :employment_status                   # All observed employment categories
    ]
)
grid = hierarchical_grid(data, policy_spec)
result = profile_margins(policy_model, data, grid; vars=[:policy_treatment])

# Comparative scenario analysis
baseline_spec = :region => [(:education, :mix_proportional), (:income, :mean)]
intervention_spec = :region => [(:education, mix("HS" => 0.2, "College" => 0.8)), (:income, :mean)]

baseline_grid = hierarchical_grid(data, baseline_spec)
intervention_grid = hierarchical_grid(data, intervention_spec)

baseline_results = profile_margins(model, data, baseline_grid; type=:predictions)
intervention_results = profile_margins(model, data, intervention_grid; type=:predictions)

# Calculate policy impact
baseline_df = DataFrame(baseline_results)
intervention_df = DataFrame(intervention_results)
policy_impact = intervention_df.estimate .- baseline_df.estimate
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
age_grid = cartesian_grid(age=25:5:65)
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
small_grid = cartesian_grid(x=[0, 1, 2])
@time profile_margins(model, huge_data, small_grid)  # ~150μs

# Large grid: 27 scenarios  
large_grid = cartesian_grid(x=[0,1,2], y=[0,1,2], z=[0,1,2])
@time profile_margins(model, huge_data, large_grid)  # ~400μs

# Dataset size doesn't matter
@time profile_margins(model, small_data, large_grid)  # Still ~400μs
```

### Hierarchical Grid Performance

Hierarchical grids provide automatic size estimation and safety validation:

```julia
# Automatic grid size warnings for large combinations
large_spec = :country => (:region => (:education => (:income, :deciles)))
# Warning: Estimated grid size ~50,000 combinations may impact performance
grid = hierarchical_grid(data, large_spec; warn_large=true)

# Depth protection prevents excessive nesting
deep_spec = :a => (:b => (:c => (:d => (:e => (:f => :g)))))
# Error: Nesting depth 7 exceeds maximum allowed depth 5
grid = hierarchical_grid(data, deep_spec; max_depth=5)

# Efficient construction through systematic generation
complex_spec = :region => [(:income, :quartiles), (:age, :mean), :education]
@time hierarchical_grid(data, complex_spec)  # ~50μs regardless of data size
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
huge_grid = cartesian_grid(x=1:100, y=1:100)  # 10,000 scenarios
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
grid3 = cartesian_grid(age=[mean(data.age)])

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

## Best Practices

1. **Start with `means_grid()`** for basic analysis
2. **Use `cartesian_grid()`** for systematic exploration
3. **Use `balanced_grid()`** for orthogonal factorial designs
4. **Use `quantile_grid()`** for distributional analysis
5. **Use `hierarchical_grid()`** for complex multi-dimensional policy analysis
6. **Use explicit DataFrame** for maximum custom control
7. **Validate grids** with small examples before scaling up
8. **Consider grid size** vs computational requirements
9. **Leverage frequency weighting** for realistic defaults
10. **Use mixture specifications** for policy counterfactual analysis

See also: [`profile_margins`](@ref) for the main function interface.