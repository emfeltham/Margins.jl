# Profile Margins

Profile margins compute marginal effects or predictions at specific covariate profiles (combinations of variable values). This is ideal for understanding effects at representative scenarios like "typical cases" or specific policy counterfactuals.

## Three-Tier API Design

Margins.jl provides three complementary approaches to profile margins, each optimized for different use cases:

### 1. Direct Builders (Canonical)
The most powerful and composable approach using specialized builder functions:

```julia
# Means profile - effects/predictions at sample means
profiles = refgrid_means(data_nt)
result = profile_margins(model, data, profiles; type=:effects)

# Cartesian product - full factorial design
profiles = refgrid_cartesian((x=[-1,0,1], education=["High School", "College"]), data_nt)
result = profile_margins(model, data, profiles; type=:effects)

# Sequence - focal variable varying, others fixed
profiles = refgrid_sequence(:age, 20:10:70, data_nt; others=:means)
result = profile_margins(model, data, profiles; type=:predictions)
```

### 2. `at=` Parameter (Convenience)
Familiar syntax for quick analysis:

```julia
# Sample means
result = profile_margins(model, data; at=:means, type=:effects)

# Custom profiles  
result = profile_margins(model, data; at=(age=[25,35,45], education=["College"]), type=:effects)

# With grouping
result = profile_margins(model, data; at=:means, over=:gender, type=:predictions)
```

### 3. DataFrame Input (Flexibility)
Direct control over the exact reference grid:

```julia
# Custom reference grid
grid = DataFrame(age=[25,35,45], education=["College","College","College"])
result = profile_margins(model, data, grid; type=:effects)
```

## Builder Functions

All builders return `Iterator{Dict{Symbol,Any}}` for memory-efficient computation and accept optional `over` parameter for grouping.

### `refgrid_means`
Single profile with continuous variables at means, categoricals at first level:

```julia
profiles = refgrid_means(data_nt)
profiles = refgrid_means(data_nt; vars=[:age, :income])  # Subset of variables
profiles = refgrid_means(data_nt; over=:gender)  # Grouped by gender
```

### `refgrid_cartesian`  
Full factorial Cartesian product from specification:

```julia
# Basic Cartesian product
profiles = refgrid_cartesian((
    age = [25, 45, 65],
    education = ["High School", "College"] 
), data_nt)

# With grouping - profiles computed within each group
profiles = refgrid_cartesian((age=[25,45],), data_nt; over=:region)
```

### `refgrid_sequence`
Focal variable sequence with others held constant:

```julia
# Age varying 20-70, others at means
profiles = refgrid_sequence(:age, 20:10:70, data_nt; others=:means)

# Age varying, specific values for other variables
profiles = refgrid_sequence(:age, [25,35,45], data_nt; 
    others=Dict(:education => "College", :experience => 10))
```

### `refgrid_quantiles`
Variables at specific quantiles:

```julia
# Income at 10th, 50th, 90th percentiles
profiles = refgrid_quantiles(data_nt; specs=(income=[:p10,:p50,:p90],))

# Multiple variables at quantiles
profiles = refgrid_quantiles(data_nt; specs=(
    income = [:p25, :p75],
    age = [:p10, :p90]
))
```

### `refgrid_levels`
Categorical variable at specific levels:

```julia
# Education at all levels, others at means
profiles = refgrid_levels(data_nt; var=:education, 
    levels=["High School", "College", "Graduate"])

# With specific values for other variables  
profiles = refgrid_levels(data_nt; var=:treatment,
    levels=["Control", "Treatment"], 
    others=Dict(:age => 40, :income => 50000))
```

## Common Use Cases

### Policy Analysis
```julia
# Compare treatment effects across age groups
profiles = refgrid_cartesian((
    treatment = ["Control", "Treatment"],
    age = [25, 45, 65]
), data_nt)
result = profile_margins(model, data, profiles; type=:effects, vars=[:treatment])
```

### Elasticity Analysis
```julia
# Price elasticity at different income levels
profiles = refgrid_quantiles(data_nt; specs=(income=[:p10,:p25,:p50,:p75,:p90],))
result = profile_margins(model, data, profiles; 
    type=:effects, vars=[:price], measure=:elasticity)
```

### Representative Case Analysis
```julia
# Typical case: means for continuous, mode for categorical
profiles = refgrid_means(data_nt)
result = profile_margins(model, data, profiles; type=:predictions)

# Young vs old comparison
profiles = refgrid_cartesian((age=[25,65],), data_nt)
result = profile_margins(model, data, profiles; type=:predictions)
```

### Grouped Analysis
```julia
# Effects by region
profiles = refgrid_means(data_nt; over=:region)
result = profile_margins(model, data, profiles; type=:effects)

# Age effects within education groups
profiles = refgrid_sequence(:age, 20:10:70, data_nt; over=:education)
result = profile_margins(model, data, profiles; type=:effects, vars=[:age])
```

## Migration Guide

### From Parameter Soup to Builders

**Old approach (parameter-heavy):**
```julia
# Multiple scattered parameters
result = profile_margins(model, data; 
    at=(x=[1,2,3],), over=:group, within=:category, 
    type=:effects, average=true)
```

**New approach (builder-based):**
```julia
# Grouping built into the profile source
profiles = refgrid_cartesian((x=[1,2,3],), data_nt; over=:group)
result = profile_margins(model, data, profiles; type=:effects, average=true)
```

### Common Patterns

| Old Pattern | New Builder Approach |
|-------------|---------------------|
| `at=:means` | `refgrid_means(data_nt)` |
| `at=(x=[1,2,3],)` | `refgrid_cartesian((x=[1,2,3],), data_nt)` |
| `at=:means, over=:group` | `refgrid_means(data_nt; over=:group)` |
| Custom DataFrame | Direct DataFrame input unchanged |

### Benefits of Builder Approach

1. **Composability**: Mix and match builders with grouping as needed
2. **Type Safety**: All builders return same `Iterator{Dict{Symbol,Any}}` type  
3. **Memory Efficiency**: Large grids stream without materialization
4. **Clarity**: "Build profiles → compute margins" mental model
5. **Extensibility**: New patterns without API changes

## Advanced Features

### Averaging Profiles
```julia
# Compute average effects across profiles
profiles = refgrid_cartesian((age=[25,35,45], gender=["M","F"]), data_nt)
result = profile_margins(model, data, profiles; type=:effects, average=true)
```

### Custom Covariance
```julia
# Robust standard errors
result = profile_margins(model, data, profiles; vcov=:robust)

# Custom covariance matrix
custom_vcov = compute_custom_vcov(model, data)
result = profile_margins(model, data, profiles; vcov=custom_vcov)
```

### Multiple Comparison Adjustments
```julia
# Bonferroni adjustment for multiple profiles
result = profile_margins(model, data, profiles; mcompare=:bonferroni)
```

## Statistical Notes

- **Standard Errors**: All results use proper delta-method standard errors with full covariance matrix
- **Deterministic Ordering**: Results follow by → over → profiles → terms order
- **Error-First Policy**: Statistical failures produce clear errors rather than invalid results
- **Publication Grade**: All confidence intervals and p-values meet econometric standards

## Performance

- **Zero Allocation**: Large profile grids stream without intermediate DataFrame materialization
- **Microsecond Timing**: Individual profile computations complete in microseconds
- **Memory Efficient**: Grouping handled within builders, not in API parameters