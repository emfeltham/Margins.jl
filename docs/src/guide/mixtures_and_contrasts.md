# Categorical Mixtures and Manual Contrasts

## Overview

When using categorical mixtures in reference grids, the package follows a clear design principle:

**Mixtures represent population composition context, not discrete scenarios to contrast.**

This means categorical variables specified as mixtures (either explicit `CategoricalMixture` objects or implicit frequency-weighted values for Bool variables) are **skipped for automatic contrast computation** in `profile_margins()`.

## Why This Design?

### Mixtures are Context, Not Contrast Targets

A mixture specification like:
```julia
education = mix("hs" => 0.4, "college" => 0.6)
```

Represents: "40% high school educated, 60% college educated" — a weighted average population context.

Computing contrasts like "0.6 college-mixture - 0.4 hs-mixture" is clearly meaningful. Contrasts require discrete levels: "college - hs" is natural, but mixtures are less clearly so.

### Bool Variables with Fractional Values

Similarly, when Bool variables are filled with fractional values during grid completion (e.g., `0.796` representing 79.6% true rate), they are treated as mixture specifications and skipped for contrasts:

```julia
# Bool variable filled by grid completion
ref_grid = DataFrame(treatment = [false, true])
# other_bool_var gets filled with frequency (e.g., 0.65 for 65% true)
result = profile_margins(model, df, ref_grid; type=:effects)
# [ Info: Skipping contrasts for variable other_bool_var: specified as mixture...
```

## Recommended Workflow: APMs + Manual Contrasts

For computing contrasts between predictions at specific profiles with mixture context, use this two-step workflow:

### Step 1: Compute Adjusted Predictions at Profiles (APMs)

Use `profile_margins()` with `type=:predictions` to get predictions at specific profile points, with categorical mixtures providing population-weighted context:

```julia
using Margins, GLM, DataFrames, Statistics

# Define profiles with mixture context
ref_grid = DataFrame(
    treatment = [false, true],
    age = [mean(df.age), mean(df.age)]
    # education will be filled with frequency-weighted mixture
)

# Get predictions at these profiles
result = profile_margins(model, df, ref_grid; type=:predictions)
result_df = DataFrame(result; include_gradients=true)
```

**Key Point**: Include `include_gradients=true` to enable proper delta-method standard errors for contrasts.

### Step 2: Compute Contrasts Manually

Use the `contrast()` function to compute contrasts between specific rows:

```julia
# Contrast row 2 vs row 1 (treatment:true - treatment:false)
contrast_result = contrast(result_df, 2, 1, vcov(model))

println("Treatment effect: ", contrast_result.contrast)
println("Standard error: ", contrast_result.se)
println("p-value: ", contrast_result.p_value)
println("95% CI: [", contrast_result.ci_lower, ", ", contrast_result.ci_upper, "]")
```

### Complete Example

```julia
using Margins, GLM, DataFrames, StatsModels, CategoricalArrays, Statistics
using Random
Random.seed!(123)

# Generate data
n = 200
df = DataFrame(
    education = categorical(rand(["hs", "college", "grad"], n)),
    treatment = rand(Bool, n),
    age = rand(25:65, n),
    y = randn(n) .+ 0.5 .* df.treatment  # True treatment effect
)

# Fit model
model = lm(@formula(y ~ education + treatment + age), df)

# Step 1: Get predictions with education as population-weighted context
ref_grid = DataFrame(
    treatment = [false, true],
    age = [mean(df.age), mean(df.age)]
)

result = profile_margins(model, df, ref_grid; type=:predictions)
result_df = DataFrame(result; include_gradients=true)

# Step 2: Compute treatment contrast
treatment_effect = contrast(result_df, 2, 1, vcov(model))

println("Treatment Effect (at mean age, population-weighted education):")
println("  Estimate: ", round(treatment_effect.contrast, digits=3))
println("  SE: ", round(treatment_effect.se, digits=3))
println("  95% CI: [", round(treatment_effect.ci_lower, digits=3),
        ", ", round(treatment_effect.ci_upper, digits=3), "]")
```

## Alternative Approaches

### Option 1: Population Average Contrasts (AME)

If you want contrasts averaged over the entire observed sample distribution:

```julia
# Average marginal effects with contrasts
result = population_margins(model, df; type=:effects, contrasts=:pairwise)
```

This computes contrasts averaged over **all observations** in your data, not at specific profile points.

### Option 2: Discrete Levels in Reference Grid

If you want automatic pairwise contrasts at specific profiles without mixtures:

```julia
# Specify discrete levels explicitly
ref_grid = DataFrame(
    treatment = [false, false, true, true],
    education = categorical(["hs", "college", "hs", "college"]),
    age = [mean(df.age), mean(df.age), mean(df.age), mean(df.age)]
)

result = profile_margins(model, df, ref_grid; type=:effects, contrasts=:pairwise)
```

This computes contrasts for categorical variables at each discrete profile point.

### Option 3: Quantile or Balanced Grids

For systematic exploration of categorical × continuous interactions:

```julia
# Balanced grid with all categorical levels
ref_grid = balanced_grid(df; education=:all, age=[25, 45, 65])
result = profile_margins(model, df, ref_grid; type=:predictions)

# Then manually contrast specific rows of interest
```

## When to Use Each Approach

| Goal | Recommended Method |
|------|-------------------|
| Contrasts at specific profiles with mixture context | APM + `contrast()` |
| Population-average contrasts (AME) | `population_margins()` with `type=:effects` |
| Automatic pairwise contrasts at discrete profiles | `profile_margins()` with discrete levels, `contrasts=:pairwise` |
| Complex custom contrasts | APM/MEM + `contrast()` for full control |

## Design Rationale

This design follows statistical correctness principles:

1. **Statistical Validity is Paramount**: Prevents meaningless "mixture vs mixture" contrasts
2. **Error-First Policy**: Skips invalid operations rather than approximating
3. **Transparency**: Clear info messages guide users toward valid workflows
4. **Publication-Grade Standards**: Manual `contrast()` uses proper delta-method with full covariance matrix

The APM + manual `contrast()` workflow provides:
- ✅ Full statistical validity (proper delta-method standard errors)
- ✅ Maximum flexibility (contrast any rows you want)
- ✅ Clear interpretability (contrasts between specific profile points)
- ✅ Population-weighted context (mixtures provide realistic backdrop)

## See Also

- [API Reference](../api.md) - See `contrast()` function documentation
- [Reference Grid Guide](../reference_grids.md)
- [Profile Margins](../profile_margins.md)
