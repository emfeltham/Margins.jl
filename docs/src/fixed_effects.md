# Fixed Effects Models

*Using Margins.jl with FixedEffectModels.jl*

## Overview

Margins.jl supports [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) via a package extension. When you load both packages, marginal effects and predictions become available for models with high-dimensional absorbed fixed effects.

```julia
using Margins, FixedEffectModels, DataFrames

model = reg(df, @formula(y ~ x1 + x2 + fe(state) + fe(year)))

# Average marginal effects
population_margins(model, df; type=:effects)

# Predictions (requires save=:fe at fit time)
model = reg(df, @formula(y ~ x1 + x2 + fe(state) + fe(year)); save=:fe)
population_margins(model, df; type=:predictions)
```

The extension loads automatically when both `Margins` and `FixedEffectModels` are imported. No additional setup is required.

## Marginal Effects

### Why Effects Work Without FE Estimates

For linear models with absorbed fixed effects, the model is:

```math
y_i = X_i \beta + \alpha_{g(i)} + \varepsilon_i
```

where ``\alpha_{g(i)}`` are the absorbed fixed effects. The marginal effect of a continuous variable ``x_k`` is:

```math
\frac{\partial y}{\partial x_k} = \beta_k + \sum_j \beta_{kj} x_j
```

This depends only on the non-FE coefficients ``\beta``, not on the absorbed ``\alpha``. Similarly, delta-method standard errors use only ``\text{Var}(\hat{\beta})``, which is what `vcov(model)` returns. This means marginal effects are valid without saving the fixed-effect estimates.

### Average Marginal Effects (AME)

```julia
model = reg(df, @formula(y ~ x1 + x2 + fe(state)))

# All variables
result = population_margins(model, df; type=:effects)
DataFrame(result)

# Specific variables
result = population_margins(model, df; type=:effects, vars=[:x1])
```

### Marginal Effects at Means (MEM)

```julia
grid = means_grid(df)
result = profile_margins(model, df, grid; type=:effects, vars=[:x1, :x2])
DataFrame(result)
```

### Interaction Models

Marginal effects for interaction models are computed correctly. When the model includes ``x_1 \times x_2``, the marginal effect of ``x_1`` varies with ``x_2``:

```julia
model = reg(df, @formula(y ~ x1 * x2 + fe(state)))

# AME: averaged over the sample distribution of x2
result = population_margins(model, df; type=:effects, vars=[:x1])

# Counterfactual: effect of x1 when x2 is set to specific values
result = population_margins(model, df; type=:effects, vars=[:x1],
                            scenarios=(x2=[0.0, 1.0],))
```

### Backend Selection

Both computation backends are supported:

```julia
# Automatic differentiation (default)
population_margins(model, df; type=:effects, backend=:ad)

# Finite differences (zero allocation)
population_margins(model, df; type=:effects, backend=:fd)
```

## Predictions

Predictions require the absorbed FE estimates, since the full predicted value is ``\hat{y}_i = X_i \hat{\beta} + \hat{\alpha}_{g(i)}``. Pass `save=:fe` when fitting the model:

```julia
model = reg(df, @formula(y ~ x1 + x2 + fe(state) + fe(year)); save=:fe)
```

Without `save=:fe`, prediction requests produce an informative error.

### Population Predictions (AAP)

Average adjusted predictions incorporate FE estimates:

```julia
result = population_margins(model, df; type=:predictions)
DataFrame(result)
```

The predicted value for each observation is ``X_i \hat{\beta} + \sum_k \hat{\alpha}_{k,g_k(i)}``, averaged across the sample. Standard errors reflect uncertainty in ``\hat{\beta}`` only, not in the FE estimates — this is standard practice.

### Profile Predictions (APM)

Profile predictions evaluate the model at specific covariate combinations. Fixed-effect variables can be included in the reference grid to select specific FE levels:

```julia
# Predictions at specific covariate values (FE averaged across sample)
grid = cartesian_grid(x1=[0.0, 1.0, 2.0], x2=[0.0])
result = profile_margins(model, df, grid; type=:predictions)

# Predictions at specific FE levels
grid = DataFrame(x1=[0.0, 0.0], x2=[0.0, 0.0], state=["CA", "NY"])
result = profile_margins(model, df, grid; type=:predictions)
```

When FE variables are **in the reference grid**, the corresponding FE estimate for that level is used. When FE variables are **not in the reference grid**, their population-average FE value is used.

### Multiple Fixed Effects

Models with multiple absorbed FE sets work as expected:

```julia
model = reg(df, @formula(y ~ x1 + x2 + fe(state) + fe(year)); save=:fe)

# Average all FEs
grid = cartesian_grid(x1=[0.0, 1.0])
result = profile_margins(model, df, grid; type=:predictions)

# Specify state, average year
grid = DataFrame(x1=[0.0], x2=[0.0], state=["CA"])
result = profile_margins(model, df, grid; type=:predictions)

# Specify both
grid = DataFrame(x1=[0.0], x2=[0.0], state=["CA"], year=[2020])
result = profile_margins(model, df, grid; type=:predictions)
```

## Instrumental Variables

IV models estimated with the `(endogenous ~ instrument)` syntax are supported. Margins are computed using the structural (second-stage) coefficients and covariance matrix, consistent with Stata's `margins` after `ivregress` and R's `marginaleffects` package.

```julia
model = reg(df, @formula(y ~ x2 + (x1 ~ z) + fe(state)))
result = population_margins(model, df; type=:effects)
```

An informational note is logged once per session when an IV model is detected. Effects for all variables (both endogenous and exogenous) are computed — the package does not distinguish between them, following the convention of existing implementations.

## Restrictions

### Absorbed FE Variables Cannot Be in `vars`

The coefficients of absorbed fixed effects are not part of `coef(model)`, so marginal effects for these variables are not computable:

```julia
# This will error:
population_margins(model, df; type=:effects, vars=[:state])

# This works:
population_margins(model, df; type=:effects, vars=[:x1, :x2])
```

### Counterfactual Scenarios on FE Variables

For predictions, counterfactual scenarios on absorbed FE variables are blocked. Setting everyone to a specific FE group conflates causal effects with unobserved heterogeneity:

```julia
# This will error for type=:predictions:
population_margins(model, df; type=:predictions, scenarios=(state=["CA"],))
```

Use the profile approach instead to examine specific FE levels.

### Standard Error Coverage

Standard errors for predictions reflect uncertainty in ``\hat{\beta}`` only. They do not incorporate estimation uncertainty in the absorbed FE estimates ``\hat{\alpha}``, which are treated as fixed. This is the standard approach in the fixed-effects literature and matches the behavior of Stata and R.

## Stata Translation

| Stata Command | Margins.jl Equivalent |
|---|---|
| `reghdfe y x1 x2, absorb(state year)` | `reg(df, @formula(y ~ x1 + x2 + fe(state) + fe(year)))` |
| `margins, dydx(*)` | `population_margins(model, df; type=:effects)` |
| `margins, dydx(x1)` | `population_margins(model, df; type=:effects, vars=[:x1])` |
| `margins, at(x1=(0 1 2))` | `profile_margins(model, df, cartesian_grid(x1=[0,1,2]); type=:predictions)` |
| `margins, at(means) dydx(*)` | `profile_margins(model, df, means_grid(df); type=:effects)` |
| `ivregress 2sls y x2 (x1=z), absorb(state)` | `reg(df, @formula(y ~ x2 + (x1 ~ z) + fe(state)))` |
