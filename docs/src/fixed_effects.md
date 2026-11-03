# Fixed Effects Models

*Using Margins.jl with FixedEffectModels.jl*

## Overview

Margins.jl supports [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) (1.13.2 or later, and 2.x) via a package extension. When you load both packages, marginal effects and predictions become available for models with high-dimensional absorbed fixed effects.

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

This depends only on the non-FE coefficients ``\beta``, not on the absorbed ``\alpha``. Similarly, delta-method standard errors use only ``\text{Var}(\hat{\beta})``, which is what `vcov(model)` returns. This means marginal effects are valid without saving the fixed-effect estimates. (The exception is a variable with absorbed slopes, `fe(g)&x`; see [Restrictions](#Restrictions).)

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
grid = means_grid(model, df)   # means over the estimation sample (esample)
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

# Finite differences
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

The predicted value for each observation is ``X_i \hat{\beta} + \sum_k \hat{\alpha}_{k,g_k(i)}``, averaged across the sample. The absorbed fixed effects enter as a per-row offset of the linear predictor, so every path computes with each row's own fixed effects:

```julia
# By group: each group's rows keep their own fixed effects
population_margins(model, df; type=:predictions, groups=:region)

# Counterfactual: x1 set for everyone, fixed effects at each row's observed values
population_margins(model, df; type=:predictions, scenarios=(x1=[0.0, 1.0],))
```

These agree with averaging FixedEffectModels' `predict(model, df)` over the corresponding rows. (Before Margins 3.0 a single sample-average fixed effect was added to every group's prediction.) Standard errors reflect uncertainty in ``\hat{\beta}`` only, not in the FE estimates — this is standard practice.

### Profile Predictions (APM)

At a reference-grid row the prediction needs the fixed effects of that row. Give them by including **every** fixed-effect variable in the grid (the estimate for each level is looked up), or pass the sum of the fixed effects with `offset=` (or an `offset` grid column):

```julia
# Predictions at specific FE levels
grid = DataFrame(x1=[0.0, 0.0], x2=[0.0, 0.0], state=["CA", "NY"], year=[2020, 2020])
result = profile_margins(model, df, grid; type=:predictions)

# Or the fixed effects' sum given directly
grid = cartesian_grid(x1=[0.0, 1.0, 2.0], x2=[0.0])
result = profile_margins(model, df, grid; type=:predictions, offset=0.0)
```

A grid without them is an error. (Before Margins 3.0 the sample-average fixed effect was used for a fixed-effect variable missing from the grid, silently.) For predictions averaged over the sample at given covariate values, use `population_margins` with `scenarios=` instead.

## Estimation Sample

`reg` drops rows with missing values, with non-positive weights and in singleton fixed-effect groups (its `esample`). Margins uses the same rows: `data` must be the table passed to `reg`, unsubset (one row per entry of `esample`), and Margins selects the estimation rows from it; averages, `groups`, weights and the fixed effects all refer to the estimation sample (`metadata[:n_obs]` is `nobs(model)`). Data with any other number of rows is an `ArgumentError`, including a table subset to the estimation sample by hand: a table that merely has `nobs(model)` rows cannot be told apart from a different one, whose rows would be misaligned silently. A weight vector may be given for every row of `data`.

## Inference

The reference distribution is t with `dof_residual(model)` degrees of freedom: FixedEffectModels' own value, net of the absorbed fixed effects and, with a cluster covariance (`Vcov.cluster(:state)`), G − 1, as `coeftable(model)` uses. So the AME of a variable entering additively reproduces `coeftable`'s estimate, standard error and confidence interval. Pass `df=` to use another reference distribution.

Because `dof_residual` is FixedEffectModels' own value, it depends on the installed version. FixedEffectModels 2.2 changed how absorbed degrees of freedom are counted: fixed-effect levels that are redundant across fixed effects, or nested in the cluster variable, are no longer counted. Degrees of freedom, and with them confidence intervals and p-values (and standard errors that use a small-sample correction), can therefore differ slightly between FixedEffectModels versions; from 2.2 on they match R's fixest.

## Weights

A `FixedEffectModel` does not store the weights it was fit with, so the default `weights=:model` cannot read them: it averages without weights and logs a warning once per session. For a weighted `reg(...; weights=:w)` pass `weights=:w` to Margins; for an unweighted fit pass `weights=:none` to silence the warning.

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

### Variables with Absorbed Slopes Cannot Be in `vars`

In a model with a fixed effect interacted with a continuous variable (`fe(state)&x` or `fe(state)*x`), the derivative with respect to `x` includes each state's slope. FixedEffectModels absorbs these slopes, so they are in neither `coef(model)` nor `vcov(model)`, and an effect computed from the coefficients would leave them out. Effects of such a variable are an error, whether it is named in `vars` or auto-detected; effects of the other variables do not involve the slopes and are computed as usual. Predictions for these models are also unavailable (see Predictions).

```julia
model = reg(df, @formula(y ~ z + x&z + fe(state)*x))

# These error (x has absorbed state-specific slopes):
population_margins(model, df; type=:effects, vars=[:x])
population_margins(model, df; type=:effects)        # x is auto-detected; the message lists vars = [:z]

# This works:
population_margins(model, df; type=:effects, vars=[:z])
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
| `margins, at(x1=(0 1 2))` | `population_margins(model, df; type=:predictions, scenarios=(x1=[0, 1, 2],))` |
| `margins, dydx(*) atmeans` | `profile_margins(model, df, means_grid(model, df); type=:effects)` |
| `ivreghdfe y x2 (x1=z), absorb(state)` (SSC) | `reg(df, @formula(y ~ x2 + (x1 ~ z) + fe(state)))` |
