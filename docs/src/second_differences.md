# Second Differences (Interaction Effects)

*Quantifying how marginal effects vary across moderator levels*

## Quick Start

One function covers every case. Give it the fitted model, the data, the focal variable(s) and the
moderator:

```julia
second_differences(model, data, :age, :treated)             # discrete modifier: treated vs not
second_differences(model, data, [:age, :income], :treated)  # several focal variables
second_differences(model, data, :age, :income)              # continuous modifier: ∂AME/∂income at mean(income)
second_differences(model, data, :age, :income; at=[20_000, 50_000])      # at chosen values
second_differences(model, data, :age, :income; levels=[20_000, 50_000])  # slope between two values
second_differences(model, data, :age, :treated; profile=(female=[0, 1],)) # with everyone's female set to 0, then 1
second_differences(model, data, :age, :treated; vcov=HC1())              # robust covariance
```

The modifier type is detected from the data (Bool, 0/1, two-valued, string or categorical →
discrete; numeric with more than two distinct values → continuous); use
`modifier_type=:categorical` for an integer-coded categorical modifier. Scenarios, AMEs and
delta-method standard errors are handled internally. `vcov` takes the same forms as in
`population_margins` and defaults to `GLM.vcov`.

If you already hold an `EffectsResult` from `population_margins(...; scenarios=..., type=:effects)`,
the same comparison is available on its rows, with `vcov` defaulting to the covariance the result
was computed with:

```julia
ames = population_margins(model, data; scenarios=(treated=[0, 1],), type=:effects)
sd = second_differences(ames, :age, :treated)
```

## Conceptual Foundation

### What Are Second Differences?

Second differences measure whether the marginal effect of one variable differs across levels of another variable. In generalized linear models, where effects are inherently nonlinear, second differences provide the natural way to quantify interaction effects on the predicted outcome scale.

**Definition**: For two variables X (focal variable) and Z (moderator), the second difference is defined as:

```math
\Delta^2 = [E(Y|X=1,Z=1) - E(Y|X=0,Z=1)] - [E(Y|X=1,Z=0) - E(Y|X=0,Z=0)]
```

This represents the difference in marginal effects, a "difference of differences", capturing how the effect of X changes when Z changes.

### Relationship to Interaction Terms

| Model Type | Linear Model | Nonlinear Model (GLM) |
|------------|--------------|----------------------|
| **Simple effect** | Coefficient on X | Marginal effect of X |
| **Interaction** | Coefficient on X × Z | Second difference |
| **Scale** | Coefficient scale = outcome scale | Must compute from predicted probabilities/means |

In linear models, the interaction coefficient directly represents the second difference. In nonlinear models (logit, probit, Poisson), the interaction coefficient does not directly represent the interaction on the predicted scale, necessitating explicit computation of second differences from marginal effects.

### Methodological Context

Second differences extend the conceptual framework of Margins.jl by addressing a fundamental question: **"Does the effect of X depend on Z?"** This complements the two core questions of marginal effects analysis:
1. **Effects**: "How much does Y change when I change X?"
2. **Predictions**: "What value of Y should I expect?"

Second differences answer: **"Does the answer to question 1 change depending on Z?"**

## Computational Framework

### Two Ways to Compare, One Function

`second_differences(model, data, variables, modifier; …)` chooses the computation from the modifier:

#### Discrete comparison (population-based)

For a discrete modifier (Bool, 0/1, two-valued, string or categorical), or a continuous modifier with
explicit `levels`, Margins.jl computes the population AME of the focal variable with every
observation's modifier set to each level, and differences them. This provides:

- **Population representativeness**: effects averaged over the sample distribution
- **Statistical validity**: delta-method standard errors that use the covariance between the AMEs
- **All pairs**: K levels give K(K−1)/2 comparisons; categorical focal variables add one row per contrast

#### Local derivative (finite differences)

For a continuous modifier, `second_differences` computes ∂AME/∂modifier at the `at` points by a
symmetric finite difference, `(AME(at + δ) − AME(at − δ)) / 2δ`, with the other variables in
`profile` held fixed. This provides:

- **A rate**: the change in the marginal effect per unit of the modifier
- **Profile control**: evaluate at chosen values of the modifier and other variables
- **Requirement**: the modifier must be numeric (the focal variable may be continuous or categorical)

### Statistical Inference

Standard errors use the delta method. For a second difference comparing two AMEs with parameter
gradients g₁ and g₂:

```math
\text{SE}(\Delta^2) = \sqrt{(g_2 - g_1)' \Sigma (g_2 - g_1)}
```

where Σ is the covariance matrix of the parameters (the `vcov` you choose). The formula accounts for
the covariance between the two AME estimates and propagates the full parameter uncertainty. For a
continuous modifier the estimate, standard error and gradient are divided by the modifier distance.

The test statistic is `statistic = second_diff / se`. Its reference distribution is recorded in the
`df` column and follows the covariance used, by the same rule as the rest of the package: t with
`df = dof_residual(model)` for a linear model (`lm`, or a Gaussian identity-link `glm`) when the
covariance is the model's own (`GLM.vcov`, or a matrix equal to `vcov(model)`), and z (`df = Inf`) for
every other model and for robust, clustered or custom covariances (see
[Inference](api.md#Output-Columns-and-Inference)).

### Output Schema

Every second-difference function returns the same columns, in this order, whatever the modifier type:

| column | meaning |
|:-------|:--------|
| `variable` | focal variable (`Symbol`) |
| `contrast` | focal contrast (`"dy/dx"` for a continuous variable, `"B - A"` for levels) |
| `modifier` | modifier variable (`Symbol`) |
| `modifier_type` | `:binary`, `:categorical` or `:continuous` (always filled) |
| `modifier_level1`, `modifier_level2` | modifier values compared (numeric levels ascending; categorical levels in level order) |
| `ame_at_level1`, `ame_at_level2` | focal effect at each modifier value |
| `second_diff` | `ame_at_level2 - ame_at_level1`; for `:continuous`, divided by the modifier distance (a slope) |
| `se` | delta-method standard error |
| `statistic` | `second_diff / se` |
| `df` | degrees of freedom of the reference distribution (`Inf` = z) |
| `p_value` | two-sided p-value for H₀: second difference = 0 (`NaN` for a second difference that is zero by construction) |
| `ci_lower`, `ci_upper` | (1 − `alpha`) confidence interval for `second_diff`, from the same t/z reference as the test |
| `significant` | `p_value < alpha` (`p_adjusted < alpha` when `p_adjust` is set) |

A local derivative adds `eval_point` and `delta_used` (with `modifier_type = :continuous`,
`modifier_level1/2 = eval_point ∓ delta_used`, and `second_diff` the derivative ∂AME/∂modifier), and every
`profile` variable adds one column. With `p_adjust` set, a `p_adjusted` column follows `p_value`; `significant` uses it, but `ci_lower`/`ci_upper` stay unadjusted. With `include_gradients=true`, a `gradient`
column (the vector ∂`second_diff`/∂β, for custom variance calculations) follows the schema columns; as in
`DataFrame(result)` it is omitted by default, so the table has only scalar columns and exports cleanly. The column names are also available as `Margins.SECOND_DIFFERENCE_COLUMNS`.

The example outputs on this page are illustrative; the numbers are not from a particular data set.

## Function Reference

### `second_differences(model, data, variables, modifier; …)`

The main entry point. It builds the scenarios, computes the AMEs and applies the delta method.

```julia
second_differences(
    model::RegressionModel, data,
    variables::Union{Symbol, Vector{Symbol}}, modifier::Symbol;
    levels = nothing, at = nothing, modifier_type::Symbol = :auto,
    profile::NamedTuple = NamedTuple(), delta = :auto,
    scale::Symbol = :response, contrasts::Symbol = :baseline, vcov = GLM.vcov,
    alpha::Real = 0.05, p_adjust::Symbol = :none, include_gradients::Bool = false, warn::Bool = true,
)
```

**Arguments**:
- `model`, `data`: the fitted model and the data frame it was fit to
- `variables`: focal variable(s), continuous or categorical; must not include the modifier
- `modifier`: the moderating variable

**Keyword arguments**:
- `levels`: modifier values to compare. Default for a discrete modifier: all its levels (a categorical
  array's level order, except that numeric levels are always ascending; otherwise sorted). For a continuous modifier, `levels` returns slopes
  `(AME₂ − AME₁)/(m₂ − m₁)` for every pair of the given values
- `at`: continuous modifier only; where to take the derivative: `:mean` (default), `:median`, a number,
  a vector of numbers, or a `NamedTuple` containing the modifier plus other variables to hold fixed.
  Cannot be combined with `levels`
- `modifier_type`: `:auto` (default), `:categorical` or `:continuous`. Detected from the data: numeric
  with more than two distinct values → continuous; Bool, 0/1, two-valued, strings, categorical arrays
  → categorical. A numeric modifier with at most 5 distinct integer values is treated as continuous with
  a warning (unless `at` or `levels` is given), because it is often a coded category. Override with
  `modifier_type=:categorical` for an integer-coded categorical modifier
- `profile`: other variables to hold fixed, e.g. `(age=40, region="north")`; vector values give one
  profile point each (Cartesian product, counterfactual rather than subgroup) and appear as extra columns
- `delta`: finite-difference step for a local derivative (`:auto` = 1% of the modifier's SD, raised
  where needed to keep `at ± δ` distinct in floating point); ignored with `levels`
- `scale`: `:response` or `:link`
- `contrasts`: `:baseline` (K−1 contrasts per categorical focal variable) or `:pairwise` (K(K−1)/2)
- `vcov`: covariance estimator, as in `population_margins`: a function of the model (`GLM.vcov`, the
  default), a matrix (any `AbstractMatrix`, e.g. `Symmetric`), or a CovarianceMatrices estimator (`HC1()`)
- `alpha`: significance level (default 0.05). `significant` is `p_value < alpha` and `ci_lower`/`ci_upper`
  are the (1 − `alpha`) interval, so `alpha=0.1` gives 90% intervals
- `p_adjust`: multiple-testing adjustment over **all rows returned**, i.e. every variable, contrast,
  level pair, evaluation point and `profile` stratum: `:none` (default), `:bonferroni`, `:holm`
  (family-wise error rate) or `:bh` (Benjamini–Hochberg false-discovery rate). It adds a `p_adjusted`
  column after `p_value` and bases `significant` on it. Rows that are zero by construction (see
  [When the model cannot support the calculation](#When-the-model-cannot-support-the-calculation)) are not
  tests: they have no p-value and are left out of the family
- `include_gradients`: add a `gradient` column (∂`second_diff`/∂β per row); omitted by default, as in
  `DataFrame(result)`
- `warn`: `true` (default) gives one warning per call about second differences that are zero by
  construction and about modifier or `profile` values outside the observed range; see
  [When the model cannot support the calculation](#When-the-model-cannot-support-the-calculation).
  `warn=false` silences both without changing any number

**Returns**: a `DataFrame` with the [output schema](#Output-Schema). Weights are not supported.

**Cost**: one `population_margins` call per `profile` stratum for a comparison of levels, and one per
evaluation point and stratum (two modifier values each) for a local derivative.

### `second_differences(result, variables, modifier[, vcov]; …)`

The same comparison on the rows of an `EffectsResult` you already hold: from
`population_margins(...; scenarios=(modifier=levels,), type=:effects)` (differences of AMEs) or from
`profile_margins(model, data, cartesian_grid(modifier=levels); type=:effects)` (differences of marginal
effects at the profile; the grid may vary the modifier only). A `PredictionsResult` is rejected with a
message pointing to `type=:effects`.

```julia
second_differences(
    ame_result::EffectsResult,
    variables::Union{Symbol, Vector{Symbol}},
    modifier::Symbol,
    vcov::AbstractMatrix{<:Real} = <the result's own covariance>;
    contrast::String = "dy/dx",
    modifier_type::Symbol = :auto,
    all_contrasts::Bool = true,
    alpha = nothing,
    p_adjust::Symbol = :none,
    include_gradients::Bool = false,
    warn::Bool = true,
)
```

- `vcov` may be omitted: `population_margins` and `profile_margins` results record the covariance they
  were computed with (`metadata[:vcov]`), so the standard errors agree with those of the AMEs. Pass a matrix to use a
  different estimator; `df` then follows that matrix.
- The result must vary over the modifier only: if another scenario variable or a `groups=` variable also
  varies across its rows, the function raises an error instead of pairing rows from different cells.
- `modifier_type`: `:auto` classifies two levels as `:binary` (with a warning if they are numeric and not
  0/1, because the result is then the raw AME difference rather than a slope), three or more numeric
  levels as `:continuous`, otherwise `:categorical`.
- `contrast` / `all_contrasts`: choose one focal contrast instead of all of them.
- `alpha` defaults to the `ci_alpha` the result was computed with; `p_adjust` and `include_gradients` are as above.
- Numeric (and Bool) modifier levels are always compared in ascending order, so
  `modifier_level1 < modifier_level2` whatever the scenario order; categorical levels keep scenario order.

### Migrating from Margins 2.x

Margins 3.0 has one function, `second_differences`, in the two forms above. The functions of 2.x are
removed (calling one raises an `UndefVarError` that names its replacement):

| Removed (2.x) | Use instead |
|:--|:--|
| `second_differences_at(model, data, vars, mod, vcov; at=…, …)` | `second_differences(model, data, vars, mod; at=…, vcov=vcov, …)` |
| `second_differences_pairwise(result, var, mod, vcov; contrast=…)` | `second_differences(result, var, mod; all_contrasts=false, contrast=…)` |
| `second_differences_all_contrasts(result, var, mod, vcov)` | `second_differences(result, var, mod)` |
| `second_differences_table(result, vars, mod, vcov)` | `second_differences(result, vars, mod)` |
| `second_difference(result, var, mod, vcov)` (a `NamedTuple`) | `second_differences(result, var, mod; include_gradients=true)` (one `DataFrame` row) |

Other differences from 2.x: the `vcov` argument is optional for results from `population_margins` and
`profile_margins`; output tables no longer include a `gradient` column unless `include_gradients=true`;
there are new `ci_lower`/`ci_upper` columns; the local derivative is in `second_diff` (it was
`derivative`); and a numeric modifier's levels are always compared in ascending order. See the CHANGELOG
for the full list.

## Usage Patterns

### Basic Workflow

```julia
using Margins, GLM, DataFrames

# Fit a model with the interaction, then ask how the effect of x varies with `treated`
model = lm(@formula(y ~ x * treated + z), data)
sd = second_differences(model, data, :x, :treated)
```

**Output structure**:
```
1×17 DataFrame (selected columns)
 Row │ variable  contrast  modifier  modifier_type  modifier_level1  modifier_level2  second_diff  se       statistic  df       p_value  significant
     │ Symbol    String    Symbol    Symbol         Int64            Int64            Float64      Float64  Float64    Float64  Float64  Bool
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ x         dy/dx     treated   binary                       0                1        0.152   0.0428       3.55    496.0   0.0004  true
```

**Interpretation**: the marginal effect of `x` is 0.152 units larger when `treated=1` than when
`treated=0` (p < 0.001), a significant positive interaction. In a linear model without other
interactions this equals the coefficient on `x & treated`.

### Binary Moderators

A binary moderator gives one second difference per focal variable and contrast.

```julia
# Does the effect of age vary by treatment status?
model = glm(@formula(outcome ~ age * treated + education), data, Binomial(), LogitLink())
sd = second_differences(model, data, :age, :treated)
```

- **Positive**: the effect is stronger when treated
- **Negative**: the effect is weaker when treated
- **Near zero**: the effect is homogeneous across treatment status

### Categorical Moderators

A moderator with K levels gives K(K−1)/2 pairwise comparisons.

```julia
# Does the income effect vary across education levels?
model = lm(@formula(consumption ~ income * education), data)
sd = second_differences(model, data, :income, :education)
```

```
3×17 DataFrame (selected columns)
 Row │ variable  modifier   modifier_type  modifier_level1  modifier_level2  second_diff  se       statistic  df       p_value  significant
     │ Symbol    Symbol     Symbol         String           String           Float64      Float64  Float64    Float64  Float64  Bool
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ income    education  categorical    hs               college                0.152    0.042       3.62    994.0   0.0003  true
   2 │ income    education  categorical    hs               grad                   0.243    0.051       4.76    994.0   <1e-5   true
   3 │ income    education  categorical    college          grad                   0.091    0.048       1.90    994.0   0.058   false
```

The effect of income is 0.152 higher for college than for high-school graduates (p < 0.001); the
college-to-grad increase (0.091) is not statistically significant. For a plain string column the
levels are compared in sorted order; for a `CategoricalArray` they follow its level order. Use
`levels=["hs", "grad"]` to compare a subset.

### Continuous Moderators

For a continuous modifier the default is the local derivative at the mean:

```julia
model = lm(@formula(outcome ~ treatment * age), data)

second_differences(model, data, :treatment, :age)                        # ∂AME/∂age at mean(age)
second_differences(model, data, :treatment, :age; at=[30, 45, 60])       # at three ages
second_differences(model, data, :treatment, :age; levels=[30, 45, 60])   # slopes between the three ages
```

- `second_diff`: the change in the marginal effect per unit increase in the moderator
- With `levels`, the comparison 30→45 gives `(AME₄₅ − AME₃₀) / 15`; with `at`, the value is the derivative at each point
- Positive slopes mean the effect strengthens as the moderator increases

### Multiple Focal Variables

```julia
# Which demographic effects vary most by treatment?
model = lm(@formula(outcome ~ (age + education_yrs + experience) * treated), data)
sd = second_differences(model, data, [:age, :education_yrs, :experience], :treated)
```

```
3×17 DataFrame (selected columns)
 Row │ variable       modifier  modifier_type  second_diff  se       statistic  df       p_value  significant
     │ Symbol         Symbol    Symbol         Float64      Float64  Float64    Float64  Float64  Bool
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ age            treated   binary               0.045    0.018       2.50    995.0   0.012    true
   2 │ education_yrs  treated   binary               0.123    0.031       3.97    995.0   <1e-4    true
   3 │ experience     treated   binary               0.012    0.024       0.50    995.0   0.617    false
```

### Categorical Focal Variables

A categorical focal variable gives one row per contrast and modifier comparison.

```julia
using CategoricalArrays
data.education = categorical(data.education; levels=["hs", "college", "grad"])

model = lm(@formula(income ~ education * region), data)

# Baseline contrasts (college − hs, grad − hs) × region pairs
sd = second_differences(model, data, :education, :region)

# All pairwise contrasts (adds grad − college)
sd = second_differences(model, data, :education, :region; contrasts=:pairwise)
```

With three education levels and three regions, `contrasts=:baseline` (the default) returns 2 × 3 = 6
rows and `contrasts=:pairwise` returns 3 × 3 = 9 rows. To look at one contrast, filter the result:
`sd[sd.contrast .== "college - hs", :]`.

### Local Derivatives at Profiles

```julia
# Does the income effect on consumption strengthen or weaken with age?
model = lm(@formula(consumption ~ income * age + education), data)

second_differences(model, data, :income, :age)                           # at mean age
second_differences(model, data, :income, :age; at=[25, 35, 45, 55, 65])  # 5 rows
second_differences(model, data, :income, :age;
                   at=40, profile=(education="college", region="urban"))
# → "If everyone were a college graduate in an urban region, how does the income effect vary with age?"
```

Local derivatives hold the `profile` variables fixed while the modifier moves by ±δ, so each row is a
statement about that profile. Discrete comparisons accept `profile` too, which runs the comparison once
per profile point. A profile value is a counterfactual (everyone's `female` is set to 0, then to 1), not a
subgroup analysis; in a nonlinear model the two differ. For subgroups, compute the effects with
`population_margins(…; groups=…)` and pass the result to `second_differences`. `profile` must not contain
the modifier.

**Which comparison you get**:

| Modifier | Default | Alternative |
|----------|---------|-------------|
| Binary or categorical | AME difference between levels | `levels=` to select levels |
| Continuous (numeric, >2 distinct values) | local derivative at `at=:mean` | `at=` for other points, `levels=` for slopes between values |
| Integer-coded categories (≤ 5 integer values) | treated as continuous, with a warning | `modifier_type=:categorical` |

**Common error**: a local derivative needs a numeric modifier. Giving `at` with a categorical
modifier raises an error that says so; omit `at` and `second_differences(model, data, …)` compares the
modifier's levels instead.

## Advanced Analysis Patterns

### Robust Standard Errors

Pass the estimator as `vcov`, as in `population_margins`:

```julia
using CovarianceMatrices

sd = second_differences(model, data, :x, :treated; vcov=HC1())
```

For a result you already hold, compute it with the estimator and the second differences follow it:

```julia
ames = population_margins(model, data; scenarios=(treated=[0, 1],), type=:effects, vcov=HC1())
sd = second_differences(ames, :x, :treated)
```

The delta method then uses the robust covariance, and `df` is `Inf` (z inference).

### Multiple Testing Considerations

A categorical moderator with many levels, several focal variables or many evaluation points produce many
tests. Ask for an adjustment over everything returned:

```julia
sd = second_differences(model, data, :x, :education; p_adjust=:holm)
# or :bonferroni, or :bh for false-discovery-rate control

sd[:, [:modifier_level1, :modifier_level2, :p_value, :p_adjusted, :significant]]
```

`p_adjusted` follows `p_value`, and `significant` is `p_adjusted < alpha`. The family is every row of the
returned table: all variables, contrasts, level pairs, evaluation points and `profile` strata. To adjust
over a different family, compute the pieces separately and adjust the `p_value` column yourself.

### Confidence Levels

`alpha` sets both the significance level and the interval: `alpha=0.1` gives 90% intervals in
`ci_lower`/`ci_upper` and `significant = p_value < 0.1`. For a result you already hold, `alpha` defaults to
the `ci_alpha` it was computed with.

### Visualization Workflows

```julia
using StatsPlots

@df sd scatter(:modifier_level2, :second_diff,
               yerr=(:second_diff .- :ci_lower, :ci_upper .- :second_diff),
               xlabel="Modifier Level", ylabel="Second Difference",
               title="Effect Heterogeneity Across Moderator", legend=false)
hline!([0], linestyle=:dash, color=:gray)
```

## Methodological Notes

### Comparison to Coefficient-Based Interactions

In linear models, second differences equal interaction coefficients:
```julia
model_linear = lm(@formula(y ~ x * z), data)          # z coded 0/1
second_differences(model_linear, data, :x, :z).second_diff   # ≈ coef(model_linear)[end]
```

In nonlinear models, they diverge:
```julia
model_logit = glm(@formula(y ~ x * z), data, Binomial(), LogitLink())
second_differences(model_logit, data, :x, :z).second_diff    # ≠ coef(model_logit)[end]
# the second difference is on the probability scale (scale=:response)
```

This divergence motivates the second-differences framework: to express interactions on the
interpretable predicted-outcome scale rather than the coefficient scale.

### Relationship to Ai & Norton (2003)

Ai & Norton (2003) demonstrated that in nonlinear models (logit, probit), the interaction effect:
1. Is not equal to the interaction coefficient
2. Varies across observations
3. Can have different signs than the coefficient

Second differences in Margins.jl operationalize this framework by computing population-averaged
interaction effects (differences of AMEs), providing delta-method standard errors, and enabling
tests of the interaction on the predicted scale.

**Reference**: Ai, C., & Norton, E. C. (2003). Interaction terms in logit and probit models. *Economics Letters*, 80(1), 123-129.

### Population vs Profile Interpretation

Comparisons of levels use **population-averaged** marginal effects: AMEs averaged over the sample
distribution at each modifier level, appropriate for policy analysis requiring external validity.
Local derivatives, and any comparison with a `profile`, are conditional on the profile you specify:
useful for scenario-specific analysis or representative cases.

### Significance Testing

Hypothesis tests evaluate H₀: second difference = 0 (no interaction on the predicted scale).

- A significant second difference means an interaction exists on the predicted outcome scale
- A significant interaction coefficient means an interaction exists on the coefficient scale

In nonlinear models these are distinct hypotheses; second differences test the one most relevant for
applied interpretation.

### When the model cannot support the calculation

A second difference asks whether the effect of the focal variable changes with the modifier, so the
fitted model has to allow that.

**Zero by construction.** In a model with an identity link, the effect of `x` can depend on `m` only
through an interaction term. Without one, the second difference is exactly zero whatever the data say;
it is a property of the specification, not an estimate. The same happens when the modifier is not in the
model at all (in any model, including a logit). `second_differences` detects this from the gradient of the
difference (it is identically zero, so no parameter uncertainty can move it) and then:

- reports `second_diff = 0`, `se = 0`, `significant = false`, and `NaN` for `statistic`, `p_value`,
  `ci_lower` and `ci_upper`, instead of `p = 1` and an interval `[0, 0]`, which would read as a precise
  finding of "no interaction";
- leaves the row out of the `p_adjust` family, since it is not a test;
- issues one warning per call that lists the affected `variable × modifier` pairs and says whether the
  modifier is missing from the model.

A real interaction term is never flagged, even if its estimated coefficient is close to zero: the test is
about the model's structure, not the estimate.

**Nonlinear models without an interaction term** are not flagged. In a logit, for example, the marginal
effect of `x` changes with `m` through the link even with no `x:m` term, which is exactly what second
differences measure on the response scale. Such a row is reported normally.

**Extrapolation.** If numeric `levels`, `at` or `profile` values fall outside the range of that variable in
the data, the counterfactual extrapolates the fitted model, and in a nonlinear model the result can be
badly wrong. `second_differences(model, data, …)` warns and lists the values. (A precomputed result
carries no data, so this check applies to the model-level form only.)

Both warnings are suppressed by `warn=false`. To find the affected rows programmatically, use
`isnan.(sd.p_value)`.

## Integration Examples

### With GLM.jl

```julia
using GLM, CategoricalArrays

model = glm(@formula(employed ~ education * experience + age), data, Binomial(), LogitLink())

# Change in the employment effect of education per year of experience, on the probability scale
sd = second_differences(model, data, :education, :experience)
sd = second_differences(model, data, :education, :experience; levels=[0, 10, 20])
```

### With MixedModels.jl

Second differences work for fitted mixed models through the same entry point:

```julia
using MixedModels

model = fit(MixedModel, @formula(outcome ~ treatment * time + (1 + time | subject)), data)

# Change in the treatment effect per unit of time (a time-varying treatment effect)
sd = second_differences(model, data, :treatment, :time; levels=[0, 6, 12])
```

### With the DataFrames ecosystem

```julia
using DataFrames, CSV

results = second_differences(model, data, :income, :region)
significant = sort(subset(results, :significant => x -> x .== true), :second_diff; rev=true)
CSV.write("significant_interactions.csv", significant)
```

## Error Handling

### Common Error Patterns

**Modifier not found, or also a focal variable**:
```julia
second_differences(model, data, :x, :nope)
# → Error: "Modifier variable :nope not found in data. Available columns: …"
```
The modifier cannot also be in `variables`; `population_margins` rejects a variable that is both
a focal variable and a scenario variable.

**`at` and `levels` together, or `at` with a categorical modifier**:
```julia
second_differences(model, data, :x, :age; at=40, levels=[30, 50])
# → Error: Give either `levels` (compare these modifier values) or `at` (local derivative), not both.
second_differences(model, data, :x, :region; at=1)
# → Error: `at` (local derivative) applies to a continuous modifier; :region is treated as categorical. …
```

**Fewer than two modifier levels**:
```julia
second_differences(model, data, :x, :treated; levels=[1])
# → Error: "Need at least 2 modifier levels to compare, got 1."
```
For a precomputed result with a single scenario level the message is
`Need at least 2 modifier levels, found 1 for variable x with contrast dy/dx`.

**Result varies over more than the modifier** (precomputed results):
```julia
ames = population_margins(model, data; vars=[:x],
                         scenarios=(treated=[0,1], female=[0,1]), type=:effects)
second_differences(ames, :x, :treated)
# → Error: "The effects result varies over scenario variable :female as well as the modifier :treated …"
```
Compute the result over the modifier alone (one call per value of the other variable), or use
`second_differences(model, data, …; profile=(female=[0, 1],))`, which does this for you. The same
applies to results computed with `groups=`.

**Variable not in the result** (precomputed results):
```julia
second_differences(ames, :z, :treated)
# → Error: "Variable :z not found in the effects result (variables: x). Include it in `vars` …"
```

**Wrong covariance dimensions** (precomputed results with an explicit `vcov`):
```julia
second_differences(ames, :x, :treated, vcov(different_model))
# → Error: "Dimension mismatch: vcov has N parameters but gradients has M"
```
Use the covariance of the model the result was computed from, or omit `vcov`.

### Validation Workflow

```julia
using Statistics: mean

function validate_second_differences(sd::DataFrame)
    if any(isnan.(sd.second_diff)) || any(isinf.(sd.second_diff))
        @warn "NaN or Inf values detected in second differences"
    end
    structural = sum(isnan.(sd.p_value))   # zero by construction: the model has no interaction for that row
    structural > 0 && @info "$structural second difference(s) are zero by construction (no test is possible)"
    println("Significant interactions: $(sum(sd.significant)) / $(nrow(sd))")
    println("Mean absolute second difference: $(mean(abs.(sd.second_diff)))")
    return sd
end

sd = validate_second_differences(second_differences(model, data, :x, :treated))
```

## Performance Considerations

The cost is the AME computation. Differencing the AMEs, applying the delta method and forming all
pairwise comparisons from the stored gradients is negligible by comparison.

- **Comparison of levels**: one `population_margins` call (all levels in one scenario grid) per `profile` stratum.
- **Local derivative**: two modifier values, so one `population_margins` call, per evaluation point and stratum.
- **Precomputed results**: `second_differences(result, …)` does no model evaluation and no pass over the data;
  it reuses the gradient matrix stored in the `EffectsResult`, so many comparisons from one result are essentially free.

## Literature and References

### Key Methodological Papers

1. **Ai, C., & Norton, E. C. (2003)**. Interaction terms in logit and probit models. *Economics Letters*, 80(1), 123-129.
   - Seminal paper demonstrating problems with interpreting interaction coefficients in nonlinear models
   - Established need for marginal effects-based interaction analysis

2. **Norton, E. C., Wang, H., & Ai, C. (2004)**. Computing interaction effects and standard errors in logit and probit models. *The Stata Journal*, 4(2), 154-167.
   - Practical implementation guidance
   - Standard error computation for interaction effects

3. **Greene, W. H. (2010)**. Testing hypotheses about interaction terms in nonlinear models. *Economics Letters*, 107(2), 291-296.
   - Hypothesis testing framework for interactions
   - Multiple comparison considerations

4. **Karaca-Mandic, P., Norton, E. C., & Dowd, B. (2012)**. Interaction terms in nonlinear models. *Health Services Research*, 47(1pt1), 255-274.
   - Applied examples in health economics
   - Interpretation guidance for practitioners

### Software Implementation References

- **Stata**: `margins` command with `dydx()` operator and `at()` option
- **R**: `margins` package (Leeper et al.) and `interactionTest` package
- **Python**: `statsmodels.discrete.discrete_model.Logit.get_margeff()` with `at` specification

Margins.jl's second differences implementation follows this established methodological tradition while leveraging Julia's high-performance computational capabilities for efficient large-scale analysis.

---

*This documentation provides comprehensive coverage of second differences functionality in Margins.jl. For the underlying marginal effects framework, see [Mathematical Foundation](mathematical_foundation.md). For population-based analysis that produces the AME inputs, see [Population Scenarios](population_scenarios.md). For integration with robust standard errors, see [Advanced Features](advanced.md).*
