# Stata Migration Guide

*Complete translation reference for economists migrating from Stata's `margins` command*

## Basic Command Translation

### Core Margins Commands

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins` | `population_margins(model, data; type=:predictions)` | Average adjusted predictions |
| `margins, dydx(*)` | `population_margins(model, data; type=:effects)` | Average marginal effects (AME) |
| `margins, atmeans` | `profile_margins(model, data, means_grid(model, data); type=:predictions)` | Prediction at the means (APM) |
| `margins, dydx(*) atmeans` | `profile_margins(model, data, means_grid(model, data); type=:effects)` | Marginal effects at the means (MEM); see the note on factor variables below |
| `margins, noweights` | `population_margins(model, data; type=:predictions, weights=:none)` | Unweighted averages after a weighted fit |
| `margins, dydx(*) atmeans noweights` | `profile_margins(model, data, means_grid(model, data; weights=:none); type=:effects)` | Unweighted means |

**Weights and the estimation sample.** After a weighted fit (`[pw=w]`, `[fw=w]`, `[aw=w]`), Stata averages over the estimation sample `e(sample)` with the estimation weights, and computes `atmeans` and the other `at()` statistics the same way. Margins does too: `population_margins` averages with the model's weights by default (`weights=:model`) over the estimation rows (FixedEffectModels drops singletons and rows with missing values), and `means_grid(model, data)`, `complete_reference_grid(grid, model, data)` and the values `profile_margins` fills in for variables a grid leaves out use the model's weights over the estimation rows. The data-only `means_grid(data)` cannot see the model: it is unweighted and uses every row, so after a weighted or FixedEffectModels fit it does not reproduce `atmeans`. `weights=:none` (or `complete_reference_grid(…; weights=:none)`) is Stata's `noweights`. The standard errors come from `vcov` (frequency-weight based for GLM.jl; pass a design-based estimator as `vcov` for `pweight` semantics); the point estimates do not depend on the weight type.

**Factor variables at the means.** `means_grid(model, data)` puts a categorical or `Bool` variable at its frequency mixture (share), as `atmeans` does for the variables it holds fixed, and `profile_margins` then reports no discrete contrast for it. Stata's `margins, dydx(*) atmeans` with factor variables (`i.g`) reports the discrete change of `g` with the other variables at their means: set `g`'s levels in the grid and let `profile_margins` fill in the rest (`profile_margins(model, data, cartesian_grid(g=levels(data.g)); type=:effects, vars=[:g])`).

### Variable Selection

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins, dydx(x1 x2)` | `population_margins(model, data; type=:effects, vars=[:x1, :x2])` | Specific variables only |
| `margins, dydx(_continuous)` | `population_margins(model, data; type=:effects)` | All continuous variables (automatic) |
| `margins, eyex(x1)` | `population_margins(model, data; type=:effects, vars=[:x1], measure=:elasticity)` | Elasticities |

## Grouping and Stratification

### Basic Grouping

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins education` | `population_margins(model, data; scenarios=(education=levels(data.education),))` | Predictive margins: everyone set to each level in turn, averaged over the sample |
| `margins education, dydx(x)` | `population_margins(model, data; type=:effects, vars=[:x], scenarios=(education=levels(data.education),))` | Effects with everyone at each level |
| `margins, over(education)` | `population_margins(model, data; groups=:education)` | Subgroups: each level's own rows |
| `margins, dydx(*) over(education)` | `population_margins(model, data; type=:effects, groups=:education)` | Effects within subgroups |
| `margins education gender` | two calls, `scenarios=(education=…,)` and `scenarios=(gender=…,)` | One table per factor |
| `margins education#gender` | `population_margins(model, data; scenarios=(education=…, gender=…))` | Every combination |
| `margins, over(education gender)` | `population_margins(model, data; groups=[:education, :gender])` | Crossed subgroups |

A factor in the marginlist (`margins education`) is a counterfactual (Stata sets every observation to each level), which is `scenarios=`; `over()` is a subgroup (each level's own observations), which is `groups=`. They differ whenever the other covariates differ across the levels.

### Nested Analysis

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins, over(region education)` (education within region) | `population_margins(model, data; groups=:region => :education)` | Nested grouping |
| `margins education, over(region)` | `population_margins(model, data; groups=:region, scenarios=(education=levels(data.education),))` | Everyone at each education level, within each region |

## Scenario Analysis (`at()` Specification)

### Basic Scenarios

Stata's `at()` sets the named variables for every observation and leaves the others **as observed** (the default, `asobserved`), then averages over the estimation sample: that is `population_margins(…; scenarios=…)`. Only with `atmeans` are the other variables held at their means, which is `profile_margins` with a grid giving the `at()` values (the other variables are filled in at their model-weighted means over the estimation sample).

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins, at(x=0)` | `population_margins(model, data; type=:predictions, scenarios=(x=[0],))` | Others as observed (Stata's default) |
| `margins, at(x=(0 1 2))` | `population_margins(model, data; type=:predictions, scenarios=(x=[0, 1, 2],))` | Multiple values |
| `margins, at(x=0 y=1)` | `population_margins(model, data; type=:predictions, scenarios=(x=[0], y=[1]))` | Multiple variables |
| `margins, at(x=(0 1 2)) atmeans` | `profile_margins(model, data, cartesian_grid(x=[0, 1, 2]); type=:predictions)` | Others at their (model-weighted) means |
| `margins, at(x=0 y=1) atmeans` | `profile_margins(model, data, cartesian_grid(x=[0], y=[1]); type=:predictions)` | |
| `margins, at((median) x)` | `population_margins(model, data; type=:predictions, scenarios=(x=[means_grid(model, data; typical=median).x[1]],))` | `means_grid(model, data; typical=median)` gives the weighted median over the estimation sample (smallest value with half the weight, as Stata's `_pctile`; on an exact half Margins takes the midpoint) |
| `margins, at((mean) x)` / `at((means) _all)` | `scenarios=(x=[means_grid(model, data).x[1]],)` / `means_grid(model, data)` with `profile_margins` | |
| `margins, at((p25) x)` | `scenarios=(x=quantile_grid(data; x=[0.25], weights=:w).x,)` | Weighted quantiles interpolate (type 7), so they can differ from Stata's weighted percentile; `quantile_grid` uses every row of `data` (for a FixedEffectModels fit, pass `data[model.esample, :]`) |
| `margins, at(x=0) noweights` | `population_margins(…; scenarios=(x=[0],), weights=:none)` | |

### Population-Level Counterfactuals

| Stata Approach | Margins.jl Population Approach | Notes |
|-----------------|-------------------------------|-------|
| `margins, at(treatment=(0 1))` | `population_margins(model, data; scenarios=(treatment=[0, 1],))` | Everyone untreated vs everyone treated |
| `margins, at(policy=(0 1)) over(education)` | `population_margins(model, data; groups=:education, scenarios=(policy=[0, 1],))` | Policy scenarios within education subgroups |

### Profile vs Population Interpretation

```julia
# Stata: margins, dydx(*) at(treatment=(0 1))          (others as observed)
population_results = population_margins(model, data;
    scenarios=(treatment=[0, 1],),
    type=:effects)

# Stata: margins, dydx(*) at(treatment=(0 1)) atmeans  (others at their means)
profile_results = profile_margins(model, data,
    cartesian_grid(treatment=[0, 1]);
    type=:effects)
```

### Conflict Rule: `dydx(x)` with `over(x)` (and scenarios)

`margins, dydx(x) over(xbin)` translates directly: `population_margins` computes the effect of a variable within bins of that same variable, because binning only selects rows and leaves x varying inside each bin. What it rejects is a specification that *pins* x — grouping by a categorical (each subgroup holds one level) or a `scenarios` override (Stata `at()`, which replaces x for every row) — since there is then no variation to differentiate.

Recommended translations:

```julia
# 1) Stata: margins, dydx(x) over(xbin)
# → Direct translation: bin x and differentiate x within each bin
by_xbins = population_margins(model, data;
    type=:effects,
    vars=[:x],
    groups=(:x, 4))          # quartiles of x; x still varies inside each

# Or evaluate the derivative at specific x values instead of averaging over bins
mem_like = profile_margins(model, data,
    cartesian_grid(x=[-2.0, 0.0, 2.0]);
    type=:effects,
    vars=[:x])

# 2) Custom bin definitions: cut x yourself and group by the resulting column.
#    Equivalent to groups=(:x, 4) when the cuts match; use it when you need
#    bin edges that quantiles and thresholds cannot express.
df.x_bin = cut(df.x, 4)
by_custom_bins = population_margins(model, df;
    type=:effects,
    vars=[:x],
    groups=:x_bin)

# 3) Effects of other variables within x strata (population approach)
effects_in_xbins = population_margins(model, data;
    type=:effects,
    vars=[:z, :w],
    groups=(:x, 4))

# 4) Counterfactual predictions as x changes (not effects of x)
preds_under_x = population_margins(model, data;
    type=:predictions,
    scenarios=(x=[-2.0, 0.0, 2.0],))
```

See also: the “Conflict Rule” section in the Population Grouping docs for rationale and guidance.

#### Short example: grouping by `x_bin` to compute `dydx(x)` across strata

```julia
using Random
using DataFrames, CategoricalArrays
using Statistics  # for quantile
using GLM
using Margins

Random.seed!(42)
n = 500
df = DataFrame(
    y = rand(Bool, n),
    x = randn(n),
    z = randn(n)
)

# Fit a simple model
m = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())

# Create quartile bins for x as a separate column "x_bin"
edges = quantile(df.x, 0:0.25:1.0)
labels = ["Q1", "Q2", "Q3", "Q4"]
df.x_bin = cut(df.x, edges; labels=labels, extend=true)

# Now compute population AME of x within x_bin strata (no contradiction)
res = population_margins(m, df;
    type=:effects,
    vars=[:x],
    groups=:x_bin)

DataFrame(res)  # Shows dydx(x) by Q1..Q4
```

## Output Columns and Test Statistics

| Stata `margins` output | `DataFrame(result)` column |
|:-----------------------|:---------------------------|
| `Margin` / `dy/dx` | `estimate` |
| `Std. err.` | `se` |
| `t` or `z` | `statistic` (with `df`: finite for t, `Inf` for z) |
| `P>|t|` / `P>|z|` | `p_value` |
| `[95% conf. interval]` | `ci_lower`, `ci_upper` |
| `over(g)` rows | `over_g` column |
| `at(x=…)` rows | `at_x` (population scenarios) or bare `x` (profile grids) |

As in Stata, t is used after a linear regression and z otherwise: for `lm` (Stata `regress`)
with any covariance, `df = dof_residual(model)` (Stata's `e(df_r)`, kept after `vce(robust)`);
after `reghdfe`/`reg`, FixedEffectModels' own residual df (G − 1 with clustering, as `reghdfe`);
for `glm` with a non-identity link (`logit`, `probit`, `poisson`, …) and mixed models, `df = Inf`.
After `regress, vce(cluster c)` Stata uses `n_clusters − 1`; pass `df = G - 1` to match it.
`DataFrame(result; format=:stata)` uses Stata's labels (`margin, std_err, t, P_t` or
`margin, std_err, z, P_z`). See [Output Columns and Inference](api.md#Output-Columns-and-Inference).

## Fixed Effects Models (reghdfe / ivregress)

### Basic Fixed Effects

| Stata Command | Margins.jl Equivalent | Notes |
|---|---|---|
| `reghdfe y x1 x2, absorb(state year)` | `reg(df, @formula(y ~ x1 + x2 + fe(state) + fe(year)))` | Uses FixedEffectModels.jl |
| `margins, dydx(*)` (after reghdfe) | `population_margins(model, df; type=:effects)` | AME with absorbed FEs |
| `margins, dydx(x1)` (after reghdfe) | `population_margins(model, df; type=:effects, vars=[:x1])` | Single variable |
| `margins, dydx(*) atmeans` | `profile_margins(model, df, means_grid(model, df); type=:effects)` | MEM over `esample`; effects that do not depend on the FEs |

### Predictions with Fixed Effects

Predictions require `save=:fe` when fitting the model:

```julia
model = reg(df, @formula(y ~ x1 + x2 + fe(state) + fe(year)); save=:fe)
```

| Stata Command | Margins.jl Equivalent | Notes |
|---|---|---|
| `margins` (after reghdfe) | `population_margins(model, df; type=:predictions)` | Requires `save=:fe` |
| `margins, at(x1=(0 1 2))` | `population_margins(model, df; type=:predictions, scenarios=(x1=[0, 1, 2],))` | Each row keeps its own FEs |
| (predict at specific FE levels) | `profile_margins(model, df, DataFrame(x1=[0.0], state=["CA"], year=[2020]); type=:predictions)` | Every FE variable in the grid (or `offset=`) |

### Instrumental Variables

| Stata Command | Margins.jl Equivalent | Notes |
|---|---|---|
| `ivregress 2sls y x2 (x1=z)` | `reg(df, @formula(y ~ x2 + (x1 ~ z)))` | IV estimation |
| `margins, dydx(*)` (after ivregress) | `population_margins(model, df; type=:effects)` | Structural coefficients |

See [Fixed Effects Models](fixed_effects.md) for full documentation.

## Combined Grouping and Scenarios

### Complex Analysis Patterns

| Stata Pattern | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins, at(treatment=(0 1)) over(education)` | `population_margins(model, data; groups=:education, scenarios=(treatment=[0, 1],))` | Group × scenario analysis |
| Multiple `margins` commands | Single comprehensive call | More efficient in Julia |

## Advanced Patterns Beyond Stata

Margins.jl extends far beyond Stata's capabilities with features unavailable in Stata:

### Continuous Variable Binning

```stata
* Stata: create the bin variable, then use over() (it is not a model variable)
xtile income_q = income if e(sample), nq(4)
margins, over(income_q)
```

```julia
# Margins.jl: the bins are computed within the call
population_margins(model, data; groups=(:income, 4))  # Automatic Q1-Q4 quartiles
```

### Custom Policy Thresholds

```stata
* Stata approach:
gen income_bracket = .
replace income_bracket = 1 if income < 25000
replace income_bracket = 2 if income >= 25000 & income < 50000
replace income_bracket = 3 if income >= 50000 & income < 75000
replace income_bracket = 4 if income >= 75000
margins, over(income_bracket)
```

```julia
# Julia approach:
population_margins(model, data; groups=(:income, [25000, 50000, 75000]))
```

### Hierarchical Grouping

```stata
* Stata approach: subgroups of region and education
margins, over(region education)
* No native support for bins computed within groups
```

```julia
# Julia approach (native hierarchical support):
population_margins(model, data; groups=:region => :education)
population_margins(model, data; groups=:country => (:region => :education))  # Deep nesting
```

### Multi-Variable Scenarios

```stata
* Stata: at() with several variables crosses their values (4 scenarios)
margins, at(treatment=(0 1) policy=(0 1))
```

```julia
# Margins.jl: scenarios also cross their values (4 scenarios)
population_margins(model, data; scenarios=(treatment=[0, 1], policy=[0, 1]))
```

Note: `scenarios` are population‑level counterfactuals (everyone receives each setting in turn, other variables as observed), which is what Stata's `at()` does by default. For `at(…) atmeans` (other variables at their means) use `profile_margins(model, data, cartesian_grid(…))`, or `means_grid(model, data)` for every variable at its mean.

## Complete Workflow Examples

### Example 1: Education Policy Analysis

#### Stata Workflow
```stata
* Fit model
logit outcome education income female urban policy_treatment

* Basic effects
margins, dydx(*)

* Effects by education (subgroups)
margins, dydx(income) over(education)

* Policy scenarios within education subgroups
margins, dydx(*) at(policy_treatment=(0 1)) over(education)

* Manual difference calculation needed for treatment effects
```

#### Julia Workflow
```julia
# Fit model  
model = glm(@formula(outcome ~ education + income + female + urban + policy_treatment),
            data, Binomial(), LogitLink())

# Basic effects
basic_effects = population_margins(model, data; type=:effects)

# Effects by education
education_effects = population_margins(model, data; 
                                     type=:effects, 
                                     vars=[:income],
                                     groups=:education)

# Policy scenarios (automatic treatment effect calculation)
policy_analysis = population_margins(model, data;
                                   type=:effects,
                                   groups=:education,
                                   scenarios=(policy_treatment=[0, 1],))

# All results readily available as DataFrames
DataFrame(policy_analysis)
```

### Example 2: Complex Demographic Analysis

#### Stata Approach (Cumbersome)
```stata
* Multiple manual commands needed:
margins, dydx(*) over(region education)
margins, dydx(*) over(region gender)
margins, dydx(*) over(region education gender)

* Income quartiles require manual creation:
xtile income_q4 = income, nq(4)
margins, dydx(*) over(region income_q4)

* No native support for hierarchical analysis
```

#### Julia Approach (Comprehensive)
```julia
# Single comprehensive analysis
comprehensive_results = population_margins(model, data;
    type=:effects,
    groups=:region => [:education, :gender, (:income, 4)]
)

# Results: Region × (Education + Gender + Income-Quartiles) automatically computed
# Professional Q1-Q4 labeling included
DataFrame(comprehensive_results)
```

## Workflow Differences

| Task | Stata | Margins.jl |
|------|-------|------------|
| **Quantile or threshold bins** | Create the bin variable (`xtile`, `recode`), then `over()` | `groups=(:x, 4)` or `groups=(:x, [cuts])`, labeled bins computed in the call |
| **Nested subgroups** | `over(a b)` (every a × b cell) | `groups=:a => :b`, deeper nesting `:a => (:b => :c)` |
| **Several breakdowns at once** | One `margins` command per breakdown | `groups=:a => [:b, (:x, 4)]` in one call |
| **Scenarios within subgroups** | `at()` with `over()` | `scenarios=` with `groups=` |

### Several breakdowns in one call

```julia
result = population_margins(model, data;
    groups=:region => [:education, (:income, 4)],
    scenarios=(treatment=[0, 1], policy=["old", "new"])
)
# Two breakdowns within each region: by education, and by income quartile, each under the
# 4 treatment × policy scenarios. In Stata: xtile the income quartile, then one
# margins, over(region education) at(...) and one margins, over(region income_q) at(...).
```

## Migration Best Practices

### Common Translation Patterns

```julia
# Pattern 1: Simple margins → population_margins
# margins → population_margins(model, data; type=:predictions)

# Pattern 2: Effects by groups → groups parameter  
# margins, dydx(*) over(education) → population_margins(model, data; type=:effects, groups=:education)

# Pattern 3: Multiple at() values → scenarios (Stata's default, others as observed)
# margins, at(x=(0 1 2))         → population_margins(model, data; scenarios=(x=[0, 1, 2],))
# margins, at(x=(0 1 2)) atmeans → profile_margins(model, data, cartesian_grid(x=[0, 1, 2]))

# Pattern 4: Complex manual analysis → comprehensive single call
# Multiple Stata commands → single population_margins with groups + scenarios
```
