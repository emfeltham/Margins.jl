# Population Scenarios (Stata `at()`)

*Counterfactual analysis for population-averaged effects and predictions*

## Overview

Population scenarios enable **"what if"** analysis by computing marginal effects or predictions under counterfactual covariate values. This is analogous to Stata's `at()` option but with population averaging semantics.

**Key Concept:** Scenarios modify variable values for the *entire population* while computing population-averaged quantities (AME or AAP), allowing you to answer questions like:
- "What would the average treatment effect be if everyone were college-educated?"
- "How do predicted outcomes differ between policy scenarios?"
- "What's the effect of X when intervention Y is set to specific levels?"

### Scope and Design

- **Supported in:** `population_margins()` only (population-averaged analysis)
- **Not supported in:** `profile_margins()` (use reference grids like `cartesian_grid()` instead)
- **Design priorities:** Statistical correctness (proper delta-method SEs with full covariance matrix Σ) and zero-allocation performance

## Conceptual Model

Given a fitted model and a dataset, a scenario specifies a set of variable overrides to evaluate counterfactuals. For population analysis, we:
- Keep the original rows and any grouping subset (if `groups` are used).
- Evaluate effects or predictions at the counterfactual covariates by overriding row values during evaluation (no data mutation).
- Average over the selected rows (weighted or unweighted), compute the averaged parameter gradient, and apply the delta method with the full covariance matrix.

## Derived Columns: the Formula Defines What Changes

Margins changes a variable — a `scenarios` value, a reference-grid value, the modifier of a
second difference, or a focal variable whose derivative or contrast is taken — by **re-evaluating
the model's formula** on the changed data. The formula is the model's dependence structure:

- Anything the formula computes from the variable follows the change. With
  `y ~ age + age^2`, setting `age = 40` also sets the squared term to 1600, and the AME of `age`
  includes the curvature.
- Anything precomputed **outside** the formula is, to the model, a separate regressor, and is
  held fixed. With `y ~ age + age2` where `age2 = age.^2` is a data column, setting `age = 40`
  leaves `age2` at each row's observed value, and the AME of `age` is ∂ŷ/∂age *holding `age2`
  fixed* — a different (and rarely meaningful) question. In a reference grid such a column is
  held at its typical value (`means_grid` gives `mean(age2)`, not `mean(age)^2`).

So write transforms in the formula, not as precomputed columns. This is the same advice Stata
gives (`c.age##c.age` rather than a generated `age2`).

**Transforms a formula can use** (FormulaCompiler compiles these; any other function in a formula
is an `ArgumentError` naming this list):

| Kind | Supported |
|:-----|:----------|
| Functions of one argument | `exp`, `log`, `log1p`, `sqrt`, `abs`, `sin`, `cos`, `inv` |
| Arithmetic | `+`, `-`, `*`, `/`, `^` (e.g. `x^2`, `x^3`, `log(x) / log(10)`), unary `-` |
| Comparisons | `<`, `<=`, `>`, `>=`, `==`, `!=` (e.g. `(age >= 65)`), and `!` |
| Interactions | `x & z`, `x * z` |
| Standardization | StandardizedPredictors.jl terms (`ZScore`, `Center`, `Scale` contrasts) |

There are no polynomial or spline bases (`poly`, `bs`, `ns`). If you precompute a spline or
polynomial basis as data columns, every counterfactual on the underlying variable **holds the
basis fixed**: the variable itself changes only where it enters the formula directly. To let the
basis follow, write the polynomial terms in the formula (`x + x^2 + x^3`), or compute
predictions on data you have transformed yourself.

**The derived-column warning.** `population_margins`, `profile_margins` and the model-level
`second_differences` check, once per call, every numeric column the formula uses (never the
response) against each variable the call changes (focal variables, `scenarios` keys, grid
columns, the second-difference modifier and `profile` variables). They warn, in one message
naming each pair, when the column is in the data an exact function (up to floating-point
rounding) of a changed variable `v`:

- a common transform of `v`: `v^2`, `v^3`, `sqrt(v)`, `log(v)`, `log10(v)`, `log(v + 1)`,
  `exp(v)`, `1/v`, or a linear rescaling `a*v + b` (a standardized copy);
- a product `v * u` with another numeric column `u` of the model (a precomputed interaction).

A column the call sets itself, consistently with the variable at every scenario or grid row
(`scenarios=(age=40, age2=1600)`), is not reported.

The check is deliberately narrow: it reports only columns that are almost certainly transforms.
It does not look for columns that are merely constant within each value of `v`, such as an age
band computed from age, or a lookup table keyed on a discrete variable, because that pattern is
also what a legitimate group-level covariate looks like (a year's GDP growth beside a numeric
`year`), and whether such a column should move with `v` is a modelling question. Such columns
are held fixed without a warning; the rule above is what governs them. It is a warning, not an
error, for the same reason: pass `warn=false` for a call where holding the column fixed is
intended.

## Basic Usage

### Single-Variable Scenarios

Evaluate effects or predictions at different values of one variable:

```julia
# Compare predictions under treatment vs control
result = population_margins(model, data;
    type=:predictions,
    scenarios=(treatment=[0, 1],)
)
# Result: 2 rows showing AAP when treatment=0 and treatment=1

# Effect of X under different policy environments
result = population_margins(model, data;
    type=:effects,
    vars=[:x],
    scenarios=(policy=["baseline", "reform"],)
)
# Result: 2 rows showing AME of x under each policy scenario
```

### Multi-Variable Scenarios (Cartesian Product)

Multiple scenario variables create a Cartesian product of all combinations:

```julia
# 2×3 = 6 scenarios
result = population_margins(model, data;
    type=:predictions,
    scenarios=(
        treatment=[0, 1],
        policy=["low", "medium", "high"]
    )
)
# Result: 6 rows for all (treatment, policy) combinations, in the order
# (0,"low"), (0,"medium"), (0,"high"), (1,"low"), …: the first scenario variable varies
# slowest, and each variable's values are in the order given

# 2×2×3 = 12 scenarios
result = population_margins(model, data;
    type=:effects,
    vars=[:income],
    scenarios=(
        education=["HS", "College"],
        region=["Urban", "Rural"],
        tax_rate=[0.15, 0.25, 0.35]
    )
)
# Result: 12 rows showing income effect under all scenario combinations
```

### Scenarios with Grouping

Combine scenarios with grouping for within-group counterfactual analysis:

```julia
# Effects of x within gender groups, under different policy scenarios
result = population_margins(model, data;
    type=:effects,
    vars=[:x],
    groups=:gender,
    scenarios=(policy=["none", "pilot", "full"],)
)
# Result: 6 rows (2 genders × 3 policies)

# Complex: education×region groups × treatment scenarios
result = population_margins(model, data;
    type=:predictions,
    groups=[:education, :region],
    scenarios=(treatment=[0, 1], dosage=[1, 2, 3])
)
# Result: (# education levels × # regions × 2 treatments × 3 dosages) rows
```

## Application examples

### Policy Impact Analysis

**Question:** How do predicted outcomes change under different policy interventions?

```julia
# Healthcare: Compare predicted health outcomes under coverage scenarios
using GLM, DataFrames, Margins

model = glm(@formula(health_score ~ age + income + insurance + education),
            health_data, Normal(), IdentityLink())

policy_comparison = population_margins(model, health_data;
    type=:predictions,
    scenarios=(insurance=["none", "basic", "comprehensive"],)
)

df = DataFrame(policy_comparison)
# Shows average predicted health score under each insurance scenario
```

**Result Interpretation:**
- Each row shows population-averaged predicted outcome under a specific policy
- Standard errors account for uncertainty in model parameters
- Can compare scenarios: `df[df.at_insurance .== "comprehensive", :estimate] - df[df.at_insurance .== "none", :estimate]`

### Treatment Effect Heterogeneity

**Question:** Does the treatment effect vary across subpopulations?

```julia
# Education program: Effect of tutoring hours across SES groups
model = lm(@formula(test_score ~ tutoring_hours + ses + prior_score), student_data)

heterogeneous_effects = population_margins(model, student_data;
    type=:effects,
    vars=[:tutoring_hours],
    groups=:ses,
    scenarios=(prior_score=[40, 50, 60, 70, 80],)  # Standardize baseline
)

df = DataFrame(heterogeneous_effects)
# Shows tutoring effect within each SES group, holding prior_score constant
```

### Intervention Dosage Analysis

**Question:** What's the optimal intervention level?

```julia
# Medication study: Predicted outcomes at different dosages
dosage_response = population_margins(medication_model, patient_data;
    type=:predictions,
    scenarios=(dosage=[0, 5, 10, 15, 20, 25],)  # mg
)

df = DataFrame(dosage_response)
# Plot estimate vs at_dosage to visualize dose-response curve
```

### Demographic Standardization

**Question:** What would effects be if population demographics were different?

```julia
# Labor economics: Income effect standardized to college-educated population
college_standardized = population_margins(wage_model, worker_data;
    type=:effects,
    vars=[:experience],
    scenarios=(education=["College"],)  # Everyone has college degree
)

# Compare to actual population (mixed education)
actual_population = population_margins(wage_model, worker_data;
    type=:effects,
    vars=[:experience]
)
```

### Multi-Dimensional Policy Space

**Question:** How do multiple policies interact?

```julia
# Tax policy: Joint effects of rates and deductions
tax_scenarios = population_margins(income_model, taxpayer_data;
    type=:predictions,
    scenarios=(
        tax_rate=[0.15, 0.22, 0.30],
        deduction_cap=[5000, 10000, 25000],
        credit_phase_out=[40000, 60000, 80000]
    )
)
# Result: 3×3×3 = 27 scenarios showing all policy combinations
```

## Common Patterns and Idioms

### Scenario Naming Convention

Scenarios appear in results with `at_` prefix:

```julia
result = population_margins(model, data;
    type=:predictions,
    scenarios=(treatment=[0, 1], dosage=[10, 20])
)

df = DataFrame(result)
names(df)  # Includes: :at_treatment, :at_dosage, :estimate, :se, ...
```

### Extracting Specific Scenarios

```julia
df = DataFrame(result)

# Filter to treatment=1 scenarios only
treated = df[df.at_treatment .== 1, :]

# Compare two specific scenarios
baseline = df[(df.at_treatment .== 0) .& (df.at_dosage .== 10), :]
intervention = df[(df.at_treatment .== 1) .& (df.at_dosage .== 20), :]
difference = intervention.estimate .- baseline.estimate
```

### Scenario Differences (Contrasts)

To compute differences between scenarios, use predictions:

```julia
# Wrong: Don't try to compute "effect of treatment" using effects + scenarios
# population_margins(model, data; type=:effects, vars=[:treatment], scenarios=(...))
# → ArgumentError (see the conflict rule in grouping.md)

# Right: Use predictions with treatment scenarios
predictions = population_margins(model, data;
    type=:predictions,
    scenarios=(treatment=[0, 1],)
)

df = DataFrame(predictions)
ate = df[df.at_treatment .== 1, :estimate][1] - df[df.at_treatment .== 0, :estimate][1]
# Average Treatment Effect (ATE) = difference in predicted outcomes
```

### Combining with Weights

Scenarios use the averaging weights (the model's by default, or `weights=`); the standard errors still come from `vcov` (see [Weights](weights.md)):

```julia
# Weighted scenarios (e.g., survey data)
result = population_margins(model, survey_data;
    type=:predictions,
    scenarios=(income=[30000, 50000, 70000],),
    weights=:survey_weight
)
# Predictions are population-weighted averages
```

## Architecture Overview

Every counterfactual in Margins goes through one internal mechanism: a spec mapping variables to
`SetTo(value)` (every row gets the value) or `Shift(δ)` (each row's own value moves by δ; used by
the second differences' `at=:observed`), applied by `Margins._apply_counterfactuals`.

- The override is column-level: a changed variable becomes one new column, every other column is
  shared with the data, and the data itself is never mutated.
- A `SetTo` on a categorical column reuses its pool (a level or its string is accepted); a `Bool`
  column stays `Bool`.
- The compiled formula is evaluated on the changed columns row by row, with the same
  zero-allocation kernels as the observed data, so everything the formula computes from a changed
  variable follows it (see *Derived Columns* above).
- Reference-grid completion uses the same `SetTo` for the model variables a grid does not give.

## Computation Details

### Continuous Effects under Scenarios
- For each row i in the context and variable x, the derivative ∂ŷᵢ/∂x is taken at that row's
  values with the scenario's overrides applied (automatic differentiation, or finite differences
  with `backend=:fd`).
- Average per-row effects and per-row gradients across the context (weighted or unweighted) and apply the delta method with the averaged gradient: `se = sqrt(ḡ' Σ ḡ)`.

### Categorical/Boolean Effects under Scenarios
- Build contrasts (baseline or pairwise); at each row, with the scenario's overrides applied, compute the difference between the two levels and its gradient, then average as above and apply the delta method.

### Predictions under Scenarios
- Evaluate predictions and gradients per row on the overridden data, average (weighted/unweighted), and apply the delta method with the averaged gradient.

## Grouping (groups = ...)

- Group subsets are determined first (categorical crosses, quantile bins, thresholds).
- Scenario evaluation occurs within each subset using the same overrides.
- Large combination protection prevents explosion: more than 250 scenario × group combinations raises an error (error-first policy), since each combination is a full pass over the data.

## Weights

- Weighted contexts use proper normalization by total weight: `Σw` is used for both effects and averaged gradients.
- Sampling and frequency weights are supported; weights can be provided as a column `Symbol` or a vector. By default (`weights = :model`) a model fit with weights is averaged with those weights; `weights = :none` gives unweighted averages (see [Weights](weights.md)).

## Column Naming and Ordering

- Group variables appear with an `over_` prefix (e.g., `over_education`; since 3.0, previously unprefixed).
- Scenario variables appear with `at_` prefix (e.g., `at_x`).
- Column order: context columns first (groups, then scenarios), then statistical columns.

## Programmatic Identification

```julia
groups, scenarios = Margins.context_columns(result)
# groups == [:education, ...], scenarios == [:x, :policy, ...]   (raw variable names;
# the DataFrame columns are over_education, at_x, at_policy, ...)
```

## Best Practices

### When to Use Scenarios

**Use scenarios when you want:**
- Population-averaged counterfactuals ("What if everyone had X=value?")
- Policy impact assessment with population averaging
- Standardization to common covariate values
- Treatment effect estimation (via prediction differences)

**Don't use scenarios when you want:**
- Effects at specific covariate points → Use `profile_margins()` with `cartesian_grid()`
- Representative individual analysis → Use `profile_margins()` with `means_grid()`
- Detailed covariate combinations → Use `profile_margins()` with reference grids

### Avoiding Common Mistakes

#### Mistake 1: Confusing Scenarios with Profile Analysis

```julia
# Wrong: Using scenarios for "effects at x=10"
# This computes population-average effect when everyone has x=10 (not what you want)
population_margins(model, data; type=:effects, vars=[:z], scenarios=(x=[10],))

# Right: Use profile analysis for "effects at x=10"
profile_margins(model, data, cartesian_grid(x=[10]); type=:effects, vars=[:z])
```

#### Mistake 2: Conflict Rule Violation

```julia
# Wrong: x appears in both vars and scenarios
# Result: ArgumentError - the whole call is rejected, not just x
population_margins(model, data; type=:effects, vars=[:x, :z], scenarios=(x=[1, 2],))

# Right: Use predictions to see how outcomes change as x varies
population_margins(model, data; type=:predictions, scenarios=(x=[1, 2],))
```

#### Mistake 3: Too Many Scenario Combinations

```julia
# Dangerous: 10×10×10×10 = 10,000 scenarios
# result = population_margins(model, data;
#     type=:predictions,
#     scenarios=(a=1:10, b=1:10, c=1:10, d=1:10)
# )

# Better: Focus on key scenarios
result = population_margins(model, data;
    type=:predictions,
    scenarios=(a=[1, 5, 10], b=[1, 5, 10])  # 3×3 = 9 scenarios
)
```

### Performance Considerations

**Scenario Count:**
- Each scenario requires a full population pass
- 100 scenarios on 10,000 observations = 1 million evaluations
- Keep scenario counts reasonable (< 100 for most applications)

**Grouping Interaction:**
- Scenarios are evaluated within each group
- 10 groups × 20 scenarios = 200 computations
- Computational cost is O(groups × scenarios × observations)

**Memory:**
- Zero per-row allocations (efficient)
- Each scenario stores one new column of n values per variable it overrides; result storage grows with the number of scenarios
- Large scenario counts use O(scenarios) memory for result storage

## Technical Implementation Notes

### Statistical Correctness

All scenarios maintain **publication-grade statistical validity**:
- Full delta-method standard errors using complete covariance matrix Σ
- Proper gradient averaging across population
- No independence assumptions
- Accounts for parameter uncertainty

### Computational Architecture

**Zero-Allocation Design:**
- Scenarios override whole columns once per context (`Margins._apply_counterfactuals`); the per-row kernels read them like the data
- Pre-allocated buffers reused across all scenario evaluations
- No data mutation (original data remains unchanged)
- Categorical types safe (no re-pooling required)

**Performance Characteristics:**
- One new column of n values per overridden variable per scenario; buffers are reused
- O(n) time per scenario where n = observations
- Scenarios evaluated sequentially within a call (separate Margins calls may run concurrently; see the caching documentation)
- FD backend: minimal override sets constructed
- AD backend: exact derivatives with no per-row allocation

### Column Naming Convention

Results use consistent naming:
- **Group variables:** `over_` prefix (e.g., `over_education`, `over_region`)
- **Scenario variables:** `at_` prefix (e.g., `at_treatment`, `at_dosage`)
- **Statistical columns:** `estimate`, `se`, `statistic`, `df`, `p_value`, `ci_lower`, `ci_upper`, `n` (see [Output Columns and Inference](api.md#Output-Columns-and-Inference))

**Programmatic Access:**
```julia
# Identify which columns are groups vs scenarios
groups, scenarios = Margins.context_columns(result)
# groups == [:education, ...], scenarios == [:treatment, :dosage, ...]   (raw variable names;
# the DataFrame columns are over_education, at_treatment, at_dosage, ...)
```

## Relationship to Profile Analysis

**Key Distinction:**
- **Population scenarios (`population_margins` + `scenarios`):** Population-averaged quantities under counterfactual values
- **Profile analysis (`profile_margins` + reference grids):** Quantities at specific covariate combinations

**When to use each:**

| Goal | Method |
|------|--------|
| "Average effect if everyone had X=10" | `population_margins` with `scenarios=(X=[10],)` |
| "Effect at X=10 for a typical individual" | `profile_margins` with `cartesian_grid(X=[10])` |
| "Compare predictions across 3 policies" | `population_margins` with `scenarios=(policy=[...],)` |
| "Effects at high/med/low values of X" | `profile_margins` with `cartesian_grid(X=[...])` |

**Remember:** `profile_margins()` does **not accept** `scenarios` parameter. Use reference grid builders (`cartesian_grid`, `means_grid`, etc.) instead.

## Further Reading

- [Grouping Framework](grouping.md) - Combining scenarios with groups
- [Reference Grids](reference_grids.md) - Profile analysis alternative
- [API Reference](api.md) - Complete parameter documentation

---

*Population scenarios enable sophisticated counterfactual analysis while maintaining computational efficiency and statistical rigor. They are a powerful tool for policy evaluation, treatment effect estimation, and "what if" analysis in econometric research.*
