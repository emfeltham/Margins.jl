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

## Basic Usage

### Single-Variable Scenarios

Evaluate effects or predictions at different values of one variable:

```julia
# Compare predictions under treatment vs control
result = population_margins(model, data;
    type=:predictions,
    scenarios=(treatment=[0, 1])
)
# Result: 2 rows showing AAP when treatment=0 and treatment=1

# Effect of X under different policy environments
result = population_margins(model, data;
    type=:effects,
    vars=[:x],
    scenarios=(policy=["baseline", "reform"])
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
# Result: 6 rows for all (treatment, policy) combinations

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
    scenarios=(policy=["none", "pilot", "full"])
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
    scenarios=(insurance=["none", "basic", "comprehensive"])
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
    scenarios=(prior_score=[40, 50, 60, 70, 80])  # Standardize baseline
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
    scenarios=(dosage=[0, 5, 10, 15, 20, 25])  # mg
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
    scenarios=(education=["College"])  # Everyone has college degree
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
# → treatment will be skipped (see skip rule in grouping.md)

# Right: Use predictions with treatment scenarios
predictions = population_margins(model, data;
    type=:predictions,
    scenarios=(treatment=[0, 1])
)

df = DataFrame(predictions)
ate = df[df.at_treatment .== 1, :estimate][1] - df[df.at_treatment .== 0, :estimate][1]
# Average Treatment Effect (ATE) = difference in predicted outcomes
```

### Combining with Weights

Scenarios respect sampling weights:

```julia
# Weighted scenarios (e.g., survey data)
result = population_margins(model, survey_data;
    type=:predictions,
    scenarios=(income=[30000, 50000, 70000]),
    weights=:survey_weight
)
# Predictions are population-weighted averages
```

## Architecture Overview

Scenario handling is built around FormulaCompiler’s DataScenario system:
- DataScenario: a lightweight structure that maps variable overrides (e.g., `:z => 0.5`) and supplies them to the compiled evaluator per-row.
- Core evaluation calls (internal):
  - `_predict_with_scenario(compiled, scenario, row, scale, β, link, row_buf)`
  - `_gradient_with_scenario!(out, compiled, scenario, row, scale, β, link, row_buf)`

Key properties:
- Zero per-row allocations (reuse pre-allocated row and gradient buffers).
- O(1) memory per context (reuse scenarios; continuous FD constructs only minimal override sets).
- No mutation of the data; categorical types remain safe (no re-pooling required).

## Computation Details

### Continuous Effects under Scenarios
- For each row i in the context and variable x, FD constructs centered differences around `x_i` while merging user overrides for other variables.
- Average per-row effects and per-row gradients across the context (weighted or unweighted) and apply the delta method with the averaged gradient: `se = sqrt(ḡ' Σ ḡ)`.

### Categorical/Boolean Effects under Scenarios
- Build contrasts (baseline or pairwise), merge overrides for non-effect variables into each level scenario, compute per-row differences and gradients, then average as above and apply the delta method.

### Predictions under Scenarios
- Construct a single DataScenario per context, evaluate predictions and gradients per-row, average (weighted/unweighted), and apply the delta method with the averaged gradient.

## Grouping (groups = ...)

- Group subsets are determined first (categorical crosses, quantile bins, thresholds).
- Scenario evaluation occurs within each subset using the same overrides.
- Large combination protection prevents explosion; invalid combinations error (error-first policy).

## Weights

- Weighted contexts use proper normalization by total weight: `Σw` is used for both effects and averaged gradients.
- Sampling and frequency weights are supported; weights can be provided as a column `Symbol` or a vector.

## Column Naming and Ordering

- Group variables appear unprefixed (e.g., `education`).
- Scenario variables appear with `at_` prefix (e.g., `at_x`).
- Column order: context columns first (groups, then scenarios), then statistical columns.

## Programmatic Identification

```julia
groups, scenarios = Margins.context_columns(result)
# groups == [:education, ...], scenarios == [:x, :policy, ...]
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
population_margins(model, data; type=:effects, vars=[:z], scenarios=(x=[10]))

# Right: Use profile analysis for "effects at x=10"
profile_margins(model, data, cartesian_grid(x=[10]); type=:effects, vars=[:z])
```

#### Mistake 2: Skip Rule Violation

```julia
# Wrong: x appears in both vars and scenarios
# Result: x will be skipped, only other variables computed
population_margins(model, data; type=:effects, vars=[:x, :z], scenarios=(x=[1, 2]))

# Right: Use predictions to see how outcomes change as x varies
population_margins(model, data; type=:predictions, scenarios=(x=[1, 2]))
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
- Memory scales with number of unique scenarios, not observations
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
- Scenarios use FormulaCompiler's DataScenario system for efficient variable overrides
- Pre-allocated buffers reused across all scenario evaluations
- No data mutation (original data remains unchanged)
- Categorical types safe (no re-pooling required)

**Performance Characteristics:**
- O(1) memory per scenario (reuse buffers)
- O(n) time per scenario where n = observations
- Scenarios evaluated in parallel when possible
- FD backend: minimal override sets constructed
- AD backend: exact derivatives with zero allocation

### Column Naming Convention

Results use consistent naming:
- **Group variables:** Unprefixed (e.g., `education`, `region`)
- **Scenario variables:** `at_` prefix (e.g., `at_treatment`, `at_dosage`)
- **Statistical columns:** `estimate`, `se`, `ci_lower`, `ci_upper`, etc.

**Programmatic Access:**
```julia
# Identify which columns are groups vs scenarios
groups, scenarios = Margins.context_columns(result)
# groups == [:education, ...], scenarios == [:treatment, :dosage, ...]
```

## Relationship to Profile Analysis

**Key Distinction:**
- **Population scenarios (`population_margins` + `scenarios`):** Population-averaged quantities under counterfactual values
- **Profile analysis (`profile_margins` + reference grids):** Quantities at specific covariate combinations

**When to use each:**

| Goal | Method |
|------|--------|
| "Average effect if everyone had X=10" | `population_margins` with `scenarios=(X=[10])` |
| "Effect at X=10 for a typical individual" | `profile_margins` with `cartesian_grid(X=[10])` |
| "Compare predictions across 3 policies" | `population_margins` with `scenarios=(policy=[...])` |
| "Effects at high/med/low values of X" | `profile_margins` with `cartesian_grid(X=[...])` |

**Remember:** `profile_margins()` does **not accept** `scenarios` parameter. Use reference grid builders (`cartesian_grid`, `means_grid`, etc.) instead.

## Further Reading

- [Grouping Framework](grouping.md) - Combining scenarios with groups
- [Reference Grids](reference_grids.md) - Profile analysis alternative
- [API Reference](api.md) - Complete parameter documentation

---

*Population scenarios enable sophisticated counterfactual analysis while maintaining computational efficiency and statistical rigor. They are a powerful tool for policy evaluation, treatment effect estimation, and "what if" analysis in econometric research.*
