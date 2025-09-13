# Population Scenarios (Stata at()) — Population Analysis

This page explains how counterfactual scenarios (Stata `at()`) are implemented for population analysis, why the design uses FormulaCompiler’s DataScenario system, and which Margins.jl functions rely on it.

## Scope and Intent

- Scenarios are supported in `population_margins` only (population-averaged effects/predictions under counterfactual covariates).
- `profile_margins` does not accept scenarios; it uses explicit reference grids (e.g., `means_grid`, `cartesian_grid`).
- The design prioritizes statistical correctness (proper delta-method SEs with full Σ) and zero-allocation per-row performance.

## Conceptual Model

Given a fitted model and a dataset, a scenario specifies a set of variable overrides to evaluate counterfactuals. For population analysis, we:
- Keep the original rows and any grouping subset (if `groups` are used).
- Evaluate effects or predictions at the counterfactual covariates by overriding row values via DataScenario during evaluation (no data mutation).
- Average over the selected rows (weighted or unweighted), compute the averaged parameter gradient, and apply the delta method with the full covariance matrix.

## API Usage

```julia
# Effects for x under counterfactual z
population_margins(model, data;
    type=:effects,
    vars=[:x],
    scenarios=(z=[0.0, 1.0])
)

# Predictions under multiple scenarios (Cartesian expansion)
population_margins(model, data;
    type=:predictions,
    scenarios=(x=[-1.0, 0.0, 1.0], treated=[0, 1])
)

# Grouped scenarios
grouped = population_margins(model, data;
    type=:effects,
    vars=[:x],
    groups=[:education, :gender],
    scenarios=(policy=["none", "pilot", "full"]))
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

## Notes

- FD step size `h` is a fixed scalar in the FD path; AD users should prefer `backend=:ad` where applicable.
- `profile_margins` uses reference grids — do not pass `scenarios` there; use `cartesian_grid`/`means_grid` instead.

