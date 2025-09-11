# Population Scenarios (at) — Design and Implementation

This document explains how counterfactual scenarios (Stata `at()`) are implemented for population analysis, why the design uses FormulaCompiler’s DataScenario system, and which Margins functions rely on it.

## Scope and Intent

- Scenarios are supported in `population_margins` only (population-averaged effects/predictions under counterfactual covariates).
- `profile_margins` does not accept scenarios; it uses explicit reference grids (e.g., `means_grid`, `cartesian_grid`).
- The design prioritizes statistical correctness (proper delta-method SEs) and zero-allocation per-row performance.

## Conceptual Model

Given a fitted model and a dataset, a scenario specifies a set of variable overrides to evaluate counterfactuals. For population analysis, we:
- Keep the original rows and any grouping subset (if `groups` are used).
- Evaluate effects or predictions at the counterfactual covariates by overriding row values via DataScenario during evaluation (no data mutation).
- Average over the selected rows (weighted or unweighted), compute the averaged parameter gradient, and apply the delta method with the full covariance matrix.

## Architecture Overview

The scenario engine is built around FormulaCompiler’s DataScenario system:
- DataScenario: a lightweight structure that maps variable overrides (e.g., `:z => 0.5`) and supplies them to the compiled evaluator per-row.
- Core evaluation calls:
  - `_predict_with_scenario(compiled, scenario, row, scale, β, link, row_buf)`
  - `_gradient_with_scenario!(out, compiled, scenario, row, scale, β, link, row_buf)`

Key properties:
- Zero per-row allocations (reuse pre-allocated row buffers and gradient buffers).
- O(1) memory per context (one DataScenario per context; for continuous-effects FD, two per row are constructed but only hold small override dicts).
- No mutation of the data; categorical types remain safe (no re-pooling required).

## What Uses It (Public and Internal)

Public entrypoint
- `population_margins(model, data; type=:effects|:predictions, scenarios=..., groups=..., weights=...)`
  - When `scenarios` is provided, the population path uses the scenario engine for:
    - Continuous effects (type=:effects, continuous vars)
    - Categorical/boolean effects (type=:effects, categorical/boolean vars)
    - Predictions (type=:predictions)

Core internal functions
- `src/population/contexts.jl`
  - `_population_margins_with_contexts(...)`: Orchestrates scenario×group contexts.
  - `_compute_population_effect_in_context(...)`:
    - Continuous: `_compute_continuous_ame_with_scenario(...)`
    - Categorical/boolean: `_compute_categorical_contrasts(..., scenario_overrides, weights)`
  - `_compute_population_prediction_in_context(...)` (scenario-aware, non-mutating).
  - `_create_context_data(...)`: Applies grouping filters only; no longer overwrites columns for scenarios.
- `src/engine/utilities.jl`
  - `_compute_categorical_contrasts` (scenario-aware & weighted) merges overrides into each level scenario and computes properly averaged effects and gradients.
  - Scenario helpers: `_predict_with_scenario`, `_gradient_with_scenario!` (and non-mutating gradient variant used in-place).

## Computation Details

### Continuous Effects under Scenarios
- For each row i in the context and variable x:
  - Create two scenarios merging user overrides with x at `x_i ± h` (centered FD):
    - `s_minus = {..., x => x_i - h}`
    - `s_plus  = {..., x => x_i + h}`
  - Compute per-row effect and gradient by difference quotients:
    - `Δ_i = (p_plus - p_minus) / (2h)`
    - `g_i = (g_plus - g_minus) / (2h)`
- Average over rows:
  - Unweighted: `ame = mean(Δ_i)`, `ḡ = mean(g_i)`
  - Weighted: `ame = (∑ w_i Δ_i) / ∑ w_i`, `ḡ = (∑ w_i g_i) / ∑ w_i`
- SEs via delta method: `se = sqrt(ḡ' Σ ḡ)` using model covariance `Σ`.

### Categorical/Boolean Effects under Scenarios
- Build contrast pairs (e.g., baseline vs non-baseline, or pairwise) and merge user overrides for all non-effect variables into each level scenario.
- For each row i and contrast (level1, level2):
  - `Δ_i = pred(level2) - pred(level1)`
  - `g_i = grad(level2) - grad(level1)`
- Average as above (weighted/unweighted), then apply delta method with averaged gradient.

### Predictions under Scenarios
- Build a single DataScenario per context (merged overrides) and evaluate predictions and gradients per-row.
- Average prediction and the gradient (weighted/unweighted) over the context.
- Apply delta method with the averaged gradient.

## Grouping (groups = ...)

- Group subsets are determined first (e.g., categorical cross products, quantile bins, or thresholds).
- Scenario evaluation occurs within each subset using the same DataScenario overrides.
- Index-based subgroup specs (e.g., around-value slices) are recognized and intersected directly.
- Large combination limits are enforced to avoid explosion:
  - Hard limit ~1000 combinations (error)
  - Soft limit ~250 combinations (explicit error; error-first policy)

## Weights

- All context computations index weights by original row indices, even for grouped contexts.
- Weighted averages of both the effect/prediction and parameter gradient are used before delta-method SEs.
- Zero total weight triggers an explicit error for statistical correctness.

## Performance Characteristics

- Zero per-row allocations; O(1) per context memory footprint.
- No data mutation and no per-row recompilation.
- Continuous effects: two scenarios per row (±h) but only small override dicts are allocated; they do not scale with data size.
- Categorical contrasts reuse buffers for gradients; creation of level scenarios is O(k) or O(k²) for pairwise, each O(1) memory.

## Statistical Guarantees

- Proper delta-method SEs via averaged gradient and full covariance matrix `Σ`.
- No silent fallbacks: empty groups and zero-weight contexts error out.
- Variables cannot be in both `vars` and `scenarios` (explicit teaching error).

## User-Facing Notes

- Scenarios are specified as a `Dict` (single value or array per variable). The engine expands a Cartesian product and computes results per scenario × group combination:

```julia
# Effects at z=0.5
population_margins(
    m, data; type=:effects, vars=[:x], scenarios=Dict(:z => 0.5)
)

# Predictions under an x grid and treated/untreated
population_margins(
    m, data;
    type = :predictions,
    scenarios = Dict(:x => [-1, 0, 1], :treated => [true, false])
)
```
- `profile_margins` remains reference-grid based and does not accept `scenarios`.

## Open Parameters

- FD step size `h` for continuous effects is currently a small fixed scalar; could be exposed or auto-tuned.
- Future: extend scenario-aware pairwise contrasts coverage and expose a user option to choose pairwise vs baseline.

## Files and Functions Reference

- `src/population/contexts.jl`
  - `_population_margins_with_contexts`, `_compute_population_effect_in_context`
  - `_compute_continuous_ame_with_scenario`, `_build_row_scenarios_for_continuous`
  - `_compute_population_prediction_in_context`
- `src/engine/utilities.jl`
  - `_compute_categorical_contrasts` (scenario-aware & weighted), `_predict_with_scenario`, `_gradient_with_scenario!`
- `test/statistical_validation/population_context_bootstrap_validation.jl` (bootstrap SE validation)
- `test/validation/test_population_scenarios_groups.jl` (scenario/group coverage)
