# Population Margins Workflow Assessment (scenario/groups)

This assessment reviews the logic and mathematical/statistical correctness of the `population_margins` workflow with an emphasis on:
- `scenarios` (Stata `at()` analogue)
- `groups` (Stata `over()` analogue with extended syntax)
- Delta-method standard errors under these contexts

It evaluates the actual implementation (not just docs) across `src/population`, `src/engine`, and supporting computation helpers.

## Scope and Entry Points

- Entry: `src/population/core.jl: population_margins(...)`
  - No-context path (baseline AME/AAP): delegates to engine utilities; statistically sound.
  - Context path (with scenarios/groups): `src/population/contexts.jl: _population_margins_with_contexts(...)` handles parsing, data subsetting, and per-context computations.
- Key helpers referenced:
  - `src/population/contexts.jl`: `_parse_at_specification`, `_parse_groups_specification` (+ nested/quantile/threshold helpers), `_create_context_data[_with_weights]`, `_compute_population_effect_in_context`, `_compute_population_prediction_in_context`.
  - `src/engine/utilities.jl`: AME/AAP core computation, categorical contrasts via DataScenario, gradients/SE utilities.
  - `src/computation/statistics.jl`: Delta-method SE utilities.

## Findings

### 1) Scenario handling (counterfactual `at()` semantics)

- Continuous effects ignore scenarios:
  - Code: `_compute_continuous_ame_context(...)` uses `engine.de` and iterates rows via `context_indices`, but does not apply scenario overrides when evaluating derivatives or values. Computation uses original covariate values instead of scenario covariates.
  - Consequence: Effects are not evaluated at the specified counterfactual scenario. This violates scenario semantics and the statistical correctness mandate.
  - Required fix: For `type=:effects` with scenarios, compute per-row effects under a full DataScenario that merges all scenario overrides. For continuous vars, use FD with scenarios per row: evaluate `p(x±h | scenario_except_x)` and gradients `g±`, then use centered-difference for value and gradient, and average (weighted/unweighted).

- Categorical effects ignore scenarios for non-effect covariates:
  - Code: `_compute_categorical_baseline_ame_context(...)` delegates to `_compute_categorical_contrasts(engine, var, context_indices, ...)`, which creates scenarios only for the target categorical var and evaluates on `engine.data_nt`. It ignores user-provided scenario overrides for other covariates.
  - Consequence: Computation averages contrasts at original covariates, not the requested counterfactuals.
  - Required fix: Merge user scenario overrides into each contrast scenario built inside `_compute_categorical_contrasts` so predictions and gradients are calculated under the full scenario.

- Scenario overrides mutate columns unsafely:
  - Code: `_create_context_data` replaces full columns with `fill(val, n_rows)` regardless of type. For `CategoricalArray`, this loses pool/levels and can break design/contrasts.
  - Required fix: Avoid column mutation for scenarios; always use FormulaCompiler DataScenario. If column replacement is retained, coerce overrides to proper `CategoricalValue` with original pool/levels (mirroring `_build_refgrid_data`). DataScenario is the preferred approach.

### 2) Groups (`over`) handling

- Parsing/design:
  - Categorical: `Symbol` or `Vector{Symbol}` → cross-tab (sensible).
  - Continuous: `(var, n_quantiles)` or `(var, thresholds::Vector)` → bins with `(lower, upper, label, bin)`; boundary handling avoids overlap (correct).
  - Nested: `outer => inner` and vectorized inner specs → cross-product of nested strata (reasonable).

- Subsetting implementation:
  - `_create_context_data` applies range-based filtering for continuous bins and equality for categoricals.
  - Bug: “continuous subgroups around specified values” are not integrated:
    - `_create_continuous_subgroups` produces NamedTuples `(center, indices, label)`.
    - `_create_context_data` only recognizes NamedTuples with `:lower`/`:upper`, otherwise treats spec as equality against data values, which cannot match a NamedTuple. This silently yields empty subgroups.
  - Required fix: In `_create_context_data`, if a spec has `:indices`, intersect `indices_to_keep` with that vector.

### 3) Weights in contexts (statistical correctness)

- Predictions (AAP) in contexts misalign weights:
  - Code path: `_create_context_data_with_weights` passes the full weights vector (not subset), then `_compute_population_prediction_in_context` calls `_compute_population_predictions!` which loops `i=1:n_obs` on `context_data` and uses `weights[i]`. Here, `i` is a position in the subset, but `weights[i]` indexes the original full vector by the subset position. This is wrong for most subsets.
  - Required fix: Either pass a subsetted `context_weights` aligned with the context row order, or modify the prediction loop to iterate over `context_indices` and use `weights[row]` while constructing rows accordingly.

- Categorical effects under weights use an invalid scaling workaround:
  - Code: `_compute_categorical_contrasts` computes unweighted sums across rows. In `_compute_categorical_baseline_ame_context`, if `weights` is provided, it scales the (unweighted) effect and gradient by `total_weight/n_obs` as a “temporary compatibility layer”. This is not the correct weighted average or gradient.
  - Correct formula: `ame = (∑ w_i Δ_i) / ∑ w_i`, `ḡ = (∑ w_i g_i) / ∑ w_i`.
  - Required fix: Thread weights into `_compute_categorical_contrasts` and apply weights per row to both contrasts and gradients, averaging properly.

### 4) Delta-method SEs

- Where gradients are correct, SEs are correctly computed as `sqrt(G Σ G')` using full covariance `Σ` via `compute_se_only`. This is sound.
- However, in the above incorrect scenario/weighting paths, the gradients are incorrect; therefore, SEs are incorrect as well. The delta-method routine itself is fine.

### 5) Additional notes

- Skipping vars that appear in scenarios/groups:
  - Behavior: `_population_margins_with_contexts` skips any effect var present in `scenario_spec` or `group_spec`.
  - Pros: Prevents “effect of x while holding x constant” contradictions.
  - Consideration: Also blocks stratifying by x and computing its effect within those strata (a valid request). This conservative rule may be intentional; if relaxed, guard against contradictions carefully.

- Tests coverage:
  - Could not find tests for `population_margins(...)` with `scenarios` or `groups`. Most validation targets no-context population, profiles, and backend/SE checks. Under-testing increases risk of the above issues slipping through.

## Summary of Critical Issues (must fix)

1) Continuous effects under scenarios are computed at original covariates (not scenario covariates).
2) Categorical effects under scenarios ignore other scenario overrides (only override the categorical var itself).
3) Weighted predictions in contexts misapply weights due to misaligned indexing.
4) Weighted categorical effects in contexts use an invalid scaling hack instead of proper weighted averaging.
5) Continuous subgroup “around value” specs produce empty groups (indices not honored in filtering).
6) Scenario overrides mutate categorical columns without safe type/level conversion (risk of broken design rows).

All six can lead to silent statistical failures and must be remedied or blocked with explicit errors per the Statistical Correctness Mandate.

## Recommended Implementation Approach

- Use DataScenario for all scenario usage:
  - Do not mutate columns; build a scenario object merging all overrides and evaluate per row via `FormulaCompiler.modelrow!(row_buf, compiled, scenario.data, row)`.

- Continuous effects under scenarios:
  - For each row r in the (possibly grouped) subset, compute centered FD under the full scenario (excluding x for ±h):
    - `pred±`, `grad±` via `_predict_with_scenario` and `_gradient_with_scenario!` on `scenario ∪ {x = x_r ± h}`.
    - Accumulate `Δ_i = (pred+ - pred-)/(2h)` and `g_i = (grad+ - grad-)/(2h)`.
    - Average across rows, weighted if provided: `∑ w_i Δ_i / ∑ w_i`, `∑ w_i g_i / ∑ w_i`.

- Categorical effects under scenarios:
  - In `_compute_categorical_contrasts`, merge global scenario overrides into each per-level scenario before computing predictions/gradients; thread weights per row; average as above.

- Weights handling in contexts:
  - Always index weights by original row indices `context_indices` (or pass subsetted weights aligned to context order). Apply weights to both estimator and gradient sums before normalization.

- Continuous subgroup “around value” support:
  - Recognize NamedTuple specs with `:indices` in `_create_context_data` and intersect `indices_to_keep` accordingly.

- Categorical overrides safety:
  - If any column override remains, coerce to correct `CategoricalValue` with original pool/levels (as `_build_refgrid_data` already does). Prefer DataScenario to avoid mutation altogether.

## Testing: gaps and additions

Add focused tests covering:
- `population_margins` with scenarios affecting other covariates (continuous and categorical effect vars); verify against manual per-row scenario evaluation and bootstraps for SEs.
- `groups` with continuous bins and nested specs, including “around value” indices; ensure subgroup n and indices match expectations.
- Weighted contexts (predictions and effects): verify using manual weighted averaging and bootstrap SE comparison.
- Categorical scenario value types: specifying values as strings on `CategoricalArray` columns; ensure proper level handling.

These should fit into the existing statistical validation and bootstrap frameworks.

## Error-First Safeguards (until fixes land)

To avoid silent errors, add explicit errors in these cases:
- `type=:effects` with `scenarios != nothing` (unless the DataScenario-based implementation is in place).
- Weighted categorical effects in contexts (until proper weighted contrasts implemented).
- Weighted predictions in contexts (until weights alignment is fixed).
- Continuous subgroup specs using index-based NamedTuples (until `_create_context_data` recognizes `:indices`).

## What is valid today

- No-context population AME/AAP:
  - Continuous AME: correct averaging and gradients (weighted and unweighted), proper delta-method SEs.
  - Predictions (AAP): correct averaging of predictions and gradients (weighted and unweighted), proper SEs.
- Groups without scenarios (unweighted):
  - Categorical and continuous subsetting implemented; continuous AME aggregation is correct; categorical effects average correctly in the unweighted case.

## Appendix: Key functions reviewed

- `src/population/core.jl`: `population_margins`, `_process_vars_parameter`, `_validate_groups_parameter`, weights processing
- `src/population/contexts.jl`: `_population_margins_with_contexts`, `_parse_at_specification`, `_parse_groups_specification`, `_create_context_data[_with_weights]`, `_compute_population_effect_in_context`, `_compute_continuous_ame_context`, `_compute_categorical_baseline_ame_context`, `_compute_population_prediction_in_context`
- `src/engine/utilities.jl`: `_ame_continuous_and_categorical`, `_compute_categorical_contrasts` (DataScenario), `_compute_variable_ame_unified`, `_accumulate_me_value`, `_average_response_over_rows`, weighted/unweighted gradient accumulation, gradient helpers
- `src/computation/statistics.jl`: `compute_se_only` and delta-method utilities

---

If desired, I can implement the DataScenario-based scenario corrections and proper weighted context handling, add error guards, and create a targeted test suite to validate these behaviors.

## Status Update (Implemented Fixes)

The critical scenario/groups issues identified above have been addressed with statistically correct implementations and tests.

### Implemented
- Scenario-aware continuous effects (population):
  - Evaluated at user-specified scenarios using DataScenario-centered finite differences; averaged gradients for delta-method SEs; supports weights with proper alignment by original indices.
- Scenario-aware categorical effects (population):
  - Merges user overrides into per-level DataScenarios; computes proper weighted averages of both effect and gradient when weights are supplied.
- Weight alignment in contexts:
  - Predictions and effects now index weights by the original row indices for grouped/scenario subsets.
- Continuous subgroup indices:
  - `_create_context_data` recognizes index-based subgroup specs (`:indices`) and intersects indices directly.
- Output semantics:
  - Standard DataFrame: `term` is the variable (effects) or `"AAP"` (predictions); `contrast` carries descriptive labels (e.g., `derivative`, `B vs A`).

### Tests Added
- `test/statistical_validation/population_context_bootstrap_validation.jl`:
  - Bootstrap SE validation for scenario-aware continuous and categorical effects (link scale).
- `test/validation/test_population_scenarios_groups.jl`:
  - Scenario-aware continuous and categorical effects; weighted categorical contrasts; grouped weighted contexts (robust to missing levels per group).

### Confirmations
- `profile_margins` and reference grid builders were not modified. Scenarios/`at` remain unsupported in profile flows. All scenario logic is implemented via DataScenario at population level only.

### Remaining Follow-ups
- Optionally, make the test runner respect `test_args/ARGS` to allow targeted Pkg.test runs.
- Optionally, add a small compatibility wrapper for legacy `cartesian_grid(df; ...)` calls in profile tests, or adjust those tests to the current `cartesian_grid(; vars...) + complete_reference_grid` pattern.

## Plan: Fix Critical Issue 1 (Scenario-Aware Continuous Effects)

### Goals
- Compute continuous marginal effects under counterfactual scenarios (at()) by evaluating both values and parameter gradients at the scenario covariates, not the original data.
- Maintain statistical correctness with proper averaging (weighted/unweighted) and delta-method SEs using the full covariance matrix.
- Avoid silent fallbacks; add error guards where correctness cannot yet be guaranteed.

### Implementation Steps
1) Thread scenario into effect computation path
   - Update `_population_margins_with_contexts` to pass `scenario_spec` into the effect computation for each variable.
   - Change signature: `_compute_population_effect_in_context(engine, context_data, context_indices, var, scale, backend, weights)` → add `scenario_spec` (NamedTuple/Dict) parameter.
   - Continue to skip contradictory requests where `var ∈ keys(scenario_spec)` (already enforced upstream) or raise a clearer error.

2) Add helper to build merged DataScenarios per row
   - New helper: `_build_row_scenarios_for_continuous(engine, var, row, scenario_spec; h=1e-6)` → `(scenario_minus, scenario_plus)`.
     - Reads `x_r = engine.data_nt[var][row]` and creates overrides `{var => x_r - h}` and `{var => x_r + h}`.
     - Merges all user overrides from `scenario_spec` (for other variables) into both scenarios.
     - Uses `FormulaCompiler.create_scenario(name, engine.data_nt, overrides)` to avoid any data mutation.

3) Implement scenario-aware continuous FD + gradient averaging
   - New function: `_compute_continuous_ame_with_scenario(engine, var, context_indices, scale, backend, scenario_spec, weights)`.
     - For each `row ∈ context_indices`:
       - Build `(s−, s+) = _build_row_scenarios_for_continuous(...)`.
       - Compute predictions `p±` via `_predict_with_scenario`.
       - Compute parameter gradients `g±` via `_gradient_with_scenario!` (reusing engine buffers).
       - Per-row effect value `Δ_i = (p+ - p−)/(2h)`; per-row gradient `g_i = (g+ - g−)/(2h)`.
       - Accumulate weighted sums: if `weights === nothing`, use 1.0; else `w_i = weights[row]`.
     - Normalize by `n` (unweighted) or by `∑ w_i` (weighted) to obtain the average effect and the average gradient.
     - Compute SE as `sqrt(ḡ' Σ ḡ)`.

4) Wire into contexts with correct weight indexing
   - `_population_margins_with_contexts` → `_compute_population_effect_in_context` should pass both `scenario_spec` and the original `context_indices` (not reindexed positions).
   - Always index weights by original row indices: `w_i = weights[row]`.
   - Handle zero total weight explicitly with an error (error-first policy) instead of returning 0.

5) Error guards for unsupported edges
   - If `Σ` is unavailable or model lacks required methods, existing validation already errors; keep that behavior.
   - If user sets both `vars` and `scenarios` with the same variable, the existing teaching error remains in force.
   - If all weights in a context are zero, throw a clear error rather than returning a value.

6) Performance considerations
   - Reuse `engine.row_buf` and two gradient buffers (e.g., `engine.de.fd_yplus/fd_yminus` or temporary Vector{Float64} sized to `length(engine.β)` when `de === nothing`).
   - Create two scenarios per row (s−, s+) with O(1) memory each; avoid any formula recompilation.
   - Keep `h` configurable (constant for now) and chosen per backend/link to balance stability and precision; consider exposing as keyword later if needed.

7) Tests (focused)
   - GLM with known derivative under scenario:
     - Simple model `η = β0 + β1 x + β2 z`; compute AME of `x` at `z = z0` via code and compare to manual average of `∂μ/∂x | z=z0`.
   - Weighted variant: construct nonuniform weights and verify weighted averaging equals manual computation; confirm SEs via bootstrap validation.
   - Groups + scenarios: ensure per-group scenario-aware AME equals manual benchmark within tolerance; confirm correct n and (if weighted) proper normalization.

### Acceptance Criteria
- For continuous `x` and scenario `at(z=z0)`, the estimator equals the average of `∂μ/∂x` evaluated at `z=z0` across rows (weighted/unweighted as applicable).
- Gradients average to the correct `ḡ` and SEs match bootstrap validation within configured tolerances.
- Weighted contexts use weights aligned to original row indices, and the implementation errors out when all weights are zero in a context.
- No mutation of data columns for scenarios; only DataScenario overrides are used for evaluation.
