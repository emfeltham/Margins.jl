# Population Margins: Grouping and Scenarios Assessment

This document assesses the statistical and implementation correctness of `population_margins` with respect to grouping and counterfactual scenarios, and compares semantics to Stata’s `over()`/`at()` options. The assessment reflects the current code in `src/population`, `src/engine`, and related test suites.

## Scope and Entrypoints

- Main API: `src/population/core.jl: population_margins(...)`
  - No-context path (baseline AME/AAP): delegates to engine utilities; statistically sound.
  - Context path (with scenarios and/or groups): `src/population/contexts.jl: _population_margins_with_contexts(...)` handles parsing, subsetting, and per-context computation.
- Core helpers used:
  - `src/population/contexts.jl`: `_parse_at_specification`, `_parse_groups_specification` (+ nested/quantile/threshold helpers), `_create_context_data[_with_weights]`, `_compute_population_effect_in_context`, `_compute_population_prediction_in_context`.
  - `src/engine/utilities.jl`: AME/AAP computation, unified categorical contrasts via DataScenario, gradient/SE utilities.
  - `src/computation/statistics.jl`: Delta-method utilities.

## Semantics and Stata Mapping

- groups ≈ Stata over()/within: Stratifies the observed sample and computes population-averaged effects or predictions within each subgroup’s actual covariate distribution.
- scenarios ≈ Stata at() but population-level: Applies counterfactual overrides to the entire population, then computes population-averaged quantities under those counterfactuals. For Stata-style evaluation points, use `profile_margins(..., at=...)` instead.

## Implementation Summary

- Group parsing and nesting:
  - Categorical: `Symbol` or `Vector{Symbol}` for cross-tabs.
  - Continuous: `(:var, n)` for quantiles (Q1/Q2/…), or `(:var, [t1, t2, ...])` for threshold bins with clear interval labels.
  - Hierarchical: `:outer => :inner` (inner within outer), with support for multiple parallel inner specs.
- Subsetting:
  - `_create_context_data` filters with equality for categoricals and range filters for continuous bins; boundary rules avoid overlap. NamedTuple specs with `:indices` are honored for around‑value subgrouping.
- Scenarios in effects:
  - Continuous: Under `backend=:ad`, builds a derivative evaluator on a DataScenario reflecting overrides; averages per-row effects and gradients. Under `backend=:fd`, uses per-row centered FD with two DataScenarios (x±h), averaging values and gradients. No silent AD→FD fallback.
  - Categorical/Boolean: `_compute_categorical_contrasts` merges scenario overrides into per-level DataScenarios and averages both contrasts and gradients (weighted/unweighted).
- Scenarios in predictions:
  - One DataScenario per context; prediction and gradient computed over the subgroup indices with correct weight indexing.
- Backend policy:
  - Strict: if `backend=:ad` cannot be honored, throw a clear error; never auto-fallback to FD.

## Statistical Validity

- Gradient averaging: Per-row gradients are computed in the appropriate context (subgroup × scenario) and averaged correctly:
  - Unweighted: arithmetic mean over rows.
  - Weighted: normalized by the sum of weights across subgroup rows.
- SEs: Delta-method SEs computed as `sqrt(G Σ G')` with the averaged gradient and full covariance matrix Σ.
- Weights: Effects and predictions apply weights by original row index for subgroup rows (no misalignment). Zero-weight-only contexts error explicitly.
- Empty subgroups: Attempting to compute effects on empty subgroups errors clearly to preserve validity.
- Combination limits: Guardrails prevent excessive scenarios×groups combination counts that could compromise reliability.

## Tests and Validation

- Scenarios (bootstrap SE): `test/statistical_validation/population_context_bootstrap_validation.jl` validates continuous and categorical scenario SEs on link scale against bootstrap within tolerance.
- Scenarios × groups, weighted/unweighted: `test/validation/test_population_scenarios_groups.jl` checks continuous and categorical cases, grouped quartiles, and weighted contexts (OLS sanity on link scale).
- Grouping basics: `test/core/test_grouping.jl` covers categorical/mixed/nested/continuous binning and predictions.
- Backend/SE machinery: existing suites cover delta-method correctness and backend behavior; AD/FD semantics are respected with no silent fallback.

## Differences from Stata to Note

- at() vs scenarios: Stata’s `at()` defines evaluation points (profile-style); `scenarios` creates population counterfactuals. Use `profile_margins(..., at=...)` to mirror Stata’s point evaluations.
- dydx(x) over(x): The package currently skips computing the effect of a variable when that variable appears in `groups` or `scenarios`, to avoid contradictory specifications (“effect of x while holding x constant”). This is a conservative policy and differs from some Stata workflows; it can be reconsidered with careful validation.

## Minor Polish Suggestions

- Metadata enrichment: Populate `metadata[:groups_vars]` for vector and nested `groups`, not only when `groups isa Symbol`, to improve display.
- Documentation note: Explicitly mention the skip rule (vars also in groups/scenarios) in the grouping docs for Stata migration clarity.

## Conclusion

The current implementations of groups and scenarios in `population_margins` are correct and statistically valid under the package’s mandate:

- groups implements Stata-compatible over()/within semantics with robust support for categorical, continuous (quantile/threshold), and hierarchical specifications.
- scenarios implements principled population-level counterfactual analysis distinct from Stata’s profile-style `at()`, with strict backend semantics and proper delta-method inference.
- Weighting, gradient averaging, and SE computation follow econometric best practices and are validated by targeted tests, including bootstrap comparisons.

These components are production-ready. The minor suggestions above are non-critical enhancements for metadata and documentation clarity.

