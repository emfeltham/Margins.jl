# Margins.jl Rewrite Plan (Built on FormulaCompiler.jl)

This document lays out a comprehensive plan to rewrite Margins.jl on top of FormulaCompiler.jl. The public API removes the legacy `margins()` entry point and uses statistical names as the canonical interface. Breaking changes are allowed to achieve the best design.

The functionality mirrors Stata’s workflows conceptually, but is Julian and consistent with the JuliaStats ecosystem.

## 1. Vision and Principles

- Foundation-first: Delegate all heavy computation to FormulaCompiler (FC): compiled evaluators, derivatives, FD/AD backends, delta-method SEs, scenarios.
- Statistical names as the API: AME/MEM/MER/APM/APR/APE are the entry points; optional wrappers `effects`/`predictions` delegate to them.
- Zero-allocation where it matters: FD backend for rowwise and AME; AD backend for MER/MEM convenience and accuracy.
- Predictable, composable API: Orthogonal options; defaults that make statistical sense.

## 2. Scope

- In-scope: `ame`, `mem`, `mer`, `ape`, `apm`, `apr`, categorical contrasts, SEs/CI via delta method, grouped results, representative values, tidy outputs.
- Out-of-scope (Phase 1): robust/cluster VCEs, bootstrap/jackknife, plotting, reporting templates. These can come later.

## 3. Core Public API

Statistical entry points (public):

- Row‑averaged over observed data
  - AME: `ame(model, data; vars=:continuous, target=:mu, vcov=:model, weights=nothing, balance=:none|:all|Vector{Symbol}, over=nothing, within=nothing, by=nothing)`
  - APE: `ape(model, data; scale=:response|:link, vcov=:model, weights=nothing, balance=:none|:all|Vector{Symbol}, over=nothing, within=nothing, by=nothing)`

- At profiles (explicit covariate settings)
  - MEM: `mem(model, data; vars=:continuous, target=:mu, vcov=:model, at=:means, over=nothing, by=nothing)`
  - MER: `mer(model, data; vars=:continuous, target=:mu, vcov=:model, at=Dict|Vector{Dict}, over=nothing, by=nothing)`
  - APM: `apm(model, data; scale=:response|:link, vcov=:model, at=:means, over=nothing, by=nothing)`
  - APR: `apr(model, data; scale=:response|:link, vcov=:model, at=Dict|Vector{Dict}, average=false, over=nothing, by=nothing)`

Optional general wrappers (secondary, for discoverability):
- `effects(model, data; vars, target, vcov, weights, balance, over, within, by, at=:none)` → AME when `at=:none`; MEM/MER when `at≠:none`.
- `predictions(model, data; at=:none|:means|Dict|Vector{Dict}, scale, vcov, average=false, weights, balance, over, within, by)` → APE when `at=:none`; APM/APR when `at≠:none`.

## 4. Result Type and Output

- `MarginsResult`: immutable struct holding
  - `table::DataFrame` tidy view with columns:
    - `term::Symbol` — variable or contrast label
    - `dydx::Float64` — estimated effect
    - `se::Float64` — standard error (delta method)
    - `z::Float64`, `p::Float64`, `ci_lo::Float64`, `ci_hi::Float64`
    - `N::Int`, `target::String`, `method::String`, `backend::String`, `link::String`
    - Group columns from `over`/`by` and `at_*` columns for representative profiles
    - For categoricals: `level_from`, `level_to` (or `contrast` string)
  - `metadata::NamedTuple` (β snapshot, Σ, backend, link, model info)
  - Optional `gradients::Vector{Vector{Float64}}` aligned with rows for advanced users

## 4.1 Two Axes and Major Paths

We organize the user API along two orthogonal axes and implement them via two core paths:

- What is computed:
  - Predictions (values on link/response scale)
  - Derivatives/contrasts (continuous slopes vs discrete changes)

- Where it is evaluated:
  - Row‑averaged over observed data (AME/APE)
  - At explicit profiles (MEM/MER/APM/APR)

Major implementation paths:
- Average over observed rows: “Average Marginal/Predicted Effects” (drives AME/APE; supports `weights`, `balance`, `over/within/by`).
- Profiles at explicit settings: “Predictions/Effects at Means/Profile grids” (drives APM/APR and MEM/MER; supports `at=:means|Dict|Vector{Dict}`, optional `average`).

## 5. Computational Mapping to FormulaCompiler

Build once per call:
- `compiled = FormulaCompiler.compile_formula(model, data)`
- `vars = (dydx == :continuous) ? FormulaCompiler.continuous_variables(compiled, data) : collect(dydx)`
- `de = FormulaCompiler.build_derivative_evaluator(compiled, data; vars)`
- `β = coef(model)`; `Σ = resolve via vcov kw (matrix/function/estimator or :model)`; `link` from model when `target=:mu`

Per-row MEs (building blocks):
- η value: `FormulaCompiler.marginal_effects_eta!(g, de, β, row; backend)`
- μ value: `FormulaCompiler.marginal_effects_mu!(g, de, β, row; link, backend)`

Gradients wrt β for SEs (delta method):
- η (single variable): `FormulaCompiler.me_eta_grad_beta!(gβ, de, β, row, var)`
- μ (single variable): `FormulaCompiler.me_mu_grad_beta!(gβ, de, β, row, var; link)`
- AME across rows: `FormulaCompiler.accumulate_ame_gradient!(gβ, de, β, rows, var; link, backend)`
- SE: `FormulaCompiler.delta_method_se(gβ, Σ)`

Adjusted predictions (per row/profile):
- η prediction: value `η = dot(β, X_row)`; gradient wrt β `gβ = X_row` → `se = sqrt(gβ' Σ gβ)`
- μ prediction: value `μ = g⁻¹(η)`; gradient wrt β `gβ = (dμ/dη) * X_row` (link derivative at η) → `se = sqrt(gβ' Σ gβ)`

Aggregations:
- APE: average predictions across rows; gradient = average of per-row `gβ` (weights optional later).
- APM/APR: profile-specific predictions; use single-profile `gβ`.

Elasticities and semi-elasticities:
- Given marginal effect m = ∂y/∂x and value y (η or μ):
  - eyex: (x / y) * m
  - dyex: x * m
  - eydx: (1 / y) * m
Handle y≈0; aggregate via (weighted) mean like AME.

Categorical discrete change:
- η: `ΔX = FormulaCompiler.contrast_modelrow!(Δ, compiled, data_or_scenario, row; var, from, to)` then `effect = dot(β, ΔX)`; gradient `gβ = ΔX`.
- μ: evaluate `X_to/X_from` and `η_to/η_from`; gradient `gβ = g'(η_to)·X_to − g'(η_from)·X_from`; SE via delta method.

MER/MEM:
- Representative profiles `at` from means or user-provided grids; validate vs data (levels/ranges).
- Strategy A (default for MER/MEM): AD scalar-gradient at `x_at` (small allocations; accurate/simple).
- Strategy B (0-alloc alternative): scenarios + FD single-column Jacobians at profile; chain rule for μ; delta method SE.

## 6. MER (Marginal Effects at Representative Values)

- Build representative profile combinations using provided Dict or `:means`.
- For each profile and (optional) group `over`, compute:
  - Values: η or μ marginal effects using `marginal_effects_eta!`/`_mu!` at the profile.
  - Gradients wrt β: use `me_eta_grad_beta!`/`me_mu_grad_beta!` at profile. For μ, compute `η` at profile to obtain link derivatives.
- Output one row per (profile × term × group) into `MarginsResult.table`.
- Profiles are recorded as `at_var1`, `at_var2`, ... columns in the result.

### Profiles UX (APM/MEM/MER/APR)

- Grid averaging: optionally provide a flag (e.g., `average_profiles=true`) to report a single summary averaged across the generated profiles (equal weights), in addition to per-profile rows.
- Labeling/ordering: normalize `at_*` column order (consistent with the order in `at`), and provide readable labels for categorical values.

## 7. Grouping (over/by) and Rows

- `over`: compute effects within each group combination (subset rows for rowwise/AME; apply profiles for MER), return group columns in output.
- `by`: purely stratification for reporting (no change to computation); carry columns to output.
- `rows`: `:all` or vector of indices to restrict to a subset (for AME or rowwise reporting). Defaults to all rows.

## 8. Backend and Performance Strategy

- FD (`:fd`): zero allocations after warmup for rowwise and AME; ideal for large-n production. Uses typed overrides, single-column FD.
- AD (`:ad`): small allocations, high accuracy, and simplest path for MER/MEM at explicit profiles.
- Defaults: `:ad` for MER/MEM; `:fd` recommended for AME or tight loops. Expose `backend` in API and document tradeoffs.

## 9. Architecture and Files

- `src/api.jl` — user-facing `margins/ame/mem/mer`, argument parsing, high-level orchestration.
- `src/engine_fc.jl` — FormulaCompiler integration: build `compiled`, `de`; small wrappers around FC calls.
- `src/compute_continuous.jl` — continuous MEs (rowwise, AME, MEM/MER) + SEs via FC gradients.
- `src/compute_categorical.jl` — contrasts (pairwise/baseline), discrete changes, μ chain-rule gradients.
- `src/profiles.jl` — `at` semantics: means/quantiles, validation, grid expansion.
- `src/grouping.jl` — implement `over`/`by`; subset rows; merge group columns.
- `src/inference.jl` — delta method, CI, z/p-values; (Phase 2+) joint covariances.
- `src/results.jl` — `MarginsResult`, builder from computed values/SEs, tidy DataFrame assembly.
- `src/link.jl` — link extraction/utilities; default to model’s link for `:mu`.
- Polish: a lightweight printer/summary for `MarginsResult` (column order, rounding, group/at labels) to improve default display.

## 10. Implementation Plan

Phase 1 — Core rewrite (Milestone: working `margins()` with AME/MEM/MER)
- Engine adapter: `engine_fc.jl` (FC build + wrappers)
- Continuous MEs (η/μ) for rowwise and AME including SEs (delta method via gradients)
- Adjusted predictions (η/μ) including APE/APM/APR and their SEs
- MER/MEM with `at` (profiles): AD first; FD optional
- Categorical contrasts: η via `contrast_modelrow!`, μ via chain rule; SEs
- Assemble tidy `MarginsResult`
- Examples and basic docs

Phase 2 — Parity and polish
- Grouping (`over`/`by`) with multi-key groups and performance sanity checks
- Robust CI/p-values (normal approx), joint covariance helpers (optional)
- Integration tests against small GLM patterns; cross-validate AD/FD paths
- Documentation: Stata-style recipes; performance characteristics

Phase 3 — Advanced features
- Weighted AME (pass weights to FC accumulator or custom accumulation)
- Robust/cluster VCEs (if practical with available model info)
- Bootstrap/jackknife (batch orchestrations around FC core)
- Plotting/reporting utilities (optional or separate package)

## 11. Testing and Validation

- Correctness:
  - AD vs FD cross-checks for η and μ on small problems
  - AME equals average of per-row effects
  - MEM equals MER with single `at=:means` profile
  - APE equals average of per-row predictions; APM/APR match matrix-based calculations
  - Categorical contrasts vs manual ΔX
- Allocations:
  - FD rowwise/AME achieve 0 bytes after warmup
  - MER/MEM AD paths acceptable small allocations; document numbers
- Numerical tolerances:
  - `rtol=1e-6`, `atol=1e-8` (align to FC guidelines)
- Performance:
  - Benchmarks on synthetic datasets; report per-row and aggregate latencies
  - CI: run tests on basic GLM; skip robust/cluster/HAC tests when CovarianceMatrices.jl is not installed (or guard with feature detection)

## 12.5. Stata Parity Feature Checklist (Roadmap)

- at semantics: `atmeans`, `at((mean|median|pXX) var|all)`, numlist sequences `(a(b)c)`, multiple at() blocks → MER/APR grids.
- asbalanced: treat factor covariates as balanced; combine with atmeans for adjusted treatment means.
- over()/within(): grouped and nested designs; compute within groups/nesting; carry labels to results.
- vce: `:delta` now; roadmap for `:unconditional` (linearization) and `nose` (skip SEs).
- weights: weighted AME/APE aggregations.
- chainrule/nochainrule: control whether to report μ (chain rule) or η directly.
- predict()/expression(): `mode=:predictions` and custom functions of η (with chain rule for SEs).
- mcompare: multiple-comparison adjustments for pairwise contrasts (noadjust, bonferroni, sidak, scheffe).
- df(#): t-based inference when dof is available from the model.

## 12. Documentation Plan

- README: quickstart (`margins` examples), core options, common workflows
- API docs: function docstrings for `margins/ame/mem/mer` and key options
- Cookbook: Stata-style recipes for `dydx`, `at`, `over`, `contrasts`, `target`
- Performance notes: backend selection; allocation behavior; FD vs AD tradeoffs

## 13. Migration Notes (Breaking Changes Accepted)

- Replace old matrix-based internals with FC-driven computation; deprecate stale APIs/types.
- Keep end-user semantics centered on `margins()` with improved consistency.
- Internal names and file layout may change significantly to reflect FC design and Stata parity.

## 14. Example Snippets

AME (η, zero-alloc FD):
```julia
using Margins, GLM, DataFrames, Tables
m = lm(@formula(y ~ x + z), df)
res = ame(m, df; dydx=:continuous, target=:eta, backend=:fd)
res.table
```

MER (μ) at representative profiles (AD):
```julia
profiles = Dict(:x => [-1, 0, 1], :group => ["A","B"])
res = mer(m, df; dydx=[:x,:z], target=:mu, at=profiles, backend=:ad)
res.table
```

Categorical contrasts (μ):
```julia
# Example sketch: contrasts are exposed via vars/levels on MER/MEM paths
res = mer(m, df; vars=:group, target=:mu, at=:means)  # contrasts configured via kwargs
res.table
```

## 15. Benchmarks and Targets

- Continuous rowwise (FD): 0 bytes, ~50–100ns per effect (after warmup), depending on model size
- AME accumulation (FD): O(n_rows × n_vars) with 0 bytes per call after warmup
- MER/MEM (AD): small allocations; target low microseconds per profile per effect

## 16. Notes on FormulaCompiler Integration

We will rely on the following FC APIs (already present):
- `compile_formula`, `length(compiled)`
- `build_derivative_evaluator(compiled, data; vars)`
- Values: `marginal_effects_eta!`, `marginal_effects_mu!`
- Gradients: `me_eta_grad_beta!`, `me_mu_grad_beta!`, `accumulate_ame_gradient!`
- SEs: `delta_method_se`
- Categorical discrete change: `contrast_modelrow!`
- Utilities: `continuous_variables`, scenarios for overrides

Where ergonomic, we may add small FC helpers (future): `marginal_effects_*_at!` with explicit `x_at` for zero-alloc MER on FD paths.

---
Adjusted predictions (per row/profile):
- η prediction: value `η = dot(β, X_row)`; gradient wrt β `gβ = X_row`.
- μ prediction: value `μ = g⁻¹(η)`; gradient wrt β `gβ = (dμ/dη) * X_row` using link derivatives at `η`.
- SE via `delta_method_se(gβ, Σ)`.

Aggregations:
- APE: average predictions; gradient = average of per-row `gβ`.
- APM/APR: profile-specific predictions; use single-profile `gβ`.


This plan enables a first-class, statistical Margins.jl on top of FormulaCompiler with strong performance guarantees and clean semantics. The implementation proceeds in phases, delivering immediate value with AME/MEM/MER and building towards richer inference and reporting.
APM/APR (μ) adjusted predictions:
```julia
res_apm = apm(m, df; target=:mu)                 # at means
res_apr = apr(m, df; at=Dict(:x=>[-1,0,1]))      # profiles grid
res_ape = ape(m, df)                             # average predictions
```

---

Deprecations/removals:
- `margins(...)` is fully removed from the public API and documentation. Use AME/MEM/MER/APM/APR/APE or optional `effects`/`predictions` wrappers.
