# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog. Version bump is deferred; entries are under "Unreleased" until a tag is cut.

## Unreleased

### Bug Fixes
- **`measure` was silently ignored under `scenarios` / `groups`.** `population_margins(...; measure=:elasticity)` (and `:semielasticity_dyex` / `:semielasticity_eydx`) returned raw average marginal effects labeled `AME (elasticity)` as soon as a context was supplied — `_population_margins_with_contexts` threaded `measure` into the result metadata only, never into the computation. Fixed in `src/population/contexts.jl`: `_compute_all_continuous_context_effects_batch` now applies `apply_measure_transformations` and the `_gradient_of_mean_response` quotient-rule correction, using the context's own x̄, ȳ and ∂ȳ/∂β (computed over the context's rows with scenario overrides applied). Example: for a logit with `groups=:g`, one group previously reported 0.218537 (the raw AME) where the elasticity is 0.513208.
- **Weighted `groups` threw `BoundsError`.** `_create_context_data_with_weights` returns a per-context weight *subset*, but the continuous-effects and predictions paths index weights by absolute row (`weights[row]` for `row in context_indices`), so `population_margins(...; weights=:w, groups=:g)` crashed for both `type=:effects` and `type=:predictions` — a combination `docs/src/weights.md` documents. Both paths now receive the full-length weights vector, matching the categorical path; the convention is documented at each function. Also removed the dead `_compute_population_effect_in_context_clean` and `_compute_continuous_context_effect_batch`, which encoded the opposite convention.
- **Elasticity measures no longer mislabel discrete contrasts.** `measure=:elasticity` with a categorical or boolean variable produced the raw level contrast under an `AME (elasticity)` label, on every path. Elasticities scale a derivative ∂y/∂x by x̄ and/or ȳ, which a discrete contrast does not have (Stata rejects factor variables in `eyex()`/`dyex()`/`eydx()`; R marginaleffects transforms numeric predictors only). Naming a discrete variable in `vars` alongside a non-`:effect` measure is now an `ArgumentError`; auto-detected discrete variables are skipped with an `@info`, so `population_margins(model, data; measure=:elasticity)` keeps working and returns only the rows it can define. Applies to `population_margins` and `profile_margins` (`_filter_vars_for_measure` in `src/core/validation.jl`).
- **The engine cache returned results computed on the wrong data.** `ENGINE_CACHE` (`src/engine/caching.jl`) keys entries on the data's column *names*, but a `MarginsEngine` is bound to the columns it was built from — its derivative and contrast evaluators store those arrays and dereference them at call time. Two datasets sharing a schema and a fitted model therefore collided on one entry, and the second caller silently received the first caller's rows (or a `BoundsError`, when the second dataset was longer). This broke any workflow reusing one model across data slices: hand-rolled subgroup analysis, train/test splits, resampled frames. Retrieval now revalidates a cache hit against the caller's columns by identity (`_engine_data_matches`) and rebuilds when they do not match; `build_engine` costs 0.03-0.13 ms, or 1-3% of a single margins call. Ordinary `Float64`/`Bool`/categorical data still hits the cache, since `_convert_numeric_to_float64` returns those columns unchanged.

New regression tests in `test/validation/test_measure_contexts.jl` validate every measure × scale × weighting × context combination against closed-form oracles built from `modelmatrix`, `coef` and `vcov`, with delta-method SEs checked by central-differencing the estimate with respect to β. `test/validation/test_engine_cache.jl` covers the cache fix the same way — both call orders of the reproducer, a subgroup loop in one session, predictions, profile margins, categorical contrasts on a slice, `Int`-column data (which revalidates to a miss every call), and cache-size bounds.

### Bug Fixes (link scale)
- **`scale=:link` was silently ignored for categorical contrasts.** `population_margins(...; vars=[<categorical>], scale=:link)` returned response-scale contrasts (both estimate and standard error) under a link-scale label. The discrete-contrast kernels select their scale from the `link` argument (`nothing` ⇒ link scale), but the callers passed `engine.link` unconditionally. Fixed on the plain path (`src/population/categorical_effects.jl`) and the `scenarios`/`groups` path (`src/population/contexts.jl`); affects `contrasts=:baseline` and `:pairwise`, weighted and unweighted. Link-scale baseline contrasts now equal the corresponding coefficients exactly, as they must.
- **Elasticity standard errors were wrong on the link scale.** `_gradient_of_mean_response` branched on `scale === :linear`, a value the API never produces, so `scale=:link` fell through to the response branch and weighted ∂ȳ/∂β by `mueta(link, η)` instead of using X̄. Point estimates were unaffected; SEs for `measure=:elasticity` and `:semielasticity_eydx` were off (e.g. 0.279631 reported vs 0.269922 correct on a logit test case). Fixed in `src/population/continuous_effects.jl`.
- **Profile categorical contrast gradients ignored `scale`.** The estimate honored `scale=:link` but the gradient was always computed on the response scale, pairing a link-scale estimate with a response-scale SE (`src/profile/categorical_effects.jl`). Note this path is currently unreachable — see `ISSUES.md`.

All three were invisible to the test suite because every existing test pairing `scale=:link` with a categorical `vars=` used an identity-link linear model, where the two scales coincide. New regression tests in `test/validation/test_link_scale.jl` use logit and probit models.

### Documentation
- Normalized `scale` docstrings that named non-existent values (`:linear`, `:eta`) to the actual API values `:link` and `:response`.
- Added `ISSUES.md` recording three confirmed, unfixed bugs: `measure` silently ignored under `scenarios`/`groups`; profile categorical contrasts unreachable (dead code, with `test/test_profile_pairwise.jl` excluded from `runtests.jl` because it would fail); and a crash when grouping by the variable being contrasted.

## v2.1.3 — 2026-03

### Features
- **FixedEffectModels.jl support** via package extension (`ext/FixedEffectModelsExt.jl`):
  - Average marginal effects (AME) and marginal effects at means (MEM) for models with absorbed fixed effects
  - Predictions (AAP/APM) with FE offset support — requires `save=:fe` at model fit time
  - Profile predictions with FE-level lookup: FE variables in grid use specific level estimates; FE variables not in grid use population-average FE
  - Instrumental variable (IV) models supported with one-time `@info` note about structural coefficients
  - Counterfactual scenarios on non-FE variables work as expected
  - Both `:ad` and `:fd` backends supported
  - Absorbed FE variables blocked from `vars` and `scenarios` with informative errors

### Documentation
- Added Fixed Effects Models guide (`docs/src/fixed_effects.md`)
- Added Stata migration section for `reghdfe`, `ivregress`, and FE predictions
- Added FixedEffectModels.jl to ecosystem integration mentions in index
- Fixed `$` LaTeX interpolation warnings in `second_differences.md` (converted to Documenter `math` blocks and Unicode)

## v2.1.2 — 2026-02

### Documentation
- Comprehensive review and correction of all `docs/src/` files:
  - Fixed CovarianceMatrices.jl API across all docs to use instantiated form (`HC1()`, `Clustered(...)`, `HAC(Bartlett())`) instead of bare types
  - Added missing `type = :effects` to semielasticity examples in index.md and advanced.md
  - Fixed `groups=:region = [...]` syntax errors in grouping.md (5 instances) — these were invalid Julia inside keyword arguments; changed to `groups=(:region => [...])`
  - Removed "Future extension" framing for `second_differences_at()` in second_differences.md — function is already implemented and exported
  - Replaced deprecated `at` parameter section in api.md with reference grid documentation
  - Fixed `Grouping (over)` header to `Grouping (groups)` in api.md
  - Added trailing comma for single-element NamedTuple scenario in api.md
  - Added explicit `type=:effects` to Basic Workflow examples in api.md
  - Fixed confusing variable reuse in api.md Performance Optimization section
  - Changed `:eta`/`:mu` to `:link`/`:response` in computational_architecture.md to match user-facing API
  - Updated performance.md backend recommendations to consistently recommend `:ad` as default
  - Changed production config examples from `backend=:fd` to `backend=:ad`
  - Fixed typo "Standard erros" → "Standard errors" in index.md
  - Changed installation from `Pkg.add(url=...)` to `Pkg.add("Margins")` in index.md
  - Fixed BibTeX triple braces to single braces in index.md
- README review and corrections (prior commit)

## v2.1.1 — 2026-02

### Bug Fixes
- Fixed `balanced_grid` for reference grids
- Test tolerance and CI fixes

### Other
- Updated and cleaned tests
- Updated Project.toml

## v2.1.0 — 2026-01

### Features
- Added Pair syntax support for scenarios (e.g., `scenarios=(:treatment => [0, 1])`)

### Documentation
- Updated citation, removed file reference

## v2.0.3 — 2025-11

### Features
- **Second differences**: Full implementation of `second_differences()`, `second_differences_pairwise()`, `second_differences_all_contrasts()`, and `second_differences_at()` for interaction effects on the predicted outcome scale
- Improved contrast functions with `ContrastResult` type
- Updated elasticity calculation
- Added pairwise contrasts for second differences with proper gradient scaling

### Bug Fixes
- Fixed mixture contrast bug
- Use `levels` order for categoricals, consistent contrast printing
- Docstring corrections

### Documentation
- Added mixture contrast documentation
- Added logo
- Updated examples guide and miscellaneous docs

### Other
- Moved BenchmarkTools from deps to extras
- Docs CI: enabled versioned docs with Documenter

## v2.0.2 — 2025-10

### Internal
- FormulaCompiler integration Phase 3 and Phase 4 complete
  - Full categorical effects migration to ContrastEvaluator primitives (zero allocations)
  - New kernel layer: `src/kernels/categorical.jl` for zero-allocation categorical AME
  - Population, profile, and context effects all using FormulaCompiler primitives exclusively
  - 547 lines of obsolete code removed
- Fixed boolean categorical allocations, `generate_contrast_pairs()`, variable shadowing
- Initial registration to Julia General registry
- All dependencies given proper compat bounds

## v2.0.1 — 2025-09-13

### Docs and CI
- Replaced legacy `at=` examples with explicit reference grids (api, index, performance, examples, mathematical_foundation, computational_architecture).
- Clarified backend policy: no `:auto`, no silent fallback; `backend=:ad` default with explicit `:fd` when requested.
- Consolidated Population Scenarios into `docs/src/population_scenarios.md`; added to User Guide nav.
- Added Weights guide (`docs/src/weights.md`) with correct weighted averaging and delta‑method SEs.
- Fixed grouping grammar (`=>`), corrected `vcov` examples to use `population_margins(...; vcov=...)`.
- Switched Installation to GitHub URL (unregistered) in README and docs.
- Seeded RNGs in doctested examples (seed 06515) to stabilize outputs.
- Added cross-links between API/Grouping/Comparison and Scenarios/Weights.
- Added Docs CI workflow (GH Actions), enabled versioned docs (stable via tags, dev via main), and manual dispatch.

### Other
- Bumped package version to 2.0.1.

## v2.0.0 — 2025-09-12

### Breaking Changes - Type System Redesign

**MAJOR API CHANGE**: Replaced single `MarginsResult` type with specialized result types for improved type safety and DataFrame formatting.

#### New Type System:
- **`EffectsResult`**: For marginal effects analysis (AME, MEM, MER)
  - Contains `variables` and `terms` fields for effect identification
  - Supports multiple DataFrame formats via `DataFrame(result; format=:standard/:compact/:confidence/:profile/:stata)`
  - Auto-detects appropriate format based on analysis type

- **`PredictionsResult`**: For predictions analysis (AAP, APM, APR)  
  - Streamlined design without variable/contrast concepts
  - Single DataFrame format optimized for predictions display
  - Clean tabular output focused on prediction values and statistics

#### Type-Safe DataFrame Conversion:
- `DataFrame(effects_result)` includes variable/contrast columns
- `DataFrame(predictions_result)` omits variable/contrast columns (predictions-focused)
- Both support automatic format detection and explicit format specification
- Tables.jl interface preserved for seamless integration

#### Migration Guide:
```julia
# OLD (v1.x):
result = population_margins(model, data; type=:effects)  # Returns MarginsResult
df = DataFrame(result)  # Generic formatting

# NEW (v2.0):
result = population_margins(model, data; type=:effects)  # Returns EffectsResult  
df = DataFrame(result)  # Type-specific formatting with variables/contrasts
df = DataFrame(result; format=:compact)  # Multiple format options available

result = population_margins(model, data; type=:predictions)  # Returns PredictionsResult
df = DataFrame(result)  # Predictions-optimized formatting
```

### Enhanced Features

#### Display and Formatting Improvements:
- **Stata-style console output**: Professional tabular display with proper alignment and confidence intervals
- **Context-aware headers**: Shows groups and scenarios for population analysis with contexts
- **Type-specific formatting**: Effects vs predictions get appropriate table layouts
- **Configurable precision**: `set_display_digits()` and `set_profile_digits()` for custom formatting

#### Column Naming Consistency:
- **Effects results**: Include `variable` (the "x" in dy/dx) and `contrast` (description) columns
- **Predictions results**: Streamlined without variable/contrast concepts
- **Profile analysis**: Reference grid variables use bare column names
- **Population contexts**: Groups unprefixed, scenarios with `at_` prefix

### Technical Improvements

#### Documentation and Accessibility:
- Updated all docstrings for improved screen reader compatibility
- Enhanced mathematical foundations documentation
- Comprehensive API reference with type-specific examples
- Added accessibility-focused function descriptions

#### Test Suite Enhancements:
- Added column naming validation tests (`test/core/test_column_naming.jl`)
- Enhanced CI and sample size logic validation (`test/validation/test_ci_and_n_logic.jl`)
- Comprehensive validation across 54+ test files maintained
- All statistical correctness guarantees preserved

#### Statistical Correctness Maintained:
- **Zero breaking changes** to statistical computations
- All delta-method standard errors remain unchanged
- Bootstrap validation continues to pass for all scenarios
- Publication-grade statistical standards maintained

### Compatibility Notes

- **Public API**: `population_margins()` and `profile_margins()` signatures unchanged
- **Tables.jl interface**: Fully preserved - existing downstream packages unaffected
- **Statistical results**: Identical numerical outputs to v1.x versions
- **Reference grids**: All grid builders (`means_grid`, `cartesian_grid`, etc.) unchanged

This major version bump reflects the significant type system improvements while maintaining full statistical compatibility and enhancing the user experience with better formatting and type safety.

## v1.0.1 — 2025-09-11

### Changed
- Remove dead/unused helpers with zero functional change:
  - Deleted obsolete categorical context helper and boolean contrast shim
  - Removed unused prediction utilities (`PredictionWithGradient`, single/batch prediction without gradients)
  - Removed unused `_is_linear_model` and minor formatting shim
  - Cleaned unreachable code and fixed a malformed docstring in contexts
- Internal refactors preserve statistical correctness guarantees and public API.

### Documentation
- Revised DEAD.md to accurately reflect which helpers are unused vs active.
- Added RELEASE.md with end-to-end release checklist and statistical gates.

### Tests
- Full test suite passes (core, features, performance, statistical validation, validation); no behavior changes.
