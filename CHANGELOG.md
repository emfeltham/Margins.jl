# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog. Unreleased changes go under "Unreleased" until a version is tagged.

## Unreleased

## v3.0.0 — 2026-10

Margins 3.0.0 is the first release after 2.2.0 (2.3.0 was never released). It contains everything below.

Highlights: a new output schema (`statistic` + `df` with t inference for linear models, `over_` group columns, one second-difference schema); population elasticities now match Stata and R; a single model-level `second_differences`; concurrent calls are safe and Margins keeps no engine cache (no stale results after `refit!` or in-place `vcov` edits); a warning when a precomputed column (`x2 = x.^2`) would be held fixed by a counterfactual; numerous correctness fixes. FormulaCompiler ≥ 1.5 is required.

### Breaking: output schema

Four changes to the columns of `DataFrame(result)` and of the second-difference tables. Estimates
and standard errors are unchanged; p-values and confidence intervals change for linear models only.

1. **`t_stat` is replaced by `statistic` and `df`.** The old column was always a z statistic
   (p-values and CIs used the Normal) despite its name. Now `statistic = estimate / se`, and `df`
   is the degrees of freedom of its reference distribution: **t with `dof_residual(model)` for a
   linear model** (`lm`, or `glm(…, Normal(), IdentityLink())`) with any `vcov`, and for a
   FixedEffectModels fit (its own, cluster-adjusted value), and **z (`df = Inf`)** for every other
   model (other GLMs, mixed models). `df=` overrides it (see "Model-class interface" below).
   `p_value`, `ci_lower`/`ci_upper` and the printed intervals use t(`df`). The rule is defined once
   (`Margins._inference_df(model)`), stored in `result.inference.df`, and inherited by
   `contrast(...)` and second differences. `ContrastResult`'s
   `t_stat` field is now `statistic` plus `df`, and its `show` prints `t (df = …)` or `z`.
   Profile effects (`format=:profile`, the default for `profile_margins` effects) now also include
   `statistic`, `df` and `p_value`. `format=:stata` labels the statistic `t`/`P_t` for t inference
   and `z`/`P_z` for z inference. Prediction intervals (`interval=:prediction`) for `lm` also use
   the t quantile, matching R's `predict(lm, interval = "prediction")`.
2. **Group columns get an `over_` prefix** (`groups=:g` gives column `over_g`, as in Stata's
   `over()`), matching the `at_` prefix of scenario columns. This applies to categorical, binned
   and nested groups, to the printed table, and to `contrast(result, (over_g = …,), …)` row lookups.
   Profile (reference-grid) columns keep their bare names; `context_columns(result)` still returns
   raw variable names.
3. **Second differences have one entry point and one schema.** See "Second differences" below.
4. **Profile-effects intervals are `ci_lower`/`ci_upper`** (were `lower`/`upper`), the same names
   as every other table, in both the default profile table and `format=:confidence`.

#### Migration guide

| before (≤ 2.x) | 3.0 |
|:---------------|:----|
| `df.t_stat` | `df.statistic` (and `df.df` for its degrees of freedom) |
| `population_margins(…; ci_alpha=0.1)`, `profile_margins(…; ci_alpha=0.1)` | `alpha=0.1` |
| `contrast(…; α=0.1)` | `contrast(…; alpha=0.1)` |
| `cr.t_stat` (`ContrastResult`) | `cr.statistic`, `cr.df` |
| `df.g` (group column from `groups=:g`) | `df.over_g` |
| `contrast(df, (g = "b",), (g = "a",), Σ)` | `contrast(result, (over_g = "b",), (over_g = "a",))` (see "Typed results" below) |
| second differences `sd.z_stat` | `sd.statistic` (and `sd.df`) |
| `second_differences_at(...).derivative` | `.second_diff` |
| `second_difference(...).z_stat` | `.statistic` |
| profile effects `df.lower`, `df.upper` | `df.ci_lower`, `df.ci_upper` |
| `DataFrame(r; format=:stata).t` (always) | `.t` for linear models (any `vcov`) and FixedEffectModels, `.z` otherwise |
| weighted models (`glm(…; wts=w)`, `fit(MixedModel, …; wts=w)`): `population_margins(m, data)` was unweighted | results are now weighted by the model's weights by default; pass `weights=:none` for the 2.x behaviour |

```julia
# before
res = DataFrame(population_margins(m, data; groups = :g))
res[res.g .== "b", :t_stat]
sd = second_differences(ames, :x, :t, vcov(m)); sd.z_stat
second_differences_at(m, data, :x, :z, vcov(m)).derivative

# 3.0
res = DataFrame(population_margins(m, data; groups = :g))
res[res.over_g .== "b", :statistic]      # res.df: dof_residual(m) for lm, Inf otherwise
sd = second_differences(ames, :x, :t); sd.statistic   # Σ from the result (ames.inference.vcov)
second_differences(m, data, :x, :z; at = :mean).second_diff
```

**Inference change for linear models.** For `lm` (and Gaussian identity GLMs) with the default
`vcov`, p-values and confidence intervals now use t with the residual df instead of the Normal,
so they are slightly wider than before — noticeably so in small samples (at n − p = 20 the 95%
critical value is 2.086 instead of 1.960, about 6% wider; at n − p = 1000 the difference is
0.1%). This matches `GLM.coeftable`, Stata's `margins` after `regress` and R's `emmeans`.
Results for GLMs with non-identity links and mixed models are unchanged. Robust/clustered `vcov` on a
linear model also gives t (see "Model-class interface": this changed again late in 3.0 development).

### Second differences: one entry point, one schema (breaking)

Second differences are redesigned around one function. The old entry points are **removed**, not
deprecated; 3.0 is a breaking release and each has a direct replacement (table below). Calling a removed
name raises an `UndefVarError` with a hint naming its replacement.

**New: `second_differences(model, data, variables, modifier; …)`.** It builds the scenarios, computes the
effects and applies the delta method itself, instead of the user assembling
`population_margins(...; scenarios=...)`, passing the result and Σ to one of five functions, or calling
`second_differences_at` for the continuous case. The modifier type is detected from the data: a discrete
modifier (Bool, 0/1, two-valued, string, categorical) compares effects at its levels (`levels=` selects
them); a continuous modifier gives the local derivative ∂AME/∂modifier at `at` (default the mean), or slopes
between `levels`. Keywords: `levels`, `at`, `modifier_type`, `profile` (strata), `delta`, `scale`,
`contrasts`, `vcov` (a function, a matrix or a CovarianceMatrices estimator such as `HC1()`), `alpha`,
`p_adjust`, `include_gradients`.

**`second_differences(result, variables, modifier; …)`** does the same on an `EffectsResult` you
already hold, from `population_margins` or `profile_margins`. It uses the covariance matrix and df the
result was computed with (`result.inference`), so the standard errors agree with those of the effects;
for another estimator, compute the result with `vcov=…`. Keywords: `contrast`, `modifier_type`, `alpha`,
`p_adjust`, `include_gradients`.

| Removed (2.x) | Use instead |
|:--|:--|
| `second_differences_at(model, data, vars, mod, vcov; at=…, …)` | `second_differences(model, data, vars, mod; at=…, vcov=vcov, …)` |
| `second_differences_pairwise(result, var, mod, vcov; contrast=…)` | `second_differences(result, var, mod; contrast=…)` |
| `second_differences_all_contrasts(result, var, mod, vcov)` | `second_differences(result, var, mod)` |
| `second_differences_table(result, vars, mod, vcov)` | `second_differences(result, vars, mod)` |
| `second_difference(result, var, mod, vcov)` (NamedTuple) | `second_differences(result, var, mod; include_gradients=true)` (one DataFrame row) |

**Output schema.** Every call returns the same columns, in this order, whatever the modifier type
(`Margins.SECOND_DIFFERENCE_COLUMNS`): `variable, contrast, modifier, modifier_type, modifier_level1,
modifier_level2, ame_at_level1, ame_at_level2, second_diff, se, statistic, df, p_value, ci_lower, ci_upper,
significant`. The binary path used to omit `modifier_type` (now `:binary`) and filled `contrast` after the
fact. A local derivative adds `eval_point` and `delta_used` (`modifier_type = :continuous`,
`modifier_level1/2 = eval_point ∓ delta_used`, `second_diff` the derivative, called `derivative` in 2.x), and
every `profile` variable adds a column.
- `ci_lower`/`ci_upper` are the (1 − `alpha`) interval, t or z by the same rule as the test; `significant` is
  `p_value < alpha`. `alpha` defaults to 0.05 (to the `alpha` of a precomputed result).
- `p_adjust=:bonferroni|:holm|:bh` adjusts over all returned rows and adds `p_adjusted` after `p_value`;
  `significant` then uses it.
- **`gradient` is no longer in the output by default**, matching `DataFrame(result)`: it was a vector per
  row, which broke CSV export and printing. `include_gradients=true` adds it after the schema columns.

**Behaviour changes and fixes.**
- **Mixed rows are rejected.** A result that varies over another scenario variable or a `groups=` variable
  as well as the modifier raises an error (naming the variable as a scenario or group variable). Before,
  rows from different cells were paired (e.g. `m: 0 vs 0`) and meaningless differences returned.
- **Level order.** Numeric (and Bool) modifier levels are sorted ascending, so `modifier_level1 <
  modifier_level2` and the sign of `second_diff` no longer depend on scenario order or on which function was
  called. Categorical levels keep scenario order.
- **Inference df is the result's** (the model class's df, or the `df=` given): a second difference of a linear
  model uses t with the residual df whatever covariance it is computed with, as the effects do. Before, the
  discrete functions took `df` from the effects result and ignored `vcov`; during 3.0 development the df briefly
  followed the covariance passed (z for any matrix other than the model's own), which the model-class df rule
  replaced (see "Model-class interface").
- **Second differences the model cannot support are flagged.** When the gradient of the difference is
  identically zero (no interaction in an identity-link model, or the modifier is not in the model) the
  second difference is zero by construction. Such rows report `second_diff = 0`, `se = 0`,
  `significant = false` and `NaN` `statistic`, `p_value`, `ci_lower`, `ci_upper` (they used to report
  `p = 1` and an interval `[0, 0]`), are left out of the `p_adjust` family, and one warning per call lists
  them. A nonlinear model without an interaction term is *not* flagged: the effect still varies with the
  modifier through the link. `second_differences(model, data, …)` also warns when numeric `levels`, `at` or
  `profile` values lie outside the observed range of the variable. `warn=false` silences both.
- `vcov` arguments accept any `AbstractMatrix{<:Real}` (e.g. `Symmetric`) or a function returning one.
- Two-level numeric modifiers typed `:binary` by `:auto` warn when the levels are not 0/1 (the result is the
  raw difference, not a slope); a numeric modifier with at most 5 distinct integer values (probably category
  codes) is treated as continuous with a warning naming `modifier_type=:categorical`.
- A small explicit `delta` near `|at| ≈ 0.1–1` no longer raises "numerically identical" (the distinctness
  check is `pt_minus < pt_plus`); the unused `MIN_ABSOLUTE_DELTA` was removed.
- `profile_margins` effects results work with `second_differences(result, …)` (the difference of marginal
  effects at the profile) and record their covariance. A `PredictionsResult` is rejected with a message to
  recompute with `type=:effects` instead of a `MethodError`.
- Errors are specific: a variable missing from the result lists the variables it has; a contrast that matches
  no rows lists the contrasts available; real errors are no longer reported as "No valid contrasts".
- Internally a result's string labels are read in one place (a typed `EffectIndex` built once per call).
  Contrast labels are canonicalized on the result's side too, so a result spelling a continuous contrast
  `"derivative"` is matched by `"dy/dx"`, and a result whose label vectors and gradient rows disagree in
  length raises `Malformed EffectsResult`.

- The model-level form reads the covariance from the `vcov` argument on every call, so a matrix modified in
  place between calls is respected. `at` accepts any vector (e.g. a range) and a NamedTuple whose modifier
  entry is a vector; `profile` containing the modifier, and a Bool modifier forced to `modifier_type=:continuous`,
  raise specific errors. Integer-valued float codes (1.0/2.0/3.0) trigger the integer-code warning, and
  `warn=false` silences every warning. The result-level form also accepts `vcov=` as a keyword.
- `profile` values are counterfactuals (everyone's value is set), not subgroups; use `groups=` for subgroups.
- **The default for a continuous modifier is now `at=:observed`** (was the derivative at the modifier's mean). The average derivative over the observed values is the quantity the rest of the package reports (an average over the sample, as an AME is), and it equals the derivative at any point whenever the AME is linear in the modifier. In a nonlinear model results change; `at=:mean` restores the old default. Rows from the default report `eval_point = NaN`.
- **`at=:observed`**: the average derivative over the observed modifier values. The modifier column is shifted by ±δ for every row (each row keeps its other values), so the result is the mean over observations of ∂²ŷᵢ/∂x∂m, the average interaction effect, from two `population_margins` calls regardless of sample size. `at=:mean` and numeric `at` set the modifier to one value for everyone, which differs in nonlinear models. Rows report `eval_point = NaN`, `modifier_level1/2 = NaN` and `delta_used`.
- **Result metadata has a documented contract.** `Margins.result_metadata_keys()` lists the keys every `EffectsResult`/`PredictionsResult` carries and their types, typed accessors replace the scattered `get(md, :k, default)` reads (one default each), and a test checks every producer. This fixed three inconsistencies: `:weighted` was missing on `scenarios=`/`groups=` results (it is now always set, `true` when weights were used), `:groups_vars`/`:scenarios_vars` were missing when there were no contexts (now empty vectors), and `:model_type` was always the string `"DataType"` (now the model's type).
- **One name for the significance level: `alpha`.** `population_margins` and `profile_margins` took `ci_alpha`, `contrast` took `α`, and second differences took neither consistently; all four now take `alpha` (confidence level `1 − alpha`; in `second_differences` it also sets `significant`). Migration: `ci_alpha=0.1` → `alpha=0.1`, `contrast(…; α=0.1)` → `alpha=0.1`. Passing `ci_alpha` to `population_margins`/`profile_margins` raises an error naming `alpha`. The result's `alpha` is `result.inference.alpha` (it was `metadata[:alpha]`).
- **`contrast=` selects one contrast in both forms; `all_contrasts` is removed.** The result-level form had `contrast::String="dy/dx"` plus `all_contrasts::Bool`, and the model-level form had `contrasts::Symbol`, which read as the same thing. Now `contrast=nothing` (default) keeps every contrast and `contrast="label"` keeps one (`"derivative"` is an alias of `"dy/dx"`), in the result-level and the model-level form alike. Plural `contrasts=` (model-level, as in `population_margins`) still chooses how contrasts are built, `:baseline` or `:pairwise`. Migration: `all_contrasts=false, contrast="x"` → `contrast="x"`; `all_contrasts=true` → omit. Passing `all_contrasts` raises an error saying so.
- **`measure=`** in the model-level `second_differences` (`:effect`, `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx`): the change in an elasticity or semielasticity across the modifier. Result-level calls already worked on such results.
- **Validated against R `marginaleffects` 1.0.0** (`test/validation/test_second_differences_r_oracle.jl`, reference values from `test/r_compare_marginaleffects/second_differences/`): 72 cases (logit and Poisson; response and link scale; with and without weights; binary, categorical and continuous modifiers; all four measures) agree to 3e-6 in the estimate and 5e-4 in the SE (R's numerical Jacobian limits the SE comparison; where the answer is exact, Margins matches it to rounding, and a closed-form oracle checks the local derivative of an elasticity). Rows that are zero by construction have no SE in R and a zero with NaN inference here.
- **`weights=`** in the model-level `second_differences` (as in `population_margins`; default `:model`, see "Breaking / behaviour change"). Every AME is weighted and `at=:mean`/`:median` use the weighted mean/median. Before, there was no way to pass weights to the model-level form, so it silently returned unweighted results.
- **`vcov` may be a covariance matrix** in `population_margins` and `profile_margins` (square, finite, symmetric, one row per coefficient; copied once per call). Like any `vcov`, it is read afresh on every call, so a matrix changed in place between calls is always picked up (see "Engine cache removed").

### Engine cache removed (behaviour change)

Margins keeps no cache of its own. Every `population_margins`, `profile_margins` and model-level `second_differences` call builds its own engine (`build_engine`), prepares the data columns it needs, validates its inputs and evaluates `vcov` once. Only FormulaCompiler's compiled-formula cache remains (one entry per model and column-type signature; the compiled operations depend on the formula's structure, not on coefficients or data values). The 2.x engine cache (`ENGINE_CACHE`, `src/engine/caching.jl`, deleted) saved at most about 0.1 ms per warm call, and its keys returned stale results:

- **Stale estimates after MixedModels `refit!` (fixed).** The cache key used the model object, which `refit!` mutates in place, so the next call returned the old estimate and SE (est 0.508 instead of 4.016 in the review's probe). Every call now reads `coef(model)` and evaluates `vcov`. Tests: `test/regression/test_stale_results.jl`.
- **Stale SEs after an in-place edit of a `vcov` matrix (fixed).** A matrix `vcov` was keyed by `hash(V)`, which samples the entries of a large array: at p = 201, 124 of 200 single-entry edits left the key unchanged and returned the old SE. Tests: `test/regression/test_stale_results.jl` (edits a diagonal entry that `hash` does not see and checks the SE).
- **Edited inputs skipped validation (fixed).** `VALIDATED_MODELS_CACHE` used the same sampled hash, so a matrix edited in place to contain `NaN` or to be asymmetric was not revalidated. Validation now runs on every call.
- **`vcov` is evaluated once per call.** It was evaluated up to three times per call (validation, engine build, and second differences' own covariance matrix), and once more per inner `population_margins` call of a model-level `second_differences`. `validate_vcov_parameter` now returns Σ, the entry point passes the matrix to `build_engine`, and `second_differences(model, data, …)` evaluates it once for the whole call (its inner calls receive the matrix together with the original estimator, `_PrecomputedVcov`, so the t/z df rule still keys on the estimator). With a CovarianceMatrices estimator this means the estimator's own cost on every call, which the cache used to hide on repeated calls (see "Performance").
- **CovarianceMatrices is called directly.** Estimators were evaluated through `Base.require(Main, :CovarianceMatrices)` + `invokelatest`, which resolves the package through the user's `Main` environment, so it failed (or loaded a different version) when that environment did not list CovarianceMatrices. It is a hard dependency, so Margins now imports it.
- **Concurrent first calls no longer serialize.** `_STATE_LOCK` was held across FormulaCompiler compilation; it is deleted with the tables it guarded (the converted-column memo, the validated-model cache). FormulaCompiler's cache has its own lock and compiles outside it.
- **Data conversions are per call.** `Union{Missing,T}` narrowing, `Int`/`Float32` → `Float64` conversion and model-categorical conversion copy the column on every call (they were memoized per source array, `_NARROWED_COLUMNS`, to keep the cache's column-identity check hitting). `Float64`, `Bool` and categorical columns are still passed through without copying.
- **The model's `vcov` is no longer probed when another `vcov` is given.** `validate_model_methods` called `GLM.vcov(model)` on every call to check that the model has a covariance method; the check now happens where Σ is evaluated (with the default `vcov=GLM.vcov`, the same "model must support vcov()" error), so a model without `vcov` can be used with a matrix or estimator.
- **`clear_engine_cache!()` remains**, as a documented alias that clears FormulaCompiler's compiled-formula cache (Margins has nothing else to clear), so existing scripts keep working.

| Removed | Use instead |
|:--|:--|
| `Margins.get_cache_stats()` | nothing: there is no engine cache to inspect (`FormulaCompiler.cache_info()` for the compiled-formula cache) |
| `Margins.get_or_build_engine(usage, deriv, model, data_nt, vars, vcov, backend)` | `Margins.build_engine(usage, model, data_nt, vars, vcov, backend)` (internal; builds a new engine; `HasDerivatives`/`NoDerivatives` are gone, see "Performance") |
| `Margins.ENGINE_CACHE` | nothing; `Margins.clear_engine_cache!()` still clears FormulaCompiler's cache |
| `empty!(Margins.VALIDATED_MODELS_CACHE)` | delete the line: validation runs on every call |
| `Margins.set_engine_cache_capacity!(n)`, `Margins.engine_cache_capacity()` (3.0 development only) | nothing; `FormulaCompiler.set_compiled_cache_capacity!(n)` bounds the compiled-formula cache |

None of these names was exported. Using one gives an `UndefVarError` whose hint names the replacement. Tests: `test/validation/test_no_engine_cache.jl` (no dataset retained across a 41-replicate bootstrap loop, repeated calls identical, `clear_engine_cache!()` clears FormulaCompiler's cache, removed names), `test/validation/test_engine_data_binding.jl` (formerly `test_engine_cache.jl`: one model on several same-schema datasets), `test/regression/test_int_column_conversion.jl` (formerly `test_int_column_engine_cache.jl`). `test/validation/test_engine_cache_bound.jl` and `test/regression/test_data_prep_memo.jl` are deleted with what they tested.

### One request layer (validation: behaviour change)

`population_margins`, `profile_margins` and the model-level `second_differences` now turn their keywords into one internal `MarginsRequest` (`Margins._normalize_request`, `src/core/request.jl`): every validator runs there once per call, in one order, and `vcov` is evaluated there once (Σ and the vcov source travel in the request, so `_PrecomputedVcov` is gone). The public functions are thin wrappers around `_run(req)`. The model-level `second_differences` builds one request for the whole call and runs each profile point, evaluation point and ±δ shift from it, with no re-entry through `population_margins`' keyword parsing; its modifier, profile and variable checks run once (they ran twice on the local-derivative path). Estimates and SEs are bitwise unchanged (golden snapshot 9419/9419 without regeneration). Inputs that were accepted but invalid now raise errors:

- **`alpha` is range-checked and accepts any `Real` (B8).** `population_margins`, `profile_margins` and `contrast` declared `alpha::Float64`, so `alpha=1//10` or `0.1f0` was a `TypeError`, while `alpha=1.5` (an inverted interval), `0`, negative values and `NaN` were accepted. Now any real number in (0, 1) is accepted and stored as `Float64` (`result.inference.alpha`), and anything else is an `ArgumentError` (`alpha must be in (0, 1), got 1.5`). The model-level `second_differences` already took a `Real` and range-checked it; it keeps its `ErrorException`, which its tests pin.
- **`contrasts` is validated on every call (B12).** `contrasts=:foo` was silently accepted when no categorical variable was requested (`population_margins(m, d; vars=[:x], contrasts=:foo)`, and the discrete-modifier `second_differences`), and the continuous-modifier `second_differences` raised a bare `ErrorException`. Every entry point now raises `ArgumentError("contrasts must be :baseline or :pairwise, got :foo")`.
- **FixedEffectModels results go through the same validators (B7).** The extension's `population_margins`/`profile_margins` skipped the measure filter (a categorical's contrasts were reported as "elasticity" rows), the vars/groups overlap, the scenarios/groups conflict, the scenario-variable and reference-grid predictor checks, and the grouping-column checks. They now normalize the request like every other model; the absorbed fixed effects count as predictors (a grid may set them for prediction), and the FE-specific checks (`save=:fe` for predictions, no absorbed FE in `vars` or in prediction `scenarios`) are model-class hooks (`Margins._validate_focal_vars`, `Margins._validate_request`). The FE offsets themselves are unchanged in this release step (see B4–B6). `profile_margins` on a `FixedEffectModel` also takes `interval` now (`:prediction` is an `ArgumentError`, as for any non-linear-Gaussian model).
- **Model-level `second_differences` on any Tables.jl table (ISSUES #4).** It indexed `data[!, modifier]` and called `names(data)`, so a `NamedTuple` of columns or a row table failed with a `MethodError`; it now reads `Tables.columntable(data)` once. `profile_margins` on a non-`DataFrame` table failed the same way (`process_reference_grid` had only a `DataFrame` method) and works too.
- **`backend` on the model-level `second_differences`** (`:ad` default, or `:fd`), passed through to every AME it computes.
- **Second differences validate their scenario variables once.** The modifier (when set to levels or to `at ± δ`) and the `profile` variables are checked as `scenarios` keys (predictors of the model, not also a focal variable) once per call instead of once per profile point. A continuous-modifier derivative of a variable the fitted model treats as categorical (`contrasts = Dict(:k => DummyCoding())`) is an `ArgumentError` naming `modifier_type=:categorical`; it used to fail deep inside with "value … is not a level of the fitted model's categorical variable".
- **Error order.** Because every check now runs in one place, a call with several invalid options may report a different one first than before (scalar options, then `scenarios`/`groups`, data, weights, variables, grid, `vcov`, model class).
- **Cost.** Warm calls are unchanged within noise (n = 5,000, the paper-style logit, `vcov = V`: APM, AAP, MEM, AME (all), AME (one variable), AME (scenario) and both second-difference forms within ±4%; about 1 KB and 10–30 allocations more per call for the request). First-call latency (fresh process, median of 3): AME 3.64 → 3.70 s, AME with `scenarios` 4.11 → 3.83 s, MEM with `means_grid` 4.98 → 5.03 s, model-level `second_differences` 5.88 → 5.44 s (discrete modifier) and 5.31 → 5.09 s (`at=:observed`).
- Tests: `test/core/test_request_layer.jl`, and a "Shared validation" testset in `test/features/test_fixed_effect_models.jl`. `test/test_second_differences_at.jl` now expects the `ArgumentError` for `contrasts=:invalid`.

### Model-class interface: df, offsets, FixedEffectModels (numbers change)

What Margins needs to know about a fitted model beyond its formula is now one set of internal generics (`src/core/model_interface.jl`), with GLM.jl methods in core and MixedModels/FixedEffectModels methods in the extensions: `_link`, `_coef`, `_default_vcov`, `_response_name`, `_inference_df`, `_model_weights`, `_has_offset`/`_offset`/`_grid_offset`, `_estimation_rows`, `_validate_request`. The scattered `isa`/`hasfield` checks are gone, and **the FixedEffectModels copies of `population_margins` and `profile_margins` are deleted**: a `FixedEffectModel` goes through the shared request path, so scenarios, groups, weights, inference and every validator apply to it as to any model. Several results change:

- **df is a property of the model class (decision D1; E8).** t with `dof_residual(model)` for `lm` and Gaussian identity-link `glm` **whatever `vcov` is passed**, including `HC1()`, `CR1(cl)`, a function or a matrix (as Stata's `regress, vce(robust)` + `margins` and R's `coeftest(m, vcovHC(m))`); before, any `vcov` other than `GLM.vcov` (or a matrix equal to `vcov(model)`) gave z. FixedEffectModels: `dof_residual(model)`, which FixedEffectModels already adjusts for clustering (G − 1 with `Vcov.cluster`) and uses in its own `coeftable`; before, z. Other GLMs and mixed models: z, unchanged. Estimates and SEs do not change; `df`, `p_value` and the intervals do (slightly wider).
- **New `df=` keyword** on `population_margins`, `profile_margins` and the model-level `second_differences`: a positive number, or `Inf` for z (e.g. `df = G - 1` with a clustered `vcov` on an `lm`, Stata's `vce(cluster)` convention). Recorded in `result.inference.df` and inherited by contrasts and second differences.
- **Second differences use the result's df** whatever covariance matrix they are given (the `Σ == vcov(model)` test is deleted), and results no longer record `metadata[:model_vcov]`/`[:model_df]`.
- **Offsets are honoured (B3).** A GLM `offset=` (`glm(f, d, Poisson(); offset=log.(exposure))`) and a GLMM `offset=` enter η = x′β + offset in every kernel that forms η: predictions, the continuous-effects accumulator, the categorical contrast kernel and profile evaluation, so ŷ and the link derivatives are taken at η + offset. Before, offsets were ignored: AAP, response-scale effects and elasticities of an offset model were computed at the wrong η (AAP 0.153 instead of 0.920 in the review's probe). For a Poisson model with an intercept the AAP now equals `mean(predict(m)) = mean(y)`. Population margins use each row's offset (row-aligned under the same length check as `weights=:model`), held at its observed value under `scenarios`. Link-scale and identity-link `:effect` results never depended on the offset and are unchanged.
- **New `offset=` keyword on `profile_margins`, and a new error (decision D3).** A grid row has no offset of its own, so for a model with one, `profile_margins` requires `offset=` (a number or one value per grid row, GLM's `predict(m, newdata; offset=)` convention) or an `offset` column in the grid, whenever the result depends on it (predictions; response-scale effects of a non-identity link; `:elasticity`, `:semielasticity_eydx`). Otherwise it raises an `ArgumentError`; the offset is never silently 0 (for an exposure model that is the rate per unit of exposure).
- **FixedEffectModels predictions are right within groups and under scenarios (B4).** The absorbed fixed effects are a per-row offset, Σₖ feₖ, so each row keeps its own fixed effects; `groups=` no longer adds the sample-average fixed effect to every group, and the results equal averages of FixedEffectModels' `predict` over the group's rows.
- **FixedEffectModels uses its estimation sample (B5).** Only the rows in `esample` are used (rows with missing values, non-positive weights or singleton fixed-effect groups are dropped by `reg`): averages, `groups`, weights and fixed effects refer to them, and `metadata[:n_obs]` is `nobs(model)`. Before, every data row was averaged, with a missing fixed effect counted as 0. The data is subset once, when the request is built; a weight vector for every row of `data` is subset the same way.
- **FixedEffectModels data must be the table given to `reg` (new error).** `data` must have one row per entry of `model.esample` (the original table); Margins subsets it to the estimation rows. Any other number of rows is an `ArgumentError` saying to pass the table given to `reg()`. A table with exactly `nobs(model)` rows used to be taken as already being the estimation sample, a heuristic under which a different table of that length (or the right rows in another order) was misaligned silently; a table subset to `esample` by hand is now rejected too. Fits that drop no rows (`nobs == length(esample)`) are unaffected. This applies to `population_margins`, `profile_margins` and `second_differences(model, data, …)`.
- **FixedEffectModels results carry their inference (B6)** (`result.inference`), so `second_differences(result, …)` works on them, and they get the measure filter and the groups/scenarios validators (B7) and `second_differences(model, data, …)`.
- **FixedEffectModels profile predictions need the fixed effects of each grid row (new error).** Give every fixed-effect variable in the grid (the estimates of those levels are looked up, as FixedEffectModels' `predict` does), or `offset=` / an `offset` column. Before, a fixed-effect variable missing from the grid was silently replaced by its sample average; now it is an `ArgumentError`, as is a level without an estimate. Effects that do not depend on ŷ (`:effect` on an identity link) need neither `save=:fe` nor grid fixed effects, as before. Predictions for a model with a fixed effect interacted with a continuous variable (`fe(g)&x`) raise an error (FixedEffectModels' `predict` does not support them either).
- **FixedEffectModels elasticities include the fixed effects in ŷ.** `:elasticity` and `:semielasticity_eydx` divide by ŷ, which for a `FixedEffectModel` is x′β + Σₖ feₖ; they were computed at x′β alone (wrong unless every fixed effect is 0). They now use each row's fixed effects, and so need `save=:fe` (else the `save=:fe` error).
- **Fixed-effect grid columns are shown.** A FixedEffectModels reference grid that sets fixed-effect variables now labels its output rows with them (they used to be dropped from the profile columns, so two grid rows differing only in the fixed effect looked identical).
- **`weights=:model` on a FixedEffectModels fit warns (decision D2; B11).** A `FixedEffectModel` stores no weights, so whether `reg` was weighted is unknown. `weights=:model` (the default) averages without weights, as before, and now logs a warning once per session saying so and to pass `weights=` explicitly (or `weights=:none` to silence it). Explicit `weights=` and `weights=:none` do not warn. `_model_weights` returns `Margins.UnknownWeights` for these models.
- **Cost.** Without an offset the kernels are compiled without one (`MarginsEngine{L,U,C,E,CE,O}`, `O = Nothing`), so results are bitwise unchanged and the hot loops still allocate nothing (also with an offset). First-call latency (fresh process, the paper-style logit at n = 5,000, median of 3 after vs 6 before): AME 4.08 → 3.86 s, AAP 2.48 → 2.43 s, AME with `scenarios` 4.23 → 4.07 s (all within run-to-run noise), MEM with `means_grid` 5.56 → 5.78 s (+4%, the profile path's offset and grid-column handling compiled on first use). The golden snapshot gained a `df` column and cases for Poisson and GLMM offsets and FixedEffectModels (groups, scenarios, weights, an `esample`-dropping fit).

| before | 3.0 |
|:--|:--|
| `population_margins(lm_model, d; vcov=HC1())` reported z (`df = Inf`) | t with `dof_residual(model)`; `df=Inf` restores z |
| a clustered `lm` with Stata's G − 1 df: compute p-values by hand | `population_margins(m, d; vcov=CR1(d.c), df=G - 1)` |
| FixedEffectModels results: z | t with `dof_residual(model)` (cluster-adjusted); `df=Inf` restores z |
| offset models: offsets ignored | offsets used; `profile_margins` needs `offset=` or an `offset` grid column for predictions and response-scale effects |
| `profile_margins(fe_model, d, cartesian_grid(x=[0, 1]); type=:predictions)` (average FE) | add every FE variable to the grid, pass `offset=`, or use `population_margins(fe_model, d; type=:predictions, scenarios=(x=[0, 1],))` |
| FixedEffectModels averaged over all data rows | averages over `esample` |
| FixedEffectModels: `population_margins(fe_model, data[fe_model.esample, :])` (pre-subset data) | `population_margins(fe_model, data)` (the table given to `reg`; pre-subset data is an `ArgumentError`) |
| `weights=:model` on a weighted `reg`: silently unweighted | still unweighted, with a once-per-session warning; pass `weights=:w` |

Tests: `test/core/test_model_interface.jl` (df rule and `df=`, offsets in every kernel and under scenarios, D3, GLMM offsets, FixedEffectModels inference/esample/groups/scenarios/weights/profile predictions, the D2 warning), offset cases in the zero-allocation tests, and the updated `test/features/test_fixed_effect_models.jl`, `test/core/test_output_schema.jl`, `test/core/test_model_weights.jl`.

### Typed results: inference and context (breaking)

A result now carries two typed fields in place of `profile_values`/`group_values` and the
inference entries of `metadata`, and everything computed from a result reads them:

- **`result.inference::Inference(vcov, df, alpha)`** (new exported type): the parameter covariance
  matrix Σ the delta method used (from `vcov=`), the df of the reference distribution (t(`df`), `Inf`
  = z) and `alpha`. `DataFrame(result)`, `show`, `contrast` and `second_differences(result, …)` read
  it, so they cannot disagree with the result. With `gᵢ = result.gradients[i, :]`, row `i` has
  standard error `√(gᵢ'Σgᵢ)`.
- **`result.context::Union{Nothing,ResultContext}`** (new exported type): the per-row context columns
  as an ordered `NamedTuple` (`context.columns`) with the kind of each column (`context.kinds`:
  `:grid`, `:group` or `:scenario`); `nothing` for a plain AME/AAP. `keys(ctx)`, `ctx[name]` and
  `pairs(ctx)` read it. `group_values` (always `nothing`) is gone.
- **`metadata` keeps provenance only.** `metadata[:vcov]`, `metadata[:df]` and `metadata[:alpha]` are
  removed (and from `Margins.result_metadata_keys()`).
- **`contrast` reads Σ, df and alpha from the result (B9; ISSUES #7).**
  `contrast(result, row1, row2; alpha=result.inference.alpha)` and the new
  `contrast(result, spec1, spec2)`, which picks each row by its `DataFrame(result)` column values
  (`(x=1,)`, `(variable="x1", at_age=60)`, `(over_g="b",)`). Before, `contrast(result, i, j, Σ)` took
  any Σ but always the result's df (a t df paired with an arbitrary covariance), rejected a
  `Symmetric` matrix, and defaulted `alpha` to 0.05 whatever the result's was. The positional-Σ
  methods and the `DataFrame` methods (`contrast(df, i, j, Σ)`, `contrast(df, spec1, spec2, Σ)`) are
  removed; calling one raises an `ArgumentError` naming the replacement. `ContrastResult`'s
  `metadata[:source]` is `:result_object` or `:result_namedtuple`.
- **`second_differences(result, variables, modifier)` loses its positional Σ and its `vcov` keyword.**
  It uses `result.inference`; a matrix argument or `vcov=` raises an `ArgumentError`. For a different
  covariance estimator, compute the result with `vcov=…`, or call
  `second_differences(model, data, …; vcov=…)`.
- **Context columns in grid and user order (B10).** `DataFrame(result)` and `show` list a profile
  result's grid columns in the reference grid's column order (they came out in `Dict` hash order), and a
  population result's `over_` columns, then its `at_` columns, each in the order given in `groups` /
  `scenarios` (the `DataFrame` reversed them before, while `show` did not). Rows, values and
  every other column are unchanged. Note that the grid builders (`means_grid`, `cartesian_grid`, …)
  produce alphabetically ordered columns, so their results now list the grid columns alphabetically;
  a hand-written grid keeps its own column order.
- Estimates, standard errors and df are bitwise unchanged (golden snapshot 9605/9605 rows). The kernels
  still allocate nothing. First-call latency (fresh process, paper-style logit at n = 5,000, median,
  before → after): AME 4.21 → 4.18 s, AME of one variable 3.61 → 3.50 s, MEM 5.94 → 5.99 s, AAP 2.74 →
  2.72 s (within run-to-run noise).

| before (≤ 2.x / early 3.0) | 3.0 |
|:--|:--|
| `result.profile_values` | `result.context.columns` (`result.context[name]`; `result.context.kinds` says which are grid, group and scenario columns) |
| `result.group_values` (always `nothing`) | removed; group columns are `result.context` columns of kind `:group` |
| `result.metadata[:vcov]` | `result.inference.vcov` |
| `result.metadata[:df]` | `result.inference.df` |
| `result.metadata[:alpha]` | `result.inference.alpha` |
| `contrast(result, i, j, Σ)` | `contrast(result, i, j)` (Σ is `result.inference.vcov`); another Σ: recompute the result with `vcov=Σ` |
| `contrast(result, i, j, Σ; alpha=0.05)` on a result computed at `alpha=0.1` (gave 95%) | `contrast(result, i, j)` gives 90%; `alpha=0.05` for 95% |
| `contrast(DataFrame(result; include_gradients=true), i, j, Σ)` | `contrast(result, i, j)` |
| `contrast(DataFrame(result; include_gradients=true), (x=1,), (x=0,), Σ)` | `contrast(result, (x=1,), (x=0,))` (same column names) |
| `second_differences(result, vars, mod, Σ)` / `…; vcov=Σ)` | `second_differences(result, vars, mod)`; another Σ: `population_margins(…; vcov=Σ)` first, or `second_differences(model, data, vars, mod; vcov=Σ)` |
| `EffectsResult(est, se, vars, terms, profile_values, group_values, G, md)` | `EffectsResult(est, se, vars, terms, context, G, inference, md)` |
| `PredictionsResult(est, se, profile_values, group_values, G, md)` | `PredictionsResult(est, se, context, G, inference, md)` |

Tests: `test/core/test_typed_results.jl` (B9: contrast Σ/df/alpha against a hand computation from
`inference`; B10: grid, groups and scenarios order where `Dict` order differs; `Inference` for lm, glm
t and z, HC1 lm, `df=` overrides), and the updated contrast, metadata-contract and second-difference
tests.

### Derived columns: one counterfactual mechanism, and a warning (new `warn` keyword)

- **The formula defines what a counterfactual changes (ISSUES #2).** Margins changes a variable — a
  focal variable's derivative or contrast, a `scenarios` value, a reference-grid value, a second
  difference's modifier or `profile` value — by re-evaluating the formula on the changed data. Terms
  the formula computes (`x^2`, `log(x)`, `x & z`) follow; a column computed outside the formula
  (`x2 = x.^2` in the data) is held fixed. This was always so; it is now documented
  (`docs/src/population_scenarios.md`, "Derived Columns", with the list of transforms a formula can
  use and the note that a precomputed spline or polynomial basis is held fixed) and cross-referenced
  from the `population_margins`, `profile_margins` and `second_differences` docstrings.
- **New: a derived-column warning.** `population_margins`, `profile_margins` and the model-level
  `second_differences` warn, once per call and listing every pair, when a numeric column the formula
  uses (never the response) is in the data an exact function (up to floating-point rounding) of a
  variable the call changes or a focal variable: `v^2`, `v^3`, `sqrt`, `log`, `log10`, `log(v + 1)`,
  `exp`, `1/v`, a linear rescaling (a standardized copy), a product `v * u` with another model column
  (a precomputed interaction). A column the call sets itself consistently (`scenarios=(x=2, x2=4)`,
  a grid giving both) is not reported. There is deliberately no "constant within each value of `v`"
  (lookup) rule: it fires on legitimate group-level covariates (a year's GDP beside `year`), so the
  documented formula-dependence rule governs those columns and the warning reports only columns
  that are almost certainly transforms. It is a warning, not an error, because holding a column
  fixed can be intended. Results are unchanged; only the warning is new.
- **New keyword `warn::Bool=true`** on `population_margins` and `profile_margins`; `warn=false`
  silences the warning. The model-level `second_differences` already had `warn`, which now also covers
  this warning.
- **Internal: one counterfactual mechanism.** Every data override goes through
  `_apply_counterfactuals(data_nt, spec)` with `SetTo(value)` / `Shift(δ)` (internal, not exported):
  `scenarios=` contexts (which the second differences' levels and evaluation points use), the
  second differences' `at=:observed`, and reference-grid completion (`SetTo` of the typical value).
  `at=:observed` is now one population computation with two contexts, `modifier => Shift(-δ)` and
  `Shift(+δ)`, instead of two calls on a copied data table stitched into a synthetic `EffectsResult`;
  the internal `_with_data` and `_shifted_modifier_request` are deleted. Not user-visible: the
  `at=:observed` output is bitwise identical (31 before/after cases — lm, logit, probit, weighted,
  HC1, Poisson with offset, LMM, GLMM, FixedEffectModels, profiles, categorical focal variables, both
  backends, all measures — 848 cells, 0 differences), as are the golden snapshot (9605/9605 rows) and
  every other result.
- Cost: the detector screens each pair on a few evenly spaced rows (up to 2,048 to fit an affine
  relation) and makes a full pass only for a column that passes, so it is O(n) only where
  it finds something. Warm, on the 50,000-row benchmark logit: 0.19 ms for AME (all 8 variables),
  0.05 ms for one variable; the calls themselves are unchanged within noise (AME all 41.57 vs 41.47 ms
  with `warn=false`). It is called through `invokelatest` and its checks are precompiled, so it adds
  little to a first call: fresh-process medians of 6 (before → after) AME (all) 4.13 → 4.08 s, AME (one
  variable) 3.27 → 3.42 s, MEM 5.68 → 5.83 s, AAP 2.58 → 2.46 s.
- Tests: `test/core/test_derived_columns.jl` (the mechanism; positive cases x², log, standardized copy,
  precomputed interaction from either factor; negative cases formula transforms, independent
  covariates, categorical predictors, constant columns, consistent settings, a lookup column on a
  discrete variable and a group-level covariate (not reported); `warn=false`).

### Second differences: `groups=`, one request, `ArgumentError`s; interaction-only categorical variables

- **New: `groups=` on `second_differences(model, data, variables, modifier; …)`** (ISSUES #5): the
  counterfactual second difference *within* each stratum, with the `groups` syntax of
  `population_margins` (categorical, binned `(:x, 4)` / `(:x, [cuts])`, crossed `[:a, :b]`, nested
  `:a => :b`). Each stratum is a block of rows with `over_<var>` columns, in the order of `groups`, after
  the schema columns and before any `profile` columns. A stratum's row equals the ungrouped call on that
  stratum's rows with the same `vcov` (bitwise in the tests). Per `at`: a discrete modifier or `levels`
  sets every row of the stratum to each level; `at=:observed` averages the cross-derivative over the
  stratum's rows; `at=:mean`/`:median` use the stratum's own (weighted) mean/median, as Stata computes
  `at()` statistics within `over()` groups; a number, vector or `at` NamedTuple is the same point in every
  stratum; `profile` points are set within each stratum. δ is the call's, from all rows. All strata come
  from one `vcov` evaluation; with `include_gradients=true` the covariance of two strata's second
  differences is `g₁'Σg₂`. The modifier, a `profile` variable or a held `at` variable in `groups` is an
  `ArgumentError`; for the modifier, the message points to the result-level form for the descriptive
  comparison across observed subgroups. Docs: "Counterfactual and subgroup second differences" in
  `docs/src/second_differences.md`.
- **`second_differences(result, …)` takes strata from a result's `groups=` columns.** It used to raise an
  error when a group variable other than the modifier varied across the result's rows; it now pairs the
  modifier levels within each stratum and adds `over_<var>` columns, so
  `population_margins(m, d; scenarios=(t=[0, 1],), groups=:g)` followed by `second_differences(r, :x, :t)`
  gives the same rows as `second_differences(m, d, :x, :t; groups=:g)` (exactly). A second varying
  *scenario* variable is still an error.
- **The model-level form runs entirely from one request.** Σ, df and `alpha` come from the request
  (`vcov` evaluated once); the separate `alpha` check before it is gone, and the significance level used
  for intervals is the request's. `backend=` is passed through to every AME computation (tests: AD and FD
  agree to 1e-6 on second differences, with and without `groups`). `alpha` accepts any `Real` (it was
  `alpha::Real`-typed; a non-real is now an `ArgumentError` rather than a `MethodError`).
- **Invalid inputs to `second_differences` raise `ArgumentError`** (both forms), as in the rest of the
  package: a bad `alpha`, `p_adjust`, `at`, `delta`, `modifier_type` or `levels`; an unknown modifier,
  focal or `profile` column; a missing contrast label or variable; a result without the modifier, with a
  second varying scenario variable or a repeated level; a `PredictionsResult`; `all_contrasts`. Internal
  consistency errors (a malformed hand-built result) stay `ErrorException`.
- **Removed: the internal `Margins._second_differences_local(model, data, vars, mod, vcov; …)`.** It
  duplicated the public local-derivative path with its own validation, and passed the full data to the
  computation without the FixedEffectModels estimation-sample subset (so `at=:mean` on a fit that drops
  rows averaged over the dropped rows). Use `second_differences(model, data, vars, mod; at=…, vcov=…,
  modifier_type=:continuous)`; its tests were ported to that call.
- **Fixed: a categorical variable that appears only inside interactions** (`y ~ x + g & w`, `g & x & w`,
  `g & log(w)`, no `g` main effect) failed as a focal variable of `population_margins`,
  `profile_margins` and `second_differences` with `Could not find categorical variable g in model
  formula terms`. FormulaCompiler's `baseline_level` reads only main-effect terms; Margins now finds the
  variable's term anywhere in the formula (`_model_baseline_level`). StatsModels codes such a variable
  with `FullDummyCoding` (no level lacks a coefficient), so the first level of its contrasts is the
  reference; the same rule makes a full-rank main effect (`y ~ 0 + g + x`) work, which also failed.
  Results checked against hand computations from `modelcols` (estimate and delta-method SE, identity and
  logit). Bool and continuous variables in the same position already worked and are now tested.
- **`groups` naming a column the data does not have** is an `ArgumentError` naming the available columns
  (was a `FieldError` from inside group parsing), in `population_margins` too.
- Tests: `test/test_second_differences_groups.jl` (exact equality with the result-level route for
  discrete, `levels`, point and vector `at`, `profile`, binned and nested groups, weights, a Tables
  source; the stratum-subset oracle for discrete, `:observed`, `:mean` and `:median`, weighted and not;
  the errors; AD vs FD; `alpha` errors; FixedEffectModels `at=:mean` over `esample`; the
  interaction-only categorical, Bool and continuous variables). Golden snapshot: rows added for
  second differences with `groups=` and for an interaction-only categorical variable; existing rows
  unchanged.
- First-call latency (fresh process, median of 7, vs the previous 3.0 commit): AME (all) 4.52 → 4.56 s,
  AME (one variable) 3.79 → 3.78 s, MEM 6.23 → 6.46 s, AAP 3.02 → 3.07 s, model-level second
  differences (`at=:observed`) 7.09 → 7.36 s; the ranges of the 7 runs overlap in every case. Warm
  timings were not re-measured; a call without `groups=` takes the same path with one stratum.

| before | 3.0 |
|:--|:--|
| `@test_throws ErrorException second_differences(m, d, :x, :t; alpha=2)` (and other invalid inputs) | `ArgumentError` |
| `second_differences(population_margins(m, d; scenarios=(t=[0, 1],), groups=:g), :x, :t)` raised an error | one block of rows per stratum, with `over_g` |
| per-subgroup second differences by subsetting the data and looping | `second_differences(m, d, :x, :t; groups=:g)` |
| `Margins._second_differences_local(m, d, :x, :w, V; at=…)` (internal) | `second_differences(m, d, :x, :w; at=…, vcov=V, modifier_type=:continuous)` |

### Weighted typical values (new `weights=` on the grid builders)

- **New: `weights=` on `means_grid`, `balanced_grid`, `quantile_grid`, `hierarchical_grid` and
  `complete_reference_grid`** (ISSUES #8, review E4). A column name (`weights=:w`) or a vector, checked
  as `population_margins`' `weights=` (real, finite, non-negative, not all zero); `complete_reference_grid`
  also takes `weights=:model` (the model's estimation weights). The typical values become weighted:
  weighted means and medians, weighted shares of `true` for `Bool` columns, weighted frequency mixtures
  for categorical and string columns, weighted quantiles for `quantile_grid` and `hierarchical_grid`'s
  quantile specifications. The default `weights=nothing` gives the same grid as before (bit for bit; the
  golden snapshot is unchanged). A row with weight 0 is dropped.
- **One typical-value computation, `Margins.typical_value(col, typical=mean; weights=nothing)`**
  (internal), now used by every builder, by reference-grid completion (including the frequency mixture
  of a variable the model treats as categorical) and by the second differences' `at=:mean`/`:median`
  (overall and per `groups` stratum), which had their own weighted mean and median.
- **Weighted median:** the smallest value whose cumulative weight reaches half the total; on an exact
  tie (cumulative weight = half the total), the midpoint of that value and the next. Equal weights give
  `Statistics.median` exactly (the average of the two middle values for an even count), integer weights
  give the median of the sample with each row repeated, and rescaling the weights changes nothing (ties
  are detected up to the rounding bound of the cumulative sums). `StatsBase.median(x, weights(w))` was
  not used: it interpolates, and it differs from `median` in the last digits for equal weights. The
  second differences' weighted median (step 8) used the same rule, except that it compared the
  cumulative weight with half the total exactly (so the tie could depend on the weights' scale) and
  could take the midpoint with a zero-weight row's value; both are fixed. The weighted quantiles are the
  weighted analogue of type 7 already used for weighted `groups=(:x, n)` bins; they interpolate, so at
  probability 0.5 they can differ from the weighted median.
- **A custom `typical` function cannot be weighted:** `weights` with a `typical` other than `mean` or
  `median` is an `ArgumentError` (the weights are not silently ignored).
- **`profile_margins` is unchanged:** it evaluates the grid it is given and takes no `weights`. For a MEM
  at weighted typical values build the grid with weights (`profile_margins(m, d, means_grid(d;
  weights=:w))`); the completion it does for variables a partial grid leaves out stays unweighted, so
  complete a partial grid first (`complete_reference_grid(cartesian_grid(x=[0, 1]), m, d;
  weights=:model)`). Docs: "Typical values" in `docs/src/reference_grids.md` and the Weights page.
- Fixed along the way: `means_grid`'s docstring was attached to an internal constant, so `?means_grid`
  showed nothing; `quantile_grid` now marks its filled-in columns as typical values (as `means_grid`
  does), so an integer-coded variable the model treats as categorical gets its frequency mixture there
  instead of an error; `cartesian_grid(x=1:3)` (a range, not a `Vector`) is three values, not one.
- Removed: the unused internal `_build_reference_grid`, `_build_means_refgrid` and
  `_build_explicit_refgrid`.
- First-call latency (fresh process, median of 7, vs the previous 3.0 commit): AME (all) 4.31 → 4.28 s,
  AME (one variable) 3.55 → 3.59 s, MEM (`means_grid`) 6.06 → 6.06 s, AAP 2.77 → 2.78 s, MEM with a partial
  `cartesian_grid` completed by `profile_margins` 6.08 → 6.14 s (+1%; the ranges do not overlap).

### Deterministic row and column order (output order changes; values do not)

Reference grids, `groups=` strata and `scenarios=` combinations now have one documented order. Only the
order of rows and columns changes; every estimate and standard error is the same (the 9740 golden rows,
matched by key, are bitwise equal). Code that reads results by row position may need updating; reading
by key (`contrast(result, (over_g = "b",), …)`, filtering on columns) is unaffected.

- **One nesting convention: the first variable varies slowest** and the last fastest, each in the order
  of its values. `scenarios=` and `second_differences(…; profile=)` used `Iterators.product` order (first
  variable fastest); `groups=[:a, :b]` and `hierarchical_grid` vectors already nested this way.
- **Categorical levels in level order; Bool, strings and numbers ascending.** `groups=` strata and
  `hierarchical_grid` levels followed first appearance in the data.
- **Grid columns:** `cartesian_grid` follows the keyword order (a `Dict` argument is taken
  alphabetically); `means_grid`, `balanced_grid`, `quantile_grid` and `hierarchical_grid` follow the data's
  column order; `complete_reference_grid` keeps the grid's columns and appends the filled-in variables in
  the data's order. All of them used to come from a `Dict`, so columns were alphabetical and
  `cartesian_grid`'s rows nested in `Dict` order. `profile_margins` results follow the grid, so they change
  accordingly.

| call | before | 3.0 |
|:--|:--|:--|
| `cartesian_grid(z=[1, 2], x=[5, 6])` | columns `x, z` (alphabetical); rows `(x, z)` = (5,1), (5,2), (6,1), (6,2) (`Dict` order) | columns `z, x`; rows `(z, x)` = (1,5), (1,6), (2,5), (2,6) |
| `profile_margins(m, d, cartesian_grid(z=[1, 2], x=[5, 6]))` | rows and columns as the grid above | `z, x` columns; z slowest |
| `means_grid(d)` (data columns `y, x, g`) | columns `g, x, y` | columns `y, x, g` |
| `population_margins(m, d; scenarios=(z=[1, 2], t=[0, 1]))` | (1,0), (2,0), (1,1), (2,1) | (1,0), (1,1), (2,0), (2,1) |
| `population_margins(m, d; groups=:g)`, `levels(d.g) == ["lo", "mid", "hi"]`, data starting with "hi" | hi, … (first appearance) | lo, mid, hi |
| `population_margins(m, d; groups=:t)` (Bool, data starting with `true`) | true, false | false, true |
| `population_margins(m, d; groups=:s)` (strings "q", "p") | q, p (first appearance) | p, q |
| `hierarchical_grid(d, :g)` | first appearance | level order |
| `second_differences(m, d, :x, :z; groups=:g)` | strata in first appearance | level order |
| `second_differences(m, d, :x, :z; profile=(a=[1, 2], b=[3, 4]))` | (1,3), (2,3), (1,4), (2,4) | (1,3), (1,4), (2,3), (2,4) |

Migration: select rows by their context columns rather than by position; to keep a specific order,
`sort` the `DataFrame(result)`.

### Validation (R oracles and smoke tests)

New committed R fixtures, with no R needed at test time
(`test/r_compare_marginaleffects/oracle_3_0/`, tested by `test/validation/test_oracle_3_0.jl`),
built with R 4.6.1, marginaleffects 1.0.0 and fixest 0.14.2:

- **The default `at = :observed` second difference** (32 rows: lm, logit, probit, weighted
  logit; continuous and categorical focal variables; `groups=`). R computes it with
  `avg_slopes` over the ±δ-shifted data and `hypothesis`, using the joint Jacobian. Agreement
  is 8.5e-10 in the estimate and 2.3e-9 in the SE (1.5e-5 for elasticities).
- **Estimation weights**, integer and non-integer (216 rows: AME, AAP, MEM, `groups=`).
  Estimates agree to 2.1e-10, and the logit and Poisson SEs to 5.0e-10. Gaussian SEs differ by
  exactly sqrt((n − p)/(Σw − p)), because GLM.jl uses frequency weights and R precision
  weights; the test checks this to 1e-8.
- **Exposure offsets** (156 rows: AAP = mean(y), AME, MEM and APM with the offset at the grid).
  Agreement is 3.1e-11 in the estimate and 2.6e-5 in the SE.
- **FixedEffectModels vs fixest** (129 rows: an esample-dropping two-way FE fit; simple, HC1,
  clustered; predictions including the fixed effects, AME, by state). Estimates agree to
  4.1e-12. SEs differ by FixedEffectModels' one-dof-larger absorbed count, sqrt(275/274) or
  sqrt(286/285) clustered, the same ratio as the coefficient SEs.

`test/validation/test_second_differences_mixed_fem.jl` smoke-tests the model-level
`second_differences` on an LMM, a Bernoulli GLMM and a FixedEffectModels fit that drops rows
(`at=:observed`/`:mean`, `levels`, a binary modifier, `groups=`, result-level form). Each is
checked against hand computations and AD ≈ FD. 56 golden rows for these paths were appended to
the snapshot after verification; existing rows are unchanged.

### Smaller fixes and clean-ups

- **`n` counts only the rows with a positive weight** (behaviour change for weighted calls with zero
  weights). A row with weight 0 contributes nothing to a weighted average, but `n` (the `n` column,
  per `groups` stratum and per scenario, and `metadata[:n_obs]`) counted it: with 500 of 2000 weights
  zero, `n` was 2000; it is now 1500. Applies to explicit weights and to `weights=:model`. Unweighted
  calls and `weights=:none` are unchanged, and so are estimates and standard errors.
- **Prediction intervals with a weighted `lm` are documented, and warn once per session.** The numbers
  are unchanged: `interval=:prediction` on `lm(…; wts=w)` is the interval for a new observation of
  weight 1 with GLM.jl's frequency-weight σ̂² = Σwᵢrᵢ²/(Σwᵢ − p) and t(Σwᵢ − p), the convention of
  `vcov(m)` and the model's df. It equals R's `predict.lm(…, interval = "prediction")` on the data with
  each row repeated wᵢ times (tested), not R's `lm(…, weights = w)`, which treats `w` as precision
  weights. The first such call in a session warns, as R's `predict.lm` does for a weighted fit.
- **`quantile_grid(data; x = n)` gives the `n − 1` cut points** (probabilities `i/n`, `i = 1, …, n−1`),
  as its docstring said: `x = 4` is the quartiles (0.25, 0.5, 0.75), `x = 5` the quintiles, `x = 2`
  the median. It gave `n` values at `i/(n+1)` (`x = 4` → 0.2, 0.4, 0.6, 0.8). A non-integer `n`
  is an `ArgumentError` (was an `InexactError`). Vectors of probabilities are unchanged.
- **`hierarchical_grid` keeps categorical columns categorical**, with the data column's levels
  (including unused ones) and ordering; it returned `String` columns. Columns that mix levels with
  frequency mixtures (e.g. `:r => [:g, (:x, :mean)]`) are unchanged (`Vector{Any}`).
- **No spurious warning for mixtures in `hierarchical_grid`**: `(:g, :mix_proportional)` warned that
  the combination "matches no observations"; a mixture is not a level and is no longer checked.
- **Categorical effects no longer allocate O(n) per call.** The observed levels of each categorical
  variable (`_model_contrast_levels`, `count_categorical_rows`) were collected as a `Set` of
  per-row strings or with `unique` on the column: about 300 KB per variable at 20K rows and 1.2 MB at
  50K. They now come from one pass over the reference codes (Bool columns: `any`/`all`), O(levels)
  bytes. `population_margins(m, d; vars=[:g, :h, :t])` (two categoricals and a Bool): 657 KB → 67 KB
  at 20K rows, 2.43 MB → 67 KB at 50K; time 7.8 → 7.4 ms and 19.3 → 18.7 ms.
- **Tests (D4): every `test/*.jl` test file is now run or gone.** `statistical_validation/function_composition_tests.jl`
  was already fixed and run (closed-form derivative checks). `test_second_diff_bool_fix.jl` (errored:
  its scenario variable was not a predictor) is rewritten against a model where it is, with a
  coefficient-based oracle, and wired in; `test_unified_api.jl` (joint call = separate calls, now
  exact for every row, both contrasts and backends, and within groups) is wired in. Deleted, as
  benchmarks/scripts without assertions, redundant with the allocation and performance suites, or using
  removed APIs: `test_ad_vs_fd_actual.jl`, `test_complex_example.jl`,
  `test_categorical_ame_allocations_bench.jl`, `test_full_dataset_optimization.jl`,
  `test_performance_regression.jl` (wall-clock thresholds) and `test_utilities.jl` (unused copy of
  FormulaCompiler test helpers). Tooling that is not a test stays: `support/` (golden snapshot and
  synthetic-data generators), `r_compare*/` and `performance/benchmark_blas_opt.jl`.
- **Docs:** `weights = :z` (a column) leaves `z` out of the auto-detected `vars`, `weights = data.z` (a
  vector) does not (ISSUES #9), now stated in the `population_margins` docstring and the weights page;
  the performance page no longer claims a whole call allocates 0 bytes (the row loops do; a call
  allocates a constant that does not grow with n); removed FormulaCompiler names that no longer exist
  (`build_derivative_evaluator`, `evaluate_scenario`) and a `(:var, :all)` hierarchical-grid spec that
  never existed (a bare `:var` gives all levels).
- Removed the unreachable NamedTuple branch of the internal `_parse_over_specification` and its
  hard-coded ±2.5-unit window (`_create_continuous_subgroups`).

### Breaking / behaviour change

- **Averages use the model's estimation weights by default (`weights=:model`).** `population_margins` and the model-level `second_differences` ignored the weights a model was fit with: `glm(…; wts=w)` followed by `population_margins(m, data)` gave unweighted AMEs and AAPs, silently, while the standard errors came from `vcov(m)`, which GLM.jl computes with `w` as frequency weights, so the estimate and its SE described different samples. The default is now `weights=:model`: the fit's weights if it has any, otherwise none. For a canonical-link GLM with an intercept, Σwᵢ(yᵢ − μ̂ᵢ) = 0, so the default AAP now equals the weighted mean of y, as in Stata (`margins` uses the estimation weights). `weights=:none` gives the 2.x unweighted result; `weights=nothing` means the same, so code that passed it explicitly is unchanged. A column name or a vector works as before, and explicit weights on an unweighted model are allowed without a warning. Results for unweighted models are unchanged.
  - Weights are read with `Margins._model_weights(model)`: GLM.jl `lm`/`glm` (the response's `wts`), MixedModels.jl `LinearMixedModel` and `GeneralizedLinearMixedModel` (the `wts=` of the fit; `StatsAPI.weights` returns ones for an unweighted fit, so the stored field decides). A FixedEffectModels `reg(…; weights=:w)` stores no weights, so `:model` finds none there: pass `weights=:w`. Other model classes are treated as unweighted. If a model carries StatsBase `AnalyticWeights`, `:model` is an error asking for an explicit choice (frequency and probability weights are used as is); GLM 1.9 stores a plain vector, treated as frequency weights.
  - If the model's weight vector does not have one entry per row of `data` (the fit dropped rows with missing values), `:model` raises an error asking for `weights=` explicitly; rows are not matched by guesswork.
  - **New metadata key `:weights_source`** (`:model`, `:user` or `:none`) in `result_metadata_keys()`, set by every producer, including the FixedEffectModels extension (whose `scenarios=`/`groups=` results also lacked `:weighted`). Model-level `second_differences` frames record it as table metadata, `metadata(sd, "weights_source")`.
  - **Known limitation: `profile_margins` is unweighted.** Its typical values (`means_grid`, `balanced_grid`, `hierarchical_grid`, the fill-in of variables a grid leaves out, and categorical frequency mixtures) are unweighted means and proportions even for a weighted model, and `profile_margins` has no `weights` keyword. Results report `weights_source = :none`. The grid builders take data but no model, and accept any `typical` function, so weighting them is a separate change. To evaluate at weighted typical values, build the grid yourself (e.g. `cartesian_grid(x = [sum(w .* x) / sum(w)])`).
  - Docs: the `population_margins` docstring no longer says weights "enable proper survey inference". Weights change the point estimates; the standard errors come from `vcov` (by default the model's, frequency-weight based for GLM.jl) and are not design-based. Pass a design-based estimator or matrix as `vcov` for those.

- **Continuous-effect contrast label is `"dy/dx"` everywhere.** The `scenarios`/`groups` path (`_compute_all_continuous_context_effects_batch`) and `store_continuous_result!` labelled continuous effects `"derivative"`, while the plain population path and profile margins used `"dy/dx"`. All now use `"dy/dx"`. Code that filters results with `contrast == "derivative"` must change. The second-difference functions' `contrast` keyword now defaults to `"dy/dx"` and still accepts `"derivative"` as an alias.
- **Population elasticities are now means of unit-level elasticities (matching the docs, Stata and R).** `population_margins(...; measure=:elasticity | :semielasticity_dyex | :semielasticity_eydx)` used to compute a *ratio of means*: `(x̄/ȳ)·AME`, `x̄·AME` and `AME/ȳ`, with x̄ and ȳ the (weighted) sample means and a quotient-rule correction `−(ε/ȳ)·∂ȳ/∂β` for the SE (`apply_measure_transformations` and `_gradient_of_mean_response` in `src/population/continuous_effects.jl`). Margins' own documentation (`docs/src/advanced.md`, `docs/src/mathematical_foundation.md`), Stata's `margins, eyex()/dyex()/eydx()` and R's `marginaleffects::avg_slopes(slope = "eyex"|"dyex"|"eydx")` all define the population measure as the *mean of the unit-level quantity*, with gᵢ = ∂ŷᵢ/∂x and ŷᵢ the prediction on the requested `scale`: eyex = mean(gᵢxᵢ/ŷᵢ), dyex = mean(gᵢxᵢ), eydx = mean(gᵢ/ŷᵢ) — weighted means under `weights`, taken over each context's rows under `groups`/`scenarios`. The two agree only when the unit quantities are constant (e.g. eyex and eydx for a log-link model, or dyex with a constant gᵢ). Margins now computes the unit-level definition. The delta-method gradient is the mean of the per-row gradients, e.g. ∇β(gᵢxᵢ/ŷᵢ) = (xᵢ/ŷᵢ)∇βgᵢ − (gᵢxᵢ/ŷᵢ²)∇βŷᵢ with ∇βŷᵢ = g⁻¹′(ηᵢ)Xᵢ (response scale) or Xᵢ (link scale), accumulated in a dedicated row loop (`_accumulate_continuous_measures!`) shared by the plain and the `scenarios`/`groups` paths, on both backends; `measure=:effect` does not go through it, so plain AMEs are computed exactly as before. A row with ŷᵢ = 0 makes eyex/eydx undefined: R silently returns `Inf`/`NaN` for that unit and propagates it into the average; Margins raises a `MarginsError` naming the row. Profile margins (`profile_margins`) are unchanged: at a single grid row the mean of ratios is the ratio, and they already used x and ŷ at the grid row.
  - Worked example (logit `yb ~ x * z + g`, n = 300, the deterministic dataset of `test/validation/test_elasticity_unit_definitions.jl`, response scale, estimate ± SE):

    | measure | ≤ 2.2 (ratio of means) | 3.0.0 | R marginaleffects 1.0.0 |
    |---|---|---|---|
    | `:elasticity` (eyex) | 1.092346 ± 0.104633 | 1.399523 ± 0.192544 | 1.399523 ± 0.192626 |
    | `:semielasticity_dyex` | 0.495197 ± 0.041409 | 0.527332 ± 0.045356 | 0.527332 ± 0.045367 |
    | `:semielasticity_eydx` | 0.622449 ± 0.059623 | 0.953084 ± 0.140625 | 0.953084 ± 0.140680 |

    (R SEs use a numerical Jacobian; Margins' SEs match a central-difference-in-β oracle to 1e-4.) For a Poisson/log model the eyex and eydx values are identical before and after, as they must be (gᵢ/ŷᵢ = β_x), while dyex changes (2.062070 → 2.225984, R 2.225984).
  - **Documentation label swap fixed.** `docs/src/advanced.md` described `:semielasticity_dyex` with the formula and interpretation of eydx (`(∂Y/∂X)×(1/Y)`, "% change in Y per unit change in X") and vice versa, as did the docstring of `_process_continuous_variables!` and `docs/src/mathematical_foundation.md` ("Semi-elasticity (X)/(Y)"); `docs/src/examples.md` used `:semielasticity_dyex` for a "% change in wages per unit change" example. The *computations* always matched the names (dyex = dy/d(ln x) = g·x, eydx = d(ln y)/dx = g/y, as in Stata); the docs now say so, with a Stata/R correspondence table.
  - Tests: `test/validation/test_elasticity_unit_definitions.jl` checks all three measures × {logit, Poisson, lm} × {response, link} × {unweighted, weighted} × {whole sample, `groups`} × {`:ad`, `:fd`} against a closed-form oracle (gᵢ from central differences of `linkinv.(X(x±h)β)`, SEs from central differences in β) and against 120 R marginaleffects 1.0.0 `avg_slopes()` values (`test/r_compare_marginaleffects/elasticity/`). Existing tests that encoded the ratio-of-means definition were updated to the unit-level one: the oracle in `test/validation/test_measure_contexts.jl` (`_measure_oracle`) and the link-scale gradient check in `test/validation/test_link_scale.jl` ("elasticity quotient-rule gradient uses the link scale").
  - This supersedes the ratio-of-means normalisation that v2.2.0 introduced for `scenarios`/`groups`.

### Features

- **One entry point for second differences**, `second_differences(model, data, variables, modifier; …)`, plus a result-level form: see "Second differences: one entry point, one schema".
- **Concurrent Margins calls are safe.** `population_margins`, `profile_margins` and everything built on them may be called from several tasks or threads at once (e.g. a `Threads.@threads` bootstrap), on the same model and data or on different ones; each result equals the serial one exactly. Previously concurrent calls shared one cached engine and FormulaCompiler's cached compiled formula, whose scratch buffers they overwrote: 64 of 64 concurrent `population_margins` calls on one model returned wrong estimates. Now each call builds its own engine (the engine cache is gone, see "Engine cache removed"), and every engine and every reference-grid evaluation uses a private copy of the compiled formula's scratch buffer (`_private_compiled`, `src/core/data_conversion.jl`; the shared cached object is used only for read-only inspection, `_shared_compiled`). The only state shared between calls is FormulaCompiler's compiled-formula cache, which has its own lock and compiles outside it; Margins holds no lock during computation (only a locked memo of model-type names for metadata). FormulaCompiler evaluators used directly (`compile_formula`, `derivativeevaluator`, ...) and engines from the internal `Margins.build_engine` remain per-task. Tests: `test/validation/test_concurrent_calls.jl` (runs `concurrent_calls_checks.jl` in a `-t 4` subprocess when the suite is single-threaded: 132 `Threads.@spawn` and 132 `Threads.@threads` calls mixing AME (AD/FD, both scales), AAP, groups, scenarios, elasticities, MEM/APM, means and hierarchical grids, plus 128 calls on eight same-schema datasets, 88 first calls racing on an empty compiled-formula cache, and 48 calls with `HC1()` and a matrix `vcov`; every result compared with `isequal` to its serial counterpart). The reproducer `r5_threads.jl` now reports 0 / 64 wrong.

### Dependencies

- Requires FormulaCompiler ≥ 1.5 (compat `"1.5"`). FormulaCompiler 1.5 resolves categorical codes through the fitted model's own levels (a `LevelIndex` on `ContrastOp`), so data pools that are reordered, rebuilt or smaller than the fitting data's evaluate correctly, and missing or unknown categorical values throw instead of silently mapping to the reference level. It also adds `categorical_levels` and the public mixture validators used here.
- MixedModels.jl is now a weak dependency (compat `"4.35"`), loaded through the new `MixedModelsExt` extension. FormulaCompiler already depends on it, so nothing extra is installed.

### Performance

- **Per-call cost of removing the engine cache** (warm medians, the paper's 65-coefficient logit, Apple M-series; estimates and SEs bitwise identical). With the default `vcov`: +0.16 to +0.33 ms for APM, MEM, AAP, AME (one variable) and AME (scenario) at n = 5,000, +0.02 to +0.5 ms at n = 50,000; AME (all variables) +1.2 ms (+6%) at n = 5,000 and +8 to 13 ms (+4 to 7%) at n = 50,000. The setup Margins now repeats (validation and `vcov`, data preparation, engine build) costs about 0.3 ms; the rest of the AME (all) increase is not Margins work at all: it is `GLM.vcov`, now evaluated on every call. Its threaded LAPACK call (a 65 × 65 Cholesky inverse) wakes OpenBLAS's worker threads, which keep spinning for a while afterwards, and on Apple-silicon machines the single-threaded AME loop that follows runs about 5% slower while they spin. Any threaded LAPACK call has the same effect (an unrelated `inv(cholesky(A))` of the same size does too), and so does a pure-Julia loop. With `BLAS.set_num_threads(1)`, or with the covariance passed as a matrix (`vcov = V`, no LAPACK call per call), AME (all) takes 18.9 ms at n = 5,000 and 186–188 ms at n = 50,000, against 20.0 and 195–198 ms with the default `vcov`. That gap is the whole step-2 increase. A freshly built engine computes as fast as a reused one (189–192 ms either way at n = 50,000). `Int` columns are copied to `Float64` on every call (8 bytes per row per column; +2.8 MB per call at n = 50,000 for the 7 integer columns of that dataset). With a CovarianceMatrices estimator the estimator now runs on every call: +0.5 to 0.9 ms at n = 5,000 and +7 ms (`HC1()`) / +4 ms (`CR1(cluster)`) at n = 50,000, exactly `CovarianceMatrices.vcov`'s own cost (and ~27 MB, its n × p score matrix). To reuse a robust covariance across many calls, compute it once and pass the matrix: `V = CovarianceMatrices.vcov(HC1(), m); population_margins(m, data; vcov=V)` (inference stays z, as for the estimator). First-call latency is unchanged (AME (all) 4.8 s → 4.4 s, the others within ±0.1 s).

Internal changes with identical estimates and standard errors (verified on the six benchmark operations — APM, MEM, AAP, AME (all), AME (one variable), AME (scenario) — at n = 5,000 and 50,000):
- **Profile engines no longer convert the whole dataset to dual numbers.** Profile margins build their own evaluator over the reference grid and only checked that `engine.de` existed, but the AD backend converted every continuous column over all n rows to `Dual`s (~1 GB at n = 500,000). Profile engines now carry no evaluator at all (see "Concrete engine type" below).
- **Concrete engine type.** `MarginsEngine{L,U,C,E,CE}` records the exact types of its derivative evaluator (`de::E`) and contrast evaluator (`contrast::CE`); `Nothing` when absent. Before, both fields were abstract, so every AME call dispatched dynamically through them. Together with the next two changes, the continuous-effects accumulator now allocates 0 bytes per call for `:effect` and for every measure, on both backends. Before it allocated 96 bytes for `:effect` and 416 bytes for a measure.
  - The data columns of the measures that multiply by x are a `Vector{Vector{Float64}}` (in the engine's `xcols_buf`). Before, they were a runtime-built heterogeneous tuple, which forced a dynamic call of its own.
  - The measure is resolved to a type (`_measure_type`) at the entry point's function barrier, so the first call compiles only the requested measure's kernel.
  - The `D<:DerivativeSupport` parameter (`HasDerivatives`/`NoDerivatives`) is removed, since `E` records the same thing. `build_engine(usage, model, data_nt, vars, vcov, backend; evaluators)` builds evaluators for population engines only. Profile engines evaluate on the reference grid, and the `scenarios`/`groups` path builds its evaluators per context, so neither carries any. Profile engines also built a contrast evaluator over all n rows that was never read.
  - `verify_and_repair_engine_buffers!` and the dead `de` checks in `validate_engine_for_computation` are deleted.
  - Warm times (paper model, `GLM.vcov`): MEM and APM 7–8% faster at n = 5,000, MEM 8% at n = 50,000; the other operations within ±2%.
  - First-call latency (fresh process, median of 3): `population_margins` AME 4.64 → 4.41 s; with `scenarios` 4.53 → 4.54 s; `profile_margins` with `means_grid` 6.96 → 6.16 s; a second AME call with a different `vars` set 0.86 → 0.79 s.
- **Context evaluators differentiate only the requested variables.** Under `scenarios`/`groups`, `_population_margins_with_contexts` (`src/population/contexts.jl`) built the derivative evaluator over every continuous model variable, widening the dual numbers and converting each such column for every context; it now builds it for the requested continuous variables only, as `build_engine` does, and only when there are continuous variables to compute.
- **Fused AD response-scale AME loop.** A specialised `_compute_me_loop_response!(…, de::ADEvaluator, …)` with a `_fused_me_rows!` function barrier (`src/population/continuous_effects.jl`) accumulates ∂AME/∂β as Σwᵢ[g′(ηᵢ)Jᵢ + g″(ηᵢ)(Jᵢ′β)xᵢ] column-wise, instead of materialising the per-row n_params × n_vars matrix Gβ and making strided writes into `batch_gradients`.
- **Categorical AME gradients reuse the contrast rows.** `categorical_contrast_ame!` (`src/kernels/categorical.jl`) called `contrast_gradient!` per row, re-evaluating both counterfactual model rows that `contrast_modelrow!` had just produced; the new `_accumulate_contrast_gradient!` reuses X₀/X₁ from the evaluator's buffers (link scale: X₁ − X₀; response: g′(η₁)X₁ − g′(η₀)X₀).
- **Counterfactual reset helper.** `_contrast_gradient_response_scale!` reset counterfactual state with FormulaCompiler's `reset_all_counterfactuals!` instead of three hand-written loops. (Superseded within this release: Margins' copy of `contrast_gradient!` was removed in favour of FormulaCompiler's — see "Removed".)
- **Reference-grid completion skips grid-specified columns.** `complete_reference_grid` (`src/profile/reference_grids.jl`) computed typical values (means, categorical frequency tables — O(n) passes) for every model variable and then discarded those the grid specifies; it now computes them only for the unspecified variables.

### Bug Fixes

- **`profile_margins` with two or more categoricals in `vars` evaluated each contrast at the wrong covariate values.** The evaluator used for categorical contrasts at grid rows (`_mem_continuous_and_categorical`, `src/profile/core.jl`) replaced *every* requested categorical column with a placeholder (its first level, or `false`) so that it could carry the model's full level pool. Contrasting one categorical therefore held the others at the placeholder instead of at the grid row's value — a pinned level or a frequency mixture — so estimates and SEs depended on which other categoricals happened to be requested: for a logit `y ~ x * b + g` at `g = "b"`, the effect of `b` was 0.180 ± 0.025 with `vars=[:b, :g]` (its value at `g = "a"`) and 0.124 ± 0.018 with `vars=[:b]` (correct). Link-scale contrasts of a variable that does not interact with the others were unaffected. Each contrasted variable now gets its own evaluator in which only that column is replaced. Tests: `test/regression/test_profile_multi_categorical.jl` (pinned and mixture grids, both scales and backends, against design-row oracles with central-difference-in-β SEs).
- **A `groups` cell whose weights are all zero reported continuous effects as `0.0 ± 0.0`.** Categorical effects and predictions already errored for such a cell; continuous effects (fused AD loop, generic and FD loops alike) divided zero sums by a total weight of 0 and skipped every row, returning a precisely estimated null. All three now raise the same `MarginsError` ("All weights are zero for the observations being averaged …") from a shared `_total_weight` / `_throw_zero_weights` (`src/population/continuous_effects.jl`); categorical effects and predictions raise `MarginsError` instead of a bare `ErrorException`. Cells with only *some* zero weights are unaffected. Tests: `test/regression/test_zero_weight_contexts.jl`.
- **`Union{Missing, T}` columns crashed.** Columns whose element type admits `missing` — produced by CSV.jl, `allowmissing`, and `dropmissing(df, cols)` (which narrows only `cols`) even when they hold no missing value — failed deep inside FormulaCompiler with `MethodError: no method matching fieldindex(::Type{Union{Missing, Float64}}, ...)` (effects), a `convert` error (predictions), or "Unsupported data type" (`means_grid` and the other grid builders). Such columns are now narrowed to their non-missing type on entry (`_narrow_missing_eltype`, `src/core/data_conversion.jl`, also applied by the grid builders and `get_typical_values`); results are identical to those on the narrowed data. The narrowed copy is memoized per source array (weakly keyed, revalidated against the source's contents) so repeated calls on the same data still hit the engine cache and in-place edits are still seen. A model variable (response included) that *does* contain missing values now raises an actionable `MarginsError` naming the columns and their missing counts and suggesting `dropmissing(data, [model variables])` — those rows cannot be evaluated and were not part of the estimation sample. Missing values in columns the model does not use are ignored. Tests: `test/regression/test_missing_eltype_columns.jl`.
- **Linear mixed models failed unless `MixedModels` was bound in `Main`.** `_auto_link` (`src/engine/core.jl`) looked MixedModels up as `Main.MixedModels`, so `import MixedModels as MM`, or loading it from inside another module, gave "Cannot determine link function for model type LinearMixedModel". (GLMMs worked through MixedModels' own `GLM.Link` method.) The link now comes from a new package extension, `ext/MixedModelsExt.jl` (MixedModels added as a weak dependency, compat `4.35`), that dispatches on `LinearMixedModel`/`GeneralizedLinearMixedModel`. Tests: `test/regression/test_mixedmodels_link_without_main.jl` (runs in a fresh process, since `runtests.jl` binds `MixedModels` in `Main`).
- **p-values underflowed to exactly 0.** Two-sided p-values were computed as `2(1 − cdf(Normal(), |z|))`, which is 0 once the CDF rounds to 1 (|z| ≳ 8.3). All seven sites (`src/types.jl` ×3, `src/computation/statistics.jl`, `src/population/effects.jl`, `src/profile/core.jl`, `src/second_differences/contrasts.jl` ×2) now use `2·ccdf(Normal(), |z|)`, accurate to |z| ≈ 38. Tests: `test/regression/test_pvalues_and_version.jl`.
- **Data whose categorical pool lacks a model level could not be analysed.** `population_margins(m, subset; vars=[:g])` on a subset re-`categorical`-ized without the reference level (pool `["b", "c"]` for a model fitted on `["a", "b", "c"]`) failed with "Baseline level 'a' not found in categorical variable :g"; `_process_categorical_variables!` caught that, logged a warning and carried on, and the half-filled results table then threw `UndefRefError`. Contrasts are now enumerated from the fitted model's levels (FormulaCompiler's `categorical_levels`, model order) rather than the data's pool, in `generate_contrast_pairs` (`src/computation/scenarios.jl`) and the profile path (`_match_pool_level`, `src/profile/core.jl`); pairs carry raw level values, which FormulaCompiler ≥ 1.5's `ContrastEvaluator` resolves against the model's levels even when they are absent from the data's pool (it pads an internal copy; the caller's data, and so the engine cache entry, is untouched — a repeated call on such data hits the cache). `count_categorical_rows` counts observed non-baseline levels instead of assuming `n_levels − 1`. Errors while computing a categorical variable's contrasts are now logged and rethrown rather than turned into a partially filled table. Tests: `test/regression/test_categorical_pool_alignment.jl` (subset without the reference level, reordered pool; design-row oracles, both backends).
- **Numeric columns the model treats as categorical were handled as continuous.** Margins decided "continuous vs categorical" from a column's element type, while the model decides from its contrasts. An integer column fitted with `contrasts = Dict(:k => DummyCoding())` therefore made `population_margins` fail ("Categorical variable k not found in ContrastEvaluator level maps"), `groups=:k` was rejected as an unbinned numeric variable, and `profile_margins(m, df, means_grid(df))` evaluated the model at the mean code (k = 1.94, not a level). The model now decides: on entry (`_prepare_margins_data`, `src/core/data_conversion.jl`) every column that FormulaCompiler's `categorical_levels` reports as model-categorical but that is not a `CategoricalArray` (integer, Float or Bool codes; a Bool used as a continuous regressor is not model-categorical and is untouched) is converted to a `CategoricalArray` with the model's levels (memoized per source array, so repeated calls still hit the engine cache), so every downstream decision — variable detection, contrast enumeration, `groups`, `scenarios`, typical values — follows the model. The number of contrasts per variable comes from `categorical_levels` instead of the evaluator's `CategoricalLevelMap`s. Grids built from data alone cannot know the model: at completion (`complete_reference_grid`) a model-categorical variable's typical value is its frequency mixture over the model's levels. The data-only builders that fill typical values (`means_grid`, `balanced_grid` for its non-balanced columns, fully-typical `hierarchical_grid` columns) now mark those columns (DataFrames column metadata `"Margins.typical_value"`), and only a marked column holding a non-level value (e.g. `means_grid`'s mean code k = 1.94) is replaced by the mixture. A value the user chose that is not a level of a model-categorical variable (e.g. `cartesian_grid(k=[1, 2, 9])`) raises a `MarginsError` naming the variable, the offending values and the valid levels. The data-only grid builders (`means_grid`, `cartesian_grid`, `balanced_grid`, `quantile_grid`) still classify columns by type, since they have no model. As a consequence a raw `String` model variable, previously rejected with "Variable :x is a raw String column … use categorical()", now works and gives the same results as its `CategoricalArray` equivalent (it is converted with the model's levels; this also avoids FormulaCompiler's per-row level lookup for raw String columns). Tests: `test/regression/test_integer_coded_categorical.jl` (AME of all variables on both backends, `groups=:k`, `scenarios`, `means_grid` and `cartesian_grid` profiles, cache reuse; oracles from `modelcols` with central differences in β).
- **Negative-binomial-link response-scale SEs were wrong; Margins no longer uses `GLM.mueta`.** Every dμ/dη in Margins (population and profile prediction gradients in `src/population/effects.jl`, `src/computation/predictions.jl`, the profile elasticity quotient-rule term in `src/profile/core.jl`, and the new unit-elasticity loop) came from `GLM.mueta`, which for `NegativeBinomialLink(θ)` returns θeᶯ/(1−eᶯ) — μ itself — instead of θeᶯ/(1−eᶯ)² (3.083 vs 7.835 at θ = 2, η = −0.5). Response-scale prediction SEs for such models were therefore too small (e.g. AAP SE 0.0319 vs a correct 0.0766 on the test model), and elasticity gradients were wrong. GLM.jl also plans to remove `mueta` (GLM.jl #591). All call sites now use FormulaCompiler's `_dmu_deta`, which is analytic for the negative-binomial and power links and correct for every link. Tests: `test/regression/test_negbin_link_derivative.jl` (AAP, APM, AME and elasticity for a `glm(..., NegativeBinomial(θ), NegativeBinomialLink(θ))` fit, against central-difference oracles).
- **Profile elasticity SEs dropped the quotient-rule term when the scale factor was exactly 1.** In `_process_profile_continuous_variable!` (`src/profile/core.jl`) the correction −(ε/ŷ)·∂ŷ/∂β for `:elasticity` and `:semielasticity_eydx` was applied only inside `if transform_factor != 1.0`, so at a grid point with x == ŷ (eyex) or ŷ == 1 (eydx) the estimate was right but the SE omitted that term. The scaling and the correction are now always applied. Profile measures that divide by ŷ also raise the same `MarginsError` as the population path when ŷ is exactly 0 at a grid row, instead of returning `Inf`/`NaN`. Profile measures are the unit definitions at the grid row (eyex = g·x/ŷ, dyex = g·x, eydx = g/ŷ), consistent with the population change above. Tests: `test/regression/test_profile_measure_quotient_rule.jl`.
- **`Margins.VERSION` was stale** (`v"2.0.0"` while Project.toml said 2.2.0). It is now `pkgversion(@__MODULE__)`, read from Project.toml.
- **`hierarchical_grid` columns only some combinations specify.** In `:g => [(:k, [1, 3]), (:z, :mean)]` the `k` column holds user values in some rows and the typical value (the mean code) in others. Such columns were not marked as typical, so for a variable the model treats as categorical (an integer fitted with `DummyCoding`) the typical value raised "not levels of this variable" instead of becoming the frequency mixture. Typical cells are now recorded per row (`_mark_typical_value_rows!`, `:default`-style column metadata, which DataFrames keeps through `copy` but drops on any edit that moves or overwrites rows, so an edited grid is validated as user-specified rather than having the wrong cells replaced); `complete_reference_grid` replaces only typical cells with the mixture and validates every user-specified cell, reporting only the offending specified values. A typical cell is replaced even when the mean code happens to equal a level (balanced codes 1/2/3 give mean 2.0, which is not a request for level 2) — this also changes `means_grid` in that case.
- **Reference grids whose categorical columns mix levels and mixtures could not be evaluated.** FormulaCompiler compiles a column either as discrete or as a single mixture, so such a grid failed with "Categorical mixture objects should not reach a standard ContrastOp" — for hierarchical grids this also affected `CategoricalArray` columns (`:g => [:h, (:x, [0, 1])]`). `profile_margins` now evaluates these grids in blocks of rows homogeneous in each such column and reassembles the results in grid-row order (`_heterogeneous_grid_blocks`, `src/profile/core.jl`); other grids take the unchanged single-pass path. Tests: `test/regression/test_hierarchical_partial_typical.jl` (integer-coded and `CategoricalArray` columns, predictions and effects against design-row oracles with analytic delta-method SEs; validation of specified cells; sorted grids; balanced codes).
- **Profile predictions copy out of the engine's buffers.** `_profile_predictions` could store views of the engine's `η_buf`/`g_buf` in the result, which a later call on the same engine overwrites; it now copies them.
- **Weighted population calls no longer copy the weight column.** `_process_weights_parameter` copied a `Float64` weight column on every call (O(n)); it is only read, so it is now used as is.

### Removed

- **The engine cache and its controls** (`ENGINE_CACHE`, `get_or_build_engine`, `get_cache_stats`, `VALIDATED_MODELS_CACHE`; the unreleased `set_engine_cache_capacity!`/`engine_cache_capacity`): see "Engine cache removed" for the migration. `clear_engine_cache!()` remains.
- **Duplicate `contrast_gradient!`/`contrast_gradient`.** `src/inference/contrast_gradient.jl` was a verbatim copy of FormulaCompiler's exported functions (the file said "migrated from FormulaCompiler"); it is deleted and the one live call site (`src/inference/delta_method.jl`) uses FormulaCompiler's. Margins never exported these names.
- **Duplicate `validate_mixture_weights`/`validate_mixture_levels`.** FormulaCompiler now exports functions of the same names (same code and messages, used by the `CategoricalMixture` constructor), so Margins' own exported copies made `using FormulaCompiler, Margins` ambiguous. Margins' copies are deleted and FormulaCompiler's are re-exported; `create_balanced_mixture(levels)` delegates to `FormulaCompiler.create_balanced_mixture` and keeps its `Dict{String, Float64}` return value and errors.

### Documentation

- **Thread safety.** `docs/src/caching.md` recommended pre-populating the engine cache before parallel computation, and `docs/src/computational_architecture.md` said buffer reuse "maintain[s] thread safety". Neither is true: a cached `MarginsEngine` (and its FormulaCompiler evaluators) owns mutable scratch buffers that every call writes, so concurrent calls on a shared engine overwrite each other — pre-populating the cache makes them share one engine, the unsafe case (64 of 64 concurrent calls returned wrong values in testing). The docs now state the thread-safety guarantee of "Concurrent Margins calls are safe" (Features). `docs/src/population_scenarios.md` no longer claims scenarios are "evaluated in parallel when possible" (they are evaluated sequentially).
- Elasticity documentation rewritten for the unit-level definition, with the dyex/eydx label swap fixed (see "Breaking / behaviour change").
- `docs/src/caching.md` rewritten for the cache-free design: what FormulaCompiler caches and what each call rebuilds (engine, data columns, validation, Σ), why the engine cache was removed, memory, and the thread-safety guarantee (replacing "Margins is not thread-safe" and its process-only parallelism advice); removed the "bootstrap 1000 iterations" row and the stale `get_or_build_engine` sections. `docs/src/computational_architecture.md`, `docs/src/population_scenarios.md` and `ARCHITECTURE.md`: caching and thread-safety notes updated.
- `docs/src/second_differences.md` rewritten around the model-level `second_differences(model, data, …)`; `api.md`, `index.md`, `advanced.md`, `mathematical_foundation.md` and `ARCHITECTURE.md` updated.

### Tests

- `test/performance/test_ame_fd_loop_zero_alloc.jl` called a removed `marginal_effects_eta!(…; backend=:fd)` signature and a removed 5-argument `get_or_build_engine`, and was not run. Rewritten to test the current FD AME loop (`_compute_continuous_marginal_effects!` → `_compute_me_loop_linear!`/`_compute_me_loop_response!` with an `FDEvaluator`): allocation identical at 1,000 and 20,000 rows and ≤ 64 bytes, both scales, weighted and unweighted. It is not redundant: the other FD allocation tests cover one row of `marginal_effects_eta!/mu!` or the full `population_margins` call, not the accumulation loop.
- `test/performance/test_population_aap_loop_zero_alloc.jl` used the same removed `get_or_build_engine` signature and was not run. Rewritten: the per-row prediction loop is allocation-free, and warm `population_margins(...; type=:predictions)` calls on Float64 data are O(1) in rows (n = 1,000 vs 50,000), unweighted and weighted; with an `Int` column its per-call `Float64` copy is the only O(n) allocation. It was the only allocation test of the predictions path.
- Both are now in `test/runtests.jl`. `test/performance/benchmark_blas_opt.jl` is a manual benchmark script using only the public API and is left as is.

### Internal (R6/R7 refactor)

No user-visible change; estimates and SEs agree with the pre-refactor golden snapshot (`test/support/golden/margins_golden.csv`) to relative difference ≤ 1e-12 (AD and backend-free rows) and ≤ 1e-9 (finite-difference rows).

- **Prediction kernel** (`src/computation/predictions.jl`). The scale is a type, `PredScale` (`LinkScale()` / `ResponseScale(link)`, from `predscale(scale, link)`, resolved once before a function barrier); `yhat`/`yhat2` return ŷ and its first (and second) derivative in η, the latter from FormulaCompiler's `link_derivatives`. `predict_row!`, `average_prediction!` and `predict_rows!` serve every prediction path. `compute_linear_predictor` (unused) and `src/computation/marginal_effects.jl` are deleted. Tests: `test/primitives/test_prediction_kernel.jl`.
- **Predictions use the kernel.** The population AAP (`_population_predictions`), the AAP within `scenarios`/`groups` contexts (`_compute_population_prediction_in_context`) and the profile APM (`_profile_predictions`) call `average_prediction!` / `predict_rows!`, replacing eight hand-copied scale × weights loop bodies and `compute_predictions_batch!`. The dead per-row stores into the engine's `η_buf`, and the `η_buf` field itself (an n-length buffer per engine), are removed, which cuts about 8n bytes from every engine build (0.4 MB at n = 50,000).
- **One categorical kernel, batched over contrast pairs** (`src/kernels/categorical.jl`). `categorical_contrast_ame!` takes the scale as a `PredScale` (replacing the `link`/`nothing` convention and the scale branches at its three call sites) and has one loop for weighted and unweighted rows (`rowweight`). The new `categorical_contrast_ame_batch!` computes all of a variable's contrasts in one pass over the rows, evaluating X₀ (and ŷ, g′ at X₀) once per row for each run of pairs sharing their "from" level — once per row for baseline contrasts — with each pair's arithmetic unchanged, so results are bitwise identical; the population and `scenarios`/`groups` paths use it. The unused `gradient_buf` argument, the profile `η_baseline_buf`/`gradient_buf`, and the engine's `contrast_grad_buf`/`contrast_grad_accum` fields are removed.
- **One continuous-effects accumulator** (`src/computation/continuous_accumulator.jl`). `accumulate_continuous!` handles every scale (`PredScale`) × measure (`Measure`: `Effect`, `Eyex`, `Dyex`, `Eydx`) × weights × backend combination with one row loop: per row it calls FormulaCompiler's `derivative_modelrow!(J, x, de, row)` (Jacobian and model row together, free for AD) and `link_derivatives`, and accumulates ∇β qⱼ = (mⱼd₁)Jᵢ[:,j] + (mⱼd₂aⱼ − δqⱼd₁/ŷ)xᵢ column-wise (the xᵢ term is skipped on the link scale for measures that do not divide by ŷ). It replaces the four population loops (`_compute_me_loop_linear!`, the generic and AD `_compute_me_loop_response!` with `_fused_me_rows!`, and `_accumulate_continuous_measures!`), which materialized the per-row Gβ matrix, made strided writes, and evaluated the model row up to three times per row for AD response-scale measures. The engine's `g_all_buf`, `Gβ_all_buf`, `deta_dx_buf` and `batch_var_indices` are replaced by one `MEWorkspace`. Population AMEs are 15–37% faster at n = 50,000 (e.g. elasticities 0.16 → 0.10 s, FD 0.46 → 0.35 s on the benchmark model).
- **Contexts call the accumulator directly.** `_compute_all_continuous_context_effects_batch` (`scenarios`/`groups`) builds an `MEWorkspace` for its per-context evaluator and calls `accumulate_continuous!`; the wrappers `_batch_compute_continuous_effects!` and `_batch_compute_continuous_measures!` are deleted.
- **Profile continuous effects use the accumulator.** At each grid row, `_process_profile_continuous_variables!` computes all requested continuous variables in one call (`rows = p:p`, weight 1), instead of re-evaluating every continuous variable's Jacobian once per requested variable; the reference-grid derivative evaluator differentiates only the requested variables (the Jacobian entries are unchanged). The profile measures now share the population code, quotient-rule term and zero-ŷ guard included. `apply_measure_transformation` and `src/engine/measures.jl` are deleted.
- **Primitives and dead buffers.** The row-level `marginal_effects_eta!`/`marginal_effects_mu!` (not exported) are kept as internal test oracles; their unused 3-D tensor batch forms, the allocating 4-argument FD `marginal_effects_eta!(g, de, β, row)` and a self-recursive generic `marginal_effects_eta!` wrapper are deleted. The never-used `EffectsBuffers` (`src/population/effects_buffers.jl`) and the engine's `effects_buffers` field are deleted, with four unreferenced `test/support/` scripts that tested removed buffers (one called a nonexistent `population_margins_raw!`). The engine-buffer descriptions in the docs are updated.
- `test/performance/test_ame_fd_loop_zero_alloc.jl` drives `_compute_continuous_effects!` for `:effect` and all three measures, on both scales, unweighted and weighted, with the FD and AD backends, and asserts 0 bytes per call. (Earlier in this cycle it tested FD only, allowing ≤ 128 bytes for `:effect` and ≤ 512 for a measure, because of the dynamic dispatch through the abstractly typed `engine.de`.)
- **Categorical kernel uses FormulaCompiler's public API.** `categorical_contrast_ame_batch!` evaluates each contrast endpoint with FormulaCompiler 1.5's exported `level_modelrow!` instead of its unexported counterfactual helpers (results bitwise unchanged).

## v2.2.0 — 2026-10

### Dependencies
- Requires FormulaCompiler ≥ 1.4.2 (compat tightened from `"1.4"`), which fixes contrast-row misalignment for categorical columns with unused or reordered pool levels.

### Features
- **Prediction intervals for profile predictions.** `profile_margins(...; type=:predictions, interval=:prediction)` widens the interval for a single new observation, `ŷ ± z·√(SE² + σ̂²)` with `σ̂² = deviance / dof_residual`, matching R's `predict(lm, interval="prediction")`. The default is `interval=:confidence`. Only Gaussian/identity-link models are supported; GLMs, mixed models, saturated models and `type=:effects` raise an `ArgumentError`. σ̂² is the pooled homoskedastic residual variance even when a robust `vcov` is supplied. Tests: `test/features/test_prediction_intervals.jl`.
- **Effects within bins of the variable being differentiated.** `population_margins(model, data; type=:effects, vars=[:income], groups=(:income, 4))` now returns the average marginal effect of income within each quartile of income — the direct translation of Stata's `margins, dydx(income) over(incomebin)`. It was previously rejected by the vars/groups conflict rule, which asked only whether a name appeared on both sides. The rule now turns on what the specification *does*: one that **pins** a variable to a single value blocks its own effect, one that **bins** it into ranges does not. `scenarios` always pins (it overrides the column for every row) and grouping by a categorical pins (each subgroup holds one level), so both keep being rejected; quantile and threshold grouping only select rows, leaving the variable varying inside each bin. No computation changed — `_compute_all_continuous_context_effects_batch` already averaged over the context's rows, reported `n` as the subgroup size, and took x̄/ȳ/∂ȳ/∂β over those rows for `measure=`. Two places had to stop asking `haskey`: the skip rule in `_population_margins_with_contexts` now consults `_group_spec_pins`, and `_validate_vars_groups_overlap` now consults `_classify_group_variables`, a purely syntactic walker (possible because a bare `Symbol` is accepted only for a categorical variable, so it always pins). Also rejected now: a variable binned by `groups` and simultaneously overridden by `scenarios`, which built bins from observed values the override then discarded, labelling each subgroup with a range its rows no longer occupied. Correctness tests in `test/validation/test_population_scenarios_groups.jl` check each bin's estimate against a manual average over exactly that bin's rows on a logit, with delta-method SEs verified by central-differencing with respect to β, across both scales, threshold bins, weights, and agreement with the pre-cut derived-column spelling.
- **Profile categorical contrasts are reachable again.** The row-specific baseline contrast behaviour documented in CLAUDE.md could not be requested: the filter in `_mem_continuous_and_categorical` (`src/profile/core.jl`) dropped a categorical variable if it was a mixture *or* if it appeared in the reference grid, and grid completion fills an absent categorical with a mixture — so naming the variable in the grid made it "scenario-defining" and leaving it out made it a mixture. Every route returned 0 rows with an `@info`, `_compute_row_specific_baseline_contrast` and `_compute_profile_pairwise_contrasts` were dead code, and `test/test_profile_pairwise.jl` was excluded from `test/runtests.jl` because it would fail. A categorical named in `vars` is now contrasted at the grid rows: **pinned** to a discrete level by the grid gives one contrast per row (that row's level against the baseline, baseline rows contributing none); **not pinned** gives the full level enumeration at each row (`K-1`, or `K(K-1)/2` for `contrasts=:pairwise`). Auto-detected `vars` is unchanged — grid categoricals stay scenario-defining and completed mixtures stay backdrop, as `test/regression/test_bool_mixture_skip.jl` requires. `test/test_profile_pairwise.jl` is re-enabled.

### Bug Fixes
- **Categorical variables with unused levels gave wrong contrasts.** A `CategoricalArray` whose pool held zero-count levels produced a spurious zero contrast for the unused level and truncated a real one, because `generate_contrast_pairs` (`src/computation/scenarios.jl`) enumerated pool levels while the results table was sized from observed levels. Unused levels are now excluded. Levels present in the full data but absent from a subgroup are kept. Requires FormulaCompiler ≥ 1.4.2, which fixes the matching contrast-row misalignment in model evaluation. Tests: `test/regression/test_unused_categorical_levels.jl`.
- **Continuous `groups` binning returned silently wrong numbers, five ways.** None of these raised, warned, or were covered by a test; all are fixed together in `src/population/contexts.jl`, with regression tests in `test/regression/test_group_binning.jl` asserting one invariant throughout — the bins partition the sample exactly, so the reported `n` values sum to `nrow(data)`.
  - **Bins did not partition the sample.** `_create_context_data_with_weights` selected `[lower, upper)` for bin 1 but `(lower, upper]` for every other bin, so a row sitting exactly on an interior boundary belonged to no bin at all. On ordinary skewed data — `groups=(:income, 4)` where 30% of incomes are zero — 167 of 500 rows vanished and Q1 was reported as `0.0 ± 0.0`. All bins are now half-open `[lower, upper)`, with the last closed so the maximum is included.
  - **Threshold labels contradicted the filtering.** `_create_threshold_groups` labels a bin `"[25000, 50000)"` while the filter selected `(25000, 50000]`. With income mass sitting exactly on the cutpoints, rows at the first threshold were dropped outright and rows at later thresholds landed one bin too low: a 200-row frame reported 160 rows, and `">= 50000"` held 40 where it should have held 80. The single half-open convention makes every label literally true.
  - **Tied quantile boundaries produced empty bins.** When enough values tie that two quantile boundaries coincide, the bin between them has no rows and was reported as `0.0 ± 0.0` — readable as a precisely estimated null. `_create_quantile_groups` now raises a `MarginsError` naming the coinciding boundaries and suggesting fewer bins, explicit thresholds, or a categorical column.
  - **A column named `<var>_bin` corrupted the result.** The group spec carried a companion `Symbol("$(var)_bin")` key. If the data happened to contain a real column of that name, it was matched against it (`findall(==(1), data[:x_bin])` against a `CategoricalVector{String}` matches nothing), every context came back empty, and all estimates were `0.0 ± 0.0`. `docs/src/stata_migration.md` and `docs/src/examples.md` instructed users to create exactly that column as a workaround, so following the documentation triggered the bug. The companion key is removed; `bin` is already a field of the spec NamedTuple.
  - **A bare numeric `groups=:x` silently auto-quartiled.** `_parse_over_specification` binned any numeric grouping column into quartiles without saying so. On an integer count this returned quartiles rather than levels — `groups=:nkids` on a 0–3 count produced two empty bins and lost 78 of 300 rows. A bare `Symbol` is now accepted only for a categorical variable; a numeric one raises a `MarginsError` directing to `(:x, 4)`, explicit thresholds, or a categorical column. This matches Stata's `over()` and R marginaleffects' `by()`, neither of which auto-bins. No test or documented example used the bare numeric form.
- **Empty subgroups were reported as `0.0 ± 0.0`.** A context with no rows produced an estimate of exactly zero with a zero standard error, which reads as a precisely estimated null rather than "no data". This was routine for cross-tabulations: `groups=[:region, :education]` emitted a row for every level combination including ones never observed. Empty contexts are now skipped, so results contain only the subgroups that exist; if every context is empty the pre-existing `MarginsError` guard reports that.

- **Profile categorical contrasts were computed incorrectly.** The dead implementation would have returned wrong numbers had it been reachable: the response-scale branch of `_compute_row_specific_baseline_contrast` reduced the design row with `sum(X)` instead of `dot(X, β)`, and took its reference point from the grid row's own level rather than the contrast's `from` level. Rather than repair it, the profile path now calls the population kernel `categorical_contrast_ame!` over the single grid row, so both scales, the gradient and the SE come from one verified implementation (`src/profile/categorical_effects.jl` deleted, ~280 lines).
- **Profile results mapped grid rows by division.** `_extract_and_filter_profile_values` expanded grid columns with `repeat(col, inner = result_length ÷ n_profiles)`, which assumes every profile contributes the same number of rows. Contrast rows break that assumption (a row pinned to the baseline contributes none), which surfaced as `ArgumentError: New columns must have the same length as old columns` from the display layer. Each result row now records the grid row it was evaluated at.
- **Grouping by the variable being contrasted crashed.** `population_margins(m, df; type=:effects, vars=[:g], groups=:g)` died with `ArgumentError: column name "estimate" not found in the data frame since it has no columns`, on `lm` and `glm` alike and on both scales. The cause was not contrast generation but the skip rule at `src/population/contexts.jl:52`, which drops any `var` present in the current `scenario_spec` or `group_spec`: with every requested variable consumed, `results` stayed a zero-column `DataFrame()` and `results.estimate` threw from DataFrames internals. The investigation found one documented rule with three behaviors — `vars` ∩ `scenarios` threw a teaching `ArgumentError` (`_validate_vars_scenarios_overlap`), `vars` ∩ `groups` crashed, and five documentation files claimed the variable was "silently" skipped. `groups` now mirrors `scenarios`: a new `_validate_vars_groups_overlap` (`src/population/core.jl`) rejects any overlap between an explicit `vars` and the grouping variables, reusing `_extract_group_variables` so every `groups` syntax is covered (`Symbol`, `Vector`, `(Symbol, spec)` binning, nested `outer => inner`). Auto-detected `vars` keep being skipped silently — erroring there would fail a reasonable call over a variable the user never named — but `_population_margins_with_contexts` now guards the empty frame and raises a `MarginsError` naming the consumed variables instead of returning a zero-row result. The suite missed all of this because every `vars=` + `groups=` pairing in `test/` used disjoint variables, and the `scenarios` validator had no test at all.

- **`measure` was silently ignored under `scenarios` / `groups`.** `population_margins(...; measure=:elasticity)` (and `:semielasticity_dyex` / `:semielasticity_eydx`) returned raw average marginal effects labeled `AME (elasticity)` as soon as a context was supplied — `_population_margins_with_contexts` threaded `measure` into the result metadata only, never into the computation. Fixed in `src/population/contexts.jl`: `_compute_all_continuous_context_effects_batch` now applies `apply_measure_transformations` and the `_gradient_of_mean_response` quotient-rule correction, using the context's own x̄, ȳ and ∂ȳ/∂β (computed over the context's rows with scenario overrides applied). Example: for a logit with `groups=:g`, one group previously reported 0.218537 (the raw AME) where the elasticity is 0.513208.
- **Weighted `groups` threw `BoundsError`.** `_create_context_data_with_weights` returns a per-context weight *subset*, but the continuous-effects and predictions paths index weights by absolute row (`weights[row]` for `row in context_indices`), so `population_margins(...; weights=:w, groups=:g)` crashed for both `type=:effects` and `type=:predictions` — a combination `docs/src/weights.md` documents. Both paths now receive the full-length weights vector, matching the categorical path; the convention is documented at each function. Also removed the dead `_compute_population_effect_in_context_clean` and `_compute_continuous_context_effect_batch`, which encoded the opposite convention.
- **Elasticity measures no longer mislabel discrete contrasts.** `measure=:elasticity` with a categorical or boolean variable produced the raw level contrast under an `AME (elasticity)` label, on every path. Elasticities scale a derivative ∂y/∂x by x̄ and/or ȳ, which a discrete contrast does not have (Stata rejects factor variables in `eyex()`/`dyex()`/`eydx()`; R marginaleffects transforms numeric predictors only). Naming a discrete variable in `vars` alongside a non-`:effect` measure is now an `ArgumentError`; auto-detected discrete variables are skipped with an `@info`, so `population_margins(model, data; measure=:elasticity)` keeps working and returns only the rows it can define. Applies to `population_margins` and `profile_margins` (`_filter_vars_for_measure` in `src/core/validation.jl`).
- **The engine cache returned results computed on the wrong data.** `ENGINE_CACHE` (`src/engine/caching.jl`) keys entries on the data's column *names*, but a `MarginsEngine` is bound to the columns it was built from — its derivative and contrast evaluators store those arrays and dereference them at call time. Two datasets sharing a schema and a fitted model therefore collided on one entry, and the second caller silently received the first caller's rows (or a `BoundsError`, when the second dataset was longer). This broke any workflow reusing one model across data slices: hand-rolled subgroup analysis, train/test splits, resampled frames. Retrieval now revalidates a cache hit against the caller's columns by identity (`_engine_data_matches`) and rebuilds when they do not match; `build_engine` costs 0.03-0.13 ms, or 1-3% of a single margins call. Ordinary `Float64`/`Bool`/categorical data still hits the cache, since `_convert_numeric_to_float64` returns those columns unchanged.

New regression tests in `test/validation/test_measure_contexts.jl` validate every measure × scale × weighting × context combination against closed-form oracles built from `modelmatrix`, `coef` and `vcov`, with delta-method SEs checked by central-differencing the estimate with respect to β. `test/validation/test_engine_cache.jl` covers the cache fix the same way — both call orders of the reproducer, a subgroup loop in one session, predictions, profile margins, categorical contrasts on a slice, `Int`-column data (which revalidates to a miss every call), and cache-size bounds.

### Bug Fixes (link scale)
- **`scale=:link` was silently ignored for categorical contrasts.** `population_margins(...; vars=[<categorical>], scale=:link)` returned response-scale contrasts (both estimate and standard error) under a link-scale label. The discrete-contrast kernels select their scale from the `link` argument (`nothing` ⇒ link scale), but the callers passed `engine.link` unconditionally. Fixed on the plain path (`src/population/categorical_effects.jl`) and the `scenarios`/`groups` path (`src/population/contexts.jl`); affects `contrasts=:baseline` and `:pairwise`, weighted and unweighted. Link-scale baseline contrasts now equal the corresponding coefficients exactly, as they must.
- **Elasticity standard errors were wrong on the link scale.** `_gradient_of_mean_response` branched on `scale === :linear`, a value the API never produces, so `scale=:link` fell through to the response branch and weighted ∂ȳ/∂β by `mueta(link, η)` instead of using X̄. Point estimates were unaffected; SEs for `measure=:elasticity` and `:semielasticity_eydx` were off (e.g. 0.279631 reported vs 0.269922 correct on a logit test case). Fixed in `src/population/continuous_effects.jl`.
- **Profile categorical contrast gradients ignored `scale`.** The estimate honored `scale=:link` but the gradient was always computed on the response scale, pairing a link-scale estimate with a response-scale SE (`src/profile/categorical_effects.jl`).

All three were invisible to the test suite because every existing test pairing `scale=:link` with a categorical `vars=` used an identity-link linear model, where the two scales coincide. New regression tests in `test/validation/test_link_scale.jl` use logit and probit models.

### Removed
- **Two files of dead buffer/context scaffolding, 303 lines.** `src/profile/continuous_effects.jl` defined `ProfileContext`, `create_profile_context` and `apply_profile_measure_transformations`; nothing referenced them, and `src/Margins.jl` never contained an `include` for the file in any commit, so it was not merely unreachable but never parsed. `src/core/buffer_management.jl` was included, but all five of its symbols — `ComputationBuffers`, `get_computation_buffers`, `prepare_computation_buffers!`, `ProfileBuffers`, `get_profile_buffers` — were referenced only inside it; the last two are the companions the profile file's own comments pointed at, and the file already recorded `CategoricalBuffers`/`get_categorical_buffers` as removed earlier, so this finishes that cleanup. The profile helpers were deleted rather than revived because `apply_profile_measure_transformations` rescaled the gradient by a single `transform_factor`, omitting the quotient-rule term `-(ε/μ)·∂μ/∂β` that the live path applies (`_process_profile_continuous_variable!`, `src/profile/core.jl`), so it would have reported wrong elasticity and `semielasticity_eydx` standard errors; it also read the variable's value from the profile `Dict` rather than the completed reference grid, and hardcoded grid row 1, so it could not handle multi-row grids. All computation uses the engine's own pre-allocated buffers. No behavior change: test counts are identical.

### Documentation
- Normalized `scale` docstrings that named non-existent values (`:linear`, `:eta`) to the actual API values `:link` and `:response`.
- Documented continuous binning semantics in `docs/src/grouping.md`: bin membership is half-open `[lower, upper)` with the last bin closed, so labels read literally and the bins partition the sample; a numeric grouping variable must be binned explicitly; tied quantile boundaries are an error; and empty subgroups are omitted from results.
- Corrected the "skip rule" documentation, which described behavior the code never had. `docs/src/grouping.md` (the canonical section, now "Conflict Rule for Statistical Validity"), `docs/src/population_scenarios.md`, `docs/src/stata_migration.md`, `docs/src/examples.md`, `ARCHITECTURE.md` and `CLAUDE.md` all stated that a variable named in both `vars` and `scenarios`/`groups` was silently dropped; `scenarios` has thrown an `ArgumentError` for some time and `groups` now does too. The one place silent skipping survives — auto-detected `vars` — is documented explicitly, along with the `MarginsError` raised when it consumes every detected variable.

## v2.1.4 — 2026-06

### Bug Fixes
- **Support CategoricalArrays 1.0/1.1** (fixes #16); compat widened to `"0.10, 1"`. CategoricalArrays 1.0 changed `convert(Vector, ::CategoricalVector)` to return a plain `Array` of unwrapped values. `complete_reference_grid` stored grid columns in a `Dict{Symbol, Vector}`, so a categorical column lost its pool and non-baseline levels collapsed to the baseline: `profile_margins` silently returned baseline estimates and standard errors for every non-baseline level in the reference grid. Fixed by using `Dict{Symbol, AbstractVector}`.

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
