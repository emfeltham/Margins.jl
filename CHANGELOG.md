# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog. Unreleased changes go under "Unreleased" until a version is tagged.

## Unreleased

## v2.2.0 — 2026-10

### Dependencies
- Requires FormulaCompiler ≥ 1.5 (compat `"1.5"`; previously `"1.4"`, then `"1.4.2"` during this cycle). 1.4.2 fixed contrast-row misalignment for categorical columns with unused or reordered pool levels; 1.5 resolves categorical codes through the model's own levels (a `LevelIndex` type parameter on `ContrastOp`), so data pools that are reordered, rebuilt or smaller than the fitting data's evaluate correctly, and missing/unknown categorical values throw instead of silently mapping to the reference level. *(Exact FormulaCompiler version to be confirmed at release.)*
- MixedModels.jl is now a weak dependency (compat `"4.35"`), loaded through the new `MixedModelsExt` extension. FormulaCompiler already depends on it, so nothing extra is installed.

### Breaking / behaviour change
- **Continuous-effect contrast label is `"dy/dx"` everywhere.** The `scenarios`/`groups` path (`_compute_all_continuous_context_effects_batch`) and `store_continuous_result!` labelled continuous effects `"derivative"`, while the plain population path and profile margins used `"dy/dx"`. All now use `"dy/dx"`. Code that filters results with `contrast == "derivative"` must change. The second-difference functions' `contrast` keyword now defaults to `"dy/dx"` and still accepts `"derivative"` as an alias.
- **Population elasticities are now means of unit-level elasticities (matching the docs, Stata and R).** `population_margins(...; measure=:elasticity | :semielasticity_dyex | :semielasticity_eydx)` used to compute a *ratio of means*: `(x̄/ȳ)·AME`, `x̄·AME` and `AME/ȳ`, with x̄ and ȳ the (weighted) sample means and a quotient-rule correction `−(ε/ȳ)·∂ȳ/∂β` for the SE (`apply_measure_transformations` and `_gradient_of_mean_response` in `src/population/continuous_effects.jl`). Margins' own documentation (`docs/src/advanced.md`, `docs/src/mathematical_foundation.md`), Stata's `margins, eyex()/dyex()/eydx()` and R's `marginaleffects::avg_slopes(slope = "eyex"|"dyex"|"eydx")` all define the population measure as the *mean of the unit-level quantity*, with gᵢ = ∂ŷᵢ/∂x and ŷᵢ the prediction on the requested `scale`: eyex = mean(gᵢxᵢ/ŷᵢ), dyex = mean(gᵢxᵢ), eydx = mean(gᵢ/ŷᵢ) — weighted means under `weights`, taken over each context's rows under `groups`/`scenarios`. The two agree only when the unit quantities are constant (e.g. eyex and eydx for a log-link model, or dyex with a constant gᵢ). Margins now computes the unit-level definition. The delta-method gradient is the mean of the per-row gradients, e.g. ∇β(gᵢxᵢ/ŷᵢ) = (xᵢ/ŷᵢ)∇βgᵢ − (gᵢxᵢ/ŷᵢ²)∇βŷᵢ with ∇βŷᵢ = g⁻¹′(ηᵢ)Xᵢ (response scale) or Xᵢ (link scale), accumulated in a dedicated row loop (`_accumulate_continuous_measures!`) shared by the plain and the `scenarios`/`groups` paths, on both backends; `measure=:effect` does not go through it, so plain AMEs are computed exactly as before. A row with ŷᵢ = 0 makes eyex/eydx undefined: R silently returns `Inf`/`NaN` for that unit and propagates it into the average; Margins raises a `MarginsError` naming the row. Profile margins (`profile_margins`) are unchanged: at a single grid row the mean of ratios is the ratio, and they already used x and ŷ at the grid row.
  - Worked example (logit `yb ~ x * z + g`, n = 300, the deterministic dataset of `test/validation/test_elasticity_unit_definitions.jl`, response scale, estimate ± SE):

    | measure | v2.1 (ratio of means) | v2.2.0 | R marginaleffects 1.0.0 |
    |---|---|---|---|
    | `:elasticity` (eyex) | 1.092346 ± 0.104633 | 1.399523 ± 0.192544 | 1.399523 ± 0.192626 |
    | `:semielasticity_dyex` | 0.495197 ± 0.041409 | 0.527332 ± 0.045356 | 0.527332 ± 0.045367 |
    | `:semielasticity_eydx` | 0.622449 ± 0.059623 | 0.953084 ± 0.140625 | 0.953084 ± 0.140680 |

    (R SEs use a numerical Jacobian; Margins' SEs match a central-difference-in-β oracle to 1e-4.) For a Poisson/log model the eyex and eydx values are identical before and after, as they must be (gᵢ/ŷᵢ = β_x), while dyex changes (2.062070 → 2.225984, R 2.225984).
  - **Documentation label swap fixed.** `docs/src/advanced.md` described `:semielasticity_dyex` with the formula and interpretation of eydx (`(∂Y/∂X)×(1/Y)`, "% change in Y per unit change in X") and vice versa, as did the docstring of `_process_continuous_variables!` and `docs/src/mathematical_foundation.md` ("Semi-elasticity (X)/(Y)"); `docs/src/examples.md` used `:semielasticity_dyex` for a "% change in wages per unit change" example. The *computations* always matched the names (dyex = dy/d(ln x) = g·x, eydx = d(ln y)/dx = g/y, as in Stata); the docs now say so, with a Stata/R correspondence table.
  - Tests: `test/validation/test_elasticity_unit_definitions.jl` checks all three measures × {logit, Poisson, lm} × {response, link} × {unweighted, weighted} × {whole sample, `groups`} × {`:ad`, `:fd`} against a closed-form oracle (gᵢ from central differences of `linkinv.(X(x±h)β)`, SEs from central differences in β) and against 120 R marginaleffects 1.0.0 `avg_slopes()` values (`test/r_compare_marginaleffects/elasticity/`). Existing tests that encoded the ratio-of-means definition were updated to the unit-level one: the oracle in `test/validation/test_measure_contexts.jl` (`_measure_oracle`) and the link-scale gradient check in `test/validation/test_link_scale.jl` ("elasticity quotient-rule gradient uses the link scale").

### Features
- **Prediction intervals for profile predictions.** `profile_margins(...; type=:predictions, interval=:prediction)` widens the interval for a single new observation, `ŷ ± z·√(SE² + σ̂²)` with `σ̂² = deviance / dof_residual`, matching R's `predict(lm, interval="prediction")`. The default is `interval=:confidence`. Only Gaussian/identity-link models are supported; GLMs, mixed models, saturated models and `type=:effects` raise an `ArgumentError`. σ̂² is the pooled homoskedastic residual variance even when a robust `vcov` is supplied. Tests: `test/features/test_prediction_intervals.jl`.
- **Effects within bins of the variable being differentiated.** `population_margins(model, data; type=:effects, vars=[:income], groups=(:income, 4))` now returns the average marginal effect of income within each quartile of income — the direct translation of Stata's `margins, dydx(income) over(incomebin)`. It was previously rejected by the vars/groups conflict rule, which asked only whether a name appeared on both sides. The rule now turns on what the specification *does*: one that **pins** a variable to a single value blocks its own effect, one that **bins** it into ranges does not. `scenarios` always pins (it overrides the column for every row) and grouping by a categorical pins (each subgroup holds one level), so both keep being rejected; quantile and threshold grouping only select rows, leaving the variable varying inside each bin. No computation changed — `_compute_all_continuous_context_effects_batch` already averaged over the context's rows, reported `n` as the subgroup size, and took x̄/ȳ/∂ȳ/∂β over those rows for `measure=`. Two places had to stop asking `haskey`: the skip rule in `_population_margins_with_contexts` now consults `_group_spec_pins`, and `_validate_vars_groups_overlap` now consults `_classify_group_variables`, a purely syntactic walker (possible because a bare `Symbol` is accepted only for a categorical variable, so it always pins). Also rejected now: a variable binned by `groups` and simultaneously overridden by `scenarios`, which built bins from observed values the override then discarded, labelling each subgroup with a range its rows no longer occupied. Correctness tests in `test/validation/test_population_scenarios_groups.jl` check each bin's estimate against a manual average over exactly that bin's rows on a logit, with delta-method SEs verified by central-differencing with respect to β, across both scales, threshold bins, weights, and agreement with the pre-cut derived-column spelling.
- **Profile categorical contrasts are reachable again.** The row-specific baseline contrast behaviour documented in CLAUDE.md could not be requested: the filter in `_mem_continuous_and_categorical` (`src/profile/core.jl`) dropped a categorical variable if it was a mixture *or* if it appeared in the reference grid, and grid completion fills an absent categorical with a mixture — so naming the variable in the grid made it "scenario-defining" and leaving it out made it a mixture. Every route returned 0 rows with an `@info`, `_compute_row_specific_baseline_contrast` and `_compute_profile_pairwise_contrasts` were dead code, and `test/test_profile_pairwise.jl` was excluded from `test/runtests.jl` because it would fail. A categorical named in `vars` is now contrasted at the grid rows: **pinned** to a discrete level by the grid gives one contrast per row (that row's level against the baseline, baseline rows contributing none); **not pinned** gives the full level enumeration at each row (`K-1`, or `K(K-1)/2` for `contrasts=:pairwise`). Auto-detected `vars` is unchanged — grid categoricals stay scenario-defining and completed mixtures stay backdrop, as `test/regression/test_bool_mixture_skip.jl` requires. `test/test_profile_pairwise.jl` is re-enabled.

### Performance
Internal changes with identical estimates and standard errors (verified on the six benchmark operations — APM, MEM, AAP, AME (all), AME (one variable), AME (scenario) — at n = 5,000 and 50,000):
- **Profile engines no longer convert the whole dataset to dual numbers.** `build_engine` (`src/engine/core.jl`) binds a profile engine's derivative evaluator to a 1-row slice of the data. Profile margins build their own evaluator over the reference grid and only check that `engine.de` exists, but the AD backend converted every continuous column over all n rows to `Dual`s (~1 GB at n = 500,000).
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
- **Converted-column memo: allocation and clearing.** The memo that keeps converted data columns stable across calls (`_NARROWED_COLUMNS`, `src/core/data_conversion.jl`) checked a hit for staleness on abstractly typed values, allocating O(n) per call (28.8 MB per warm call at n = 200,000 with a `Union{Missing,Float64}` column and Int/String model categoricals, vs 0.007 MB for plain `CategoricalArray` data). The check now runs behind a function barrier (`_same_contents`), and such a warm call allocates about 0.01 MB. Dead entries (source column garbage-collected) are purged on every lookup, and `clear_engine_cache!()` now empties the memo too. Tests: `test/regression/test_data_prep_memo.jl`.
- **`Margins.VERSION` was stale** (`v"2.0.0"` while Project.toml said 2.2.0). It is now `pkgversion(@__MODULE__)`, read from Project.toml.

 A `CategoricalArray` whose pool held zero-count levels produced a spurious zero contrast for the unused level and truncated a real one, because `generate_contrast_pairs` (`src/computation/scenarios.jl`) enumerated pool levels while the results table was sized from observed levels. Unused levels are now excluded. Levels present in the full data but absent from a subgroup are kept. Requires FormulaCompiler ≥ 1.4.2, which fixes the matching contrast-row misalignment in model evaluation. Tests: `test/regression/test_unused_categorical_levels.jl`.
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

- **`measure` was silently ignored under `scenarios` / `groups`.** `population_margins(...; measure=:elasticity)` (and `:semielasticity_dyex` / `:semielasticity_eydx`) returned raw average marginal effects labeled `AME (elasticity)` as soon as a context was supplied — `_population_margins_with_contexts` threaded `measure` into the result metadata only, never into the computation. Fixed in `src/population/contexts.jl`: `_compute_all_continuous_context_effects_batch` now applies `apply_measure_transformations` and the `_gradient_of_mean_response` quotient-rule correction, using the context's own x̄, ȳ and ∂ȳ/∂β (computed over the context's rows with scenario overrides applied). Example: for a logit with `groups=:g`, one group previously reported 0.218537 (the raw AME) where the elasticity is 0.513208. (The ratio-of-means normalisation described here was itself replaced by the unit-level definition in this release; see "Breaking / behaviour change".)
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
- **Duplicate `contrast_gradient!`/`contrast_gradient`.** `src/inference/contrast_gradient.jl` was a verbatim copy of FormulaCompiler's exported functions (the file said "migrated from FormulaCompiler"); it is deleted and the one live call site (`src/inference/delta_method.jl`) uses FormulaCompiler's. Margins never exported these names.
- **Duplicate `validate_mixture_weights`/`validate_mixture_levels`.** FormulaCompiler now exports functions of the same names (same code and messages, used by the `CategoricalMixture` constructor), so Margins' own exported copies made `using FormulaCompiler, Margins` ambiguous. Margins' copies are deleted and FormulaCompiler's are re-exported; `create_balanced_mixture(levels)` delegates to `FormulaCompiler.create_balanced_mixture` and keeps its `Dict{String, Float64}` return value and errors.
- **Two files of dead buffer/context scaffolding, 303 lines.** `src/profile/continuous_effects.jl` defined `ProfileContext`, `create_profile_context` and `apply_profile_measure_transformations`; nothing referenced them, and `src/Margins.jl` never contained an `include` for the file in any commit, so it was not merely unreachable but never parsed. `src/core/buffer_management.jl` was included, but all five of its symbols — `ComputationBuffers`, `get_computation_buffers`, `prepare_computation_buffers!`, `ProfileBuffers`, `get_profile_buffers` — were referenced only inside it; the last two are the companions the profile file's own comments pointed at, and the file already recorded `CategoricalBuffers`/`get_categorical_buffers` as removed earlier, so this finishes that cleanup. The profile helpers were deleted rather than revived because `apply_profile_measure_transformations` rescaled the gradient by a single `transform_factor`, omitting the quotient-rule term `-(ε/μ)·∂μ/∂β` that the live path applies (`_process_profile_continuous_variable!`, `src/profile/core.jl`), so it would have reported wrong elasticity and `semielasticity_eydx` standard errors; it also read the variable's value from the profile `Dict` rather than the completed reference grid, and hardcoded grid row 1, so it could not handle multi-row grids. All computation uses the engine's own pre-allocated buffers. No behavior change: test counts are identical.

### Documentation
- **Thread safety.** `docs/src/caching.md` recommended pre-populating the engine cache before parallel computation, and `docs/src/computational_architecture.md` said buffer reuse "maintain[s] thread safety". Neither is true: a cached `MarginsEngine` (and its FormulaCompiler evaluators) owns mutable scratch buffers that every call writes, so concurrent calls on a shared engine overwrite each other — pre-populating the cache makes them share one engine, the unsafe case (64 of 64 concurrent calls returned wrong values in testing). The docs now state that Margins calls must not run concurrently in one process, and recommend separate processes (`Distributed.pmap`) or serial Margins calls. `docs/src/population_scenarios.md` no longer claims scenarios are "evaluated in parallel when possible" (they are evaluated sequentially).
- Elasticity documentation rewritten for the unit-level definition, with the dyex/eydx label swap fixed (see "Breaking / behaviour change").
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
