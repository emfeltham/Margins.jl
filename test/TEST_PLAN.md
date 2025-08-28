# Margins.jl Test Plan

This plan specifies a comprehensive, correctness‑focused test suite for Margins.jl, modeled after the spirit of FormulaCompiler’s correctness tests (e.g., `test/test_models.jl`). The priority is to verify numerical correctness and invariants across a wide range of models, links, data shapes, and API options.

Allocation/performance checks are secondary and lightweight.

## Goals and Principles

- Correctness first: validate values against reference computations and invariants.
- Backed by FormulaCompiler primitives and GLM predictions where applicable.
- Cover typical and edge cases for continuous and categorical variables, links, profiles, grouping, and weights.
- Deterministic: fixed PRNG seeds; small/medium datasets.
- Guard optional dependencies (e.g., CovarianceMatrices) to keep CI robust.

## Test Matrix Overview

- Models: OLS (Identity), GLM (Logit/Probit/Log/Cloglog/Sqrt/etc.).
- Variables: continuous, categorical (including Bool), interactions, transformations (log, ^2).
- Targets and scales: effects on `:eta` and `:mu`; predictions on `scale=:response|:link`.
- Effects: AME, MEM, MER; Predictions: APE, APM, APR.
- Contrasts: baseline and pairwise; μ via chain rule.
- Grouping: `over`, `within`, and `by` stratification.
- Profiles: `at=:means`, Dict with `:all` and numlists (e.g., "10(5)30"), multiple at blocks; `average_profiles` flag.
- Weights: user weights; `asbalanced=true`; `asbalanced=[:subset...]`.
- Covariance: `vcov=:model`; `vcov=Σ` (matrix); `vcov = m->Σ` (function); estimators (HC1/Clustered/HAC) if CovarianceMatrices available.
- MixedModels: fixed‑effects only (LMM/GLMM) validation.

## Core Correctness Checks

1) AME (η and μ)
- Compare AME values against per‑row marginal effects averaged explicitly (build with `marginal_effects_*` in a loop).
- For μ, also verify chain rule consistency: `dμ/dx = (dμ/dη)*(dη/dx)`.
- Check that `vcov` choice only changes SEs, not point estimates.

2) MEM (at=:means) and MER (at=Dict)
- Verify MEM equals MER with a single `at` profile set to means.
- For MER, check each profile value equals direct evaluation at those covariate settings.

3) APE/APM/APR (predictions)
- APE: compare to averaging GLM predictions across rows on response/link scales.
- APR/APM: per‑profile predictions equal predictions at fixed `at` values; optional `average_profiles=true` equals simple average of per‑profile values.

4) Categorical contrasts (η and μ)
- For η: `Δ = X(to) − X(from)`; effect = `β'Δ`; gradient = Δ.
- For μ: `μ_to − μ_from`; gradient `= g'(η_to)·X_to − g'(η_from)·X_from`.
- Validate against manual contrast computations and small finite differences on η.

5) Links
- Identity, Logit, Probit, Log, Cloglog, Sqrt (and any others present in GLM).
- Cross‑validate predictions against GLM.predict for response/link scales.

6) Interactions and transforms
- Include terms like `x*z`, `x^2`, `log(abs(x)+c)`, and factor interactions to ensure derivatives propagate correctly.

7) Grouping and stratification
- `over` and `within`: compute effects/predictions by groups and nested designs; verify row partitioning and group labels in output.
- `by`: split computation by strata; verify concatenation and by columns.

8) Profiles (at) semantics
- Dict with `:all` summary and specific overrides (general→specific precedence) produces expected grids.
- Multiple at blocks (Vector{Dict}) concatenate profile sets.
- `average_profiles=true` collapses to a single summary equal to the average of per‑profile values.

9) Weights and asbalanced
- User weights: weighted AME/APE equal manual weighted averages of per‑row values.
- `asbalanced=true`: averages neutralize factor imbalance while preserving continuous distribution; compare against manual reweighting.
- `asbalanced=[:subset]`: balance only specified factor variables; verify against manual two‑way balancing for small examples.

10) Covariance (vcov kw)
- `vcov=:model` vs matrix vs function overrides: SEs change appropriately; point estimates unchanged.
- If CovarianceMatrices.jl available: estimators (HC1, Clustered, HAC) produce valid Σ and change SEs accordingly. Guard with feature detection; skip when not installed.

11) MixedModels (fixed effects)
- LMM/GLMM minimal examples: extract fixed effects via FormulaCompiler; compute AME/MER/APR on fixed components only; ensure sanity of outputs.

## Invariants and Metadata

- Result table schema: contains `term`, `dydx`, `se`, `ci_lo`, `ci_hi`, `z`/`p`, and group/`at_*` columns as appropriate.
- Metadata: includes `mode`, `dydx`, `target`, `at`, `backend`, `vcov`, `n`, `link`, `dof`. Printer shows compact summary.
- Error handling: invalid var names, empty profiles, or incompatible vcov specs throw clear errors.

## Numerical Tolerances

- Effects (η/μ), predictions, and contrasts: `rtol=1e-6, atol=1e-8` where double rounding is involved.
- Cross‑validation AD vs FD derivatives for small problems; allow slightly looser tolerances for μ if links are highly curved.

## Allocation/Performance (lightweight)

- Confirm FD AME path remains 0 allocations after warmup on a small dataset (skip weighted path for now).
- Quick timing to ensure no pathological slowdowns; avoid strict time thresholds.

## CI Strategy

- Base GLM tests always on; guard robust/cluster/HAC with `Base.find_package("CovarianceMatrices")`.
- Use fixed seeds for synthetic data.

## File Organization Proposal

- `test_glm_basic.jl`: AME/MEM/MER/AP* across links and simple terms.
- `test_profiles.jl`: `at` semantics, average_profiles, multi‑block, `:all` precedence.
- `test_grouping.jl`: `over`, `within`, `by` behavior and labels.
- `test_weights_asbalanced.jl`: user weights and asbalanced variants (all/subset), plus invariants.
- `test_contrasts.jl`: baseline/pairwise contrasts (η/μ) and gradients.
- `test_vcov.jl`: `vcov` as :model/matrix/function; CovarianceMatrices estimators (guarded).
- `test_mixedmodels.jl`: fixed‑effects only demonstration.
- `test_errors.jl`: invalid inputs and messages.
- `test_allocations.jl`: tiny allocation checks for FD AME path (optional).

## References

- GLM.jl and StatsModels.jl predict/predict! scale semantics.
- FormulaCompiler.jl derivative and variance primitives.
- CovarianceMatrices.jl estimators and usage (when available).
