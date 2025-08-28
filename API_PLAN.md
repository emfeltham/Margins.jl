# Margins.jl API Plan (Finalized; remove `margins()`)

This document finalizes the public API for Margins.jl. We remove the legacy `margins(...)` entry point and make statistical names the canonical API. Optional general wrappers provide discoverability but are not required.

## Goals

- Clear, single‑purpose entry points (no `mode` switch).
- Canonical statistical names (AME/MEM/MER/APM/APR/APE).
- Keep results tidy and table‑friendly (DataFrame with a lightweight printer).
- Preserve performance guarantees by building on FormulaCompiler.

## Canonical API (public)

- Row‑averaged (observed data):
  - AME: `ame(model, data; vars=:continuous, target=:mu, vcov=:model, weights=nothing, balance=:none|:all|Vector{Symbol}, over=nothing, within=nothing, by=nothing)`
  - APE: `ape(model, data; scale=:response|:link, vcov=:model, weights=nothing, balance=:none|:all|Vector{Symbol}, over=nothing, within=nothing, by=nothing)`

- At profiles (explicit covariate settings):
  - MEM: `mem(model, data; vars=:continuous, target=:mu, vcov=:model, at=:means, over=nothing, by=nothing)`
  - MER: `mer(model, data; vars=:continuous, target=:mu, vcov=:model, at=Dict|Vector{Dict}, over=nothing, by=nothing)`
  - APM: `apm(model, data; scale=:response|:link, vcov=:model, at=:means, over=nothing, by=nothing)`
  - APR: `apr(model, data; scale=:response|:link, vcov=:model, at=Dict|Vector{Dict}, average=false, over=nothing, by=nothing)`

Optional discoverability wrappers:
- `effects(model, data; vars=:continuous, target=:mu, vcov=:model, weights, balance, over, within, by, at=:none)` → AME when `at=:none`; MEM/MER when `at≠:none`.
- `predictions(model, data; at=:none|:means|Dict|Vector{Dict}, scale=:response|:link, vcov=:model, average=false, weights, balance, over, within, by)` → APE when `at=:none`; APM/APR when `at≠:none`.

## Kwarg names (canonical)

- `vars`: variables for marginal effects (alias: `dydx`).
- `target`: `:mu|:eta` for effects (GLM response/link).
- `scale`: `:response|:link` for predictions (GLM predict scale).
- `vcov`: `:model | AbstractMatrix | Function | Estimator` (CovarianceMatrices-aware).
- `weights` and `balance`: `weights` is a user vector or column; `balance=:none|:all|Vector{Symbol}` balances factor distributions on row‑averaged paths.
- `at`: `:means | Dict | Vector{Dict}` (for profile paths). `:none` is only valid for AME/APE and the wrappers mapping to them.
- `average` (predictions only): collapse a profiles grid to one row.
- Grouping: `over`, `within`, `by` as in current implementation.

## Major Axis Separation (internal design)

All six functions are implemented atop two internal paths:
- Average over observed rows: “Average Marginal/Predicted Effects” (AME/APE). Handles `weights`, `balance`, `over/within/by`.
- Profiles at explicit settings: “Predictions/Effects at Means or Profiles” (APM/APR and MEM/MER). Handles `at=:means|Dict|Vector{Dict}` and optional `average`.

This split keeps hot loops allocation‑free for AME/APE and uses accurate AD paths or zero‑alloc FD for profiles as appropriate.

## Examples

```julia
# Effects (η): average over rows for continuous vars (AME)
res = ame(m, df; vars=:continuous, target=:eta, vcov=:model)

# Effects (μ) on selected vars with grouping and weights
res = ame(m, df; vars=[:x, :z], target=:mu, vcov=:model, over=:g, weights=w, balance=:all)

# Predictions at means (APM) and at a profile grid (APR)
res_apm = apm(m, df; scale=:response)
res_apr = apr(m, df; at=Dict(:x=>[-1,0,1], :g=>["A","B"]), average=true)

# Profile-based effects (MEM/MER)
res_mem = mem(m, df; vars=:continuous, target=:mu)
res_mer = mer(m, df; vars=[:x], target=:mu, at=Dict(:x=>[-1,0,1]))
```

## Migration

- Remove `margins(...)` from the public API (unexport and delete docs/examples).
- Include optional wrappers `effects(...)`/`predictions(...)` that delegate to the six functions (or omit entirely if minimalism is preferred).
- Accept `vars` (alias `dydx`); accept `balance` (replaces `asbalanced`).
- Keep `at` as `:means | Dict | Vector{Dict}`.

## Documentation updates

- README and API docs lead with AME/MEM/MER/APM/APR/APE.
- Clarify `target` vs `scale` (effects vs predictions) and `vcov` sources.
- Short “advanced” section explaining the two axes: row‑averaged vs profiles; effects vs predictions.

## Tests

- Ensure all tests reference AME/MEM/MER/APM/APR/APE (and wrappers if present).
- Add thin wrapper tests asserting equivalence to the six functions (if wrappers exist).

## Non‑goals

- Re‑introducing `margins(...)` or a `mode` switch.
- Over‑structuring: kwargs remain primary; helper specs remain optional sugar.
