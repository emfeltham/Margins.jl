# Robust/Cluster SEs in Margins.jl (Plan)

This document outlines how to integrate robust, clustered, and HAC standard errors into Margins.jl while staying idiomatic with GLM.jl/StatsModels.jl and leveraging CovarianceMatrices.jl when available.

## Goals

- Use the same covariance sources as the fitted model ecosystem (StatsBase/StatsAPI-compatible `vcov`).
- Make robust/cluster/HAC SEs a drop-in, without re-implementing model-specific sandwich math where possible.
- Keep a clean fallback for OLS robust (HC1) when CovarianceMatrices.jl is not installed.
- Maintain the delta-method interface in Margins (we only need a parameter covariance `Σ`).

## Background: CovarianceMatrices.jl

CovarianceMatrices.jl extends `vcov(model, estimator)` with a family of estimators:

- Heteroskedasticity-robust: `HC0()` … `HC5()`
- Clustered: `Clustered(id)` (or multiple clusters)
- HAC: `HAC(; kernel, bandwidth)` (e.g., Bartlett/Newey–West)
- Panel-robust, Driscoll–Kraay, and more

When present, it provides robust/clustered covariance matrices for common model types (OLS/GLM, etc.).

## Proposed Margins API (Robust options)

- Variance selection via a single `vcov` keyword:
  - `vcov = :model` (default): use `vcov(model)`
  - `vcov = Σ::AbstractMatrix`: explicit covariance override
  - `vcov = f::Function`: callable returning a covariance, e.g. `m -> vcov(m, HC1())`
  - `vcov = estimator` (when CovarianceMatrices.jl is present): e.g., `HC1()`, `Clustered(id)`, `HAC(; kernel, bandwidth)` → internally calls `vcov(model, estimator)`

Behavior (precedence):
- If `vcov` is a matrix, use it (highest precedence).
- Else if `vcov` is a function, use `vcov(model)`.
- Else if `vcov` is an estimator and CovarianceMatrices.jl is available, call `vcov(model, estimator)`.
- Else use `vcov(model)`.

This keeps responsibility for covariance construction outside Margins (consistent with GLM/StatsModels), while Margins continues to apply the delta method using `Σ`.

## Dependency Policy

We will not reimplement robust estimators inside Margins.jl. Instead, we rely on CovarianceMatrices.jl (or the model’s own `vcov` methods) to provide `Σ` for the delta method. If CovarianceMatrices.jl is not available for non-model vcov sources, Margins will:

- Error with a clear message suggesting either installing CovarianceMatrices.jl or passing `vcov` as a matrix or function, or
- Proceed with `vcov=:model` if requested, documenting that SEs are not robust.

## Unconditional (Survey-style) VCE

Stata’s `vce(unconditional)` treats covariates as sampled (variance includes uncertainty in X). This is not provided by CovarianceMatrices.jl. Future options:
- Survey.jl linearization or replicate-weight methods.
- Bootstrap in Margins (recompute margins across replicates and form SEs).

We will defer unconditional until a later phase.

## Wiring in Margins

- Already in place:
  - Single `vcov` keyword controls covariance source for delta-method SEs.
- Next (optional) wiring:
  - `vce`, `robust`, `cluster`, `hac` kwargs.
  - If CovarianceMatrices.jl is present, construct estimators and call `vcov(model, estimator)`.
  - OLS-only fallback HC1 when `vce=:robust` and CovarianceMatrices.jl is absent.

## Examples

- Robust HC1 via CovarianceMatrices:
```julia
using CovarianceMatrices
res = margins(m, df; vcov = HC1())
```

- Clustered by group:
```julia
using CovarianceMatrices
res = margins(m, df; vcov = Clustered(df.group))
```

- Direct Σ override (any source):
```julia
Σ = vcov(m)  # or a custom covariance
res = margins(m, df; vcov=Σ)
```

- OLS fallback HC1 (no CovarianceMatrices):
```julia
# Under the hood, margins will compute HC1 if vce=:robust and model isa LinearModel
res = margins(m, df; vce=:robust)
```

## Testing Plan

- Cross-validate robust SEs against CovarianceMatrices.jl for OLS on small designs.
- For GLM, require CovarianceMatrices.jl path (skip when not available); ensure SEs change appropriately under robust/cluster.
- Confirm that `vcov` overrides drive delta-method SEs in all modes (effects/predictions).

## Roadmap

1) Document and encourage `vcov` usage for robust/cluster/HAC.
2) Consider Bootstrap and Survey integrations for unconditional variance.

This design keeps Margins focused on delta-method application and delegates covariance construction to the model ecosystem, following GLM.jl/StatsModels.jl practices.
