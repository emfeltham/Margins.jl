# Margins.jl

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

Stata-like marginal effects for the JuliaStats stack, built on:
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) (formulas/design)
- [GLM.jl](https://github.com/JuliaStats/GLM.jl) (links/predict types)
- [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) (zero-allocation compiled evaluators)
- [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) (robust/cluster/HAC covariance)

This rewrite provides:
- Marginal effects: AME, MEM, MER (η or μ)
- Adjusted predictions: APE, APM, APR (link or response scale)
- Categorical contrasts: baseline/pairwise
- Weights, grouping (over), profiles (at), elasticities
- Delta-method SEs with normal/t CIs, multiple-comparison adjustments

See MARGINS_PLAN.md for full design and ROBUST.md for VCE integration.

## API

```julia
Margins.margins(model, data; 
  mode=:effects,              # :effects or :predictions
  dydx=:continuous,           # variables for effects (or :continuous)
  target=:mu,                 # :mu (response) or :eta (link)
  at=:none,                   # :none | :means | Dict/Vector{Dict} profiles
  over=nothing,               # grouping columns
  backend=:ad,                # FormulaCompiler backend (:ad or :fd)
  scale=:auto,                # :auto|:response|:link for predictions
  vcov=:model,                # :model | matrix | function | estimator (CovarianceMatrices)
  weights=nothing,            # vector or column Symbol
  measure=:effect,            # :effect|:elasticity|:semielasticity_x|:semielasticity_y
  mcompare=:noadjust,         # :noadjust|:bonferroni|:sidak
)

# Convenience
ame(...); mem(...); mer(...)
ape(...); apm(...); apr(...)
```

Results return a `MarginsResult` with DataFrame conversion support and metadata (Σ source, link, dof, etc.).

## Examples

```julia
using DataFrames, CategoricalArrays, GLM, Margins

df = DataFrame(y = rand(Bool, 1000), x = randn(1000), z = randn(1000), g = categorical(rand(["A","B"], 1000)))
m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())

# AME on response scale
res_ame = ame(m, df; dydx=[:x, :z], target=:mu)

# MER at representative values
res_mer = mer(m, df; dydx=[:x], target=:mu, at=Dict(:x=>[-1,0,1], :g=>["A","B"]))

# Adjusted predictions (APR) on link or response
res_apr_mu = apr(m, df; target=:mu, at=Dict(:x=>[-2,0,2]))
res_apr_eta = apr(m, df; target=:eta, at=Dict(:x=>[-2,0,2]))

# Robust covariance via CovarianceMatrices
# using CovarianceMatrices
# res_robust = ame(m, df; dydx=[:x], vcov = HC1())
```

## JuliaStats Compatibility

- Formulas and design: StatsModels.jl
- Predict types: `pred_type=:response|:link` matches GLM conventions
- Covariance: defaults to `vcov(model)`; for robust/cluster/HAC use CovarianceMatrices.jl 
- Data: any Tables.jl table
- Mixed models: operates on fixed effects (FormulaCompiler extracts fixed part)

## Notes

- This is an active rewrite; APIs may evolve.
- Delta method SEs require only a parameter covariance `Σ` and gradients; all robust logic remains in the covariance provider.
