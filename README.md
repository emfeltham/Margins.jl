# Margins

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

This Julia package provides a suite of functions to compute
1. **Adjusted predictions at the mean** and **marginal effects at the mean**
2. **Average Marginal Effects** (AMEs) and related contrasts for continuous and categorical predictors in GLM/GLMM models.

## Core Functions

* **`ame_continuous(df, model, x; δ, invlink, dinvlink, d2invlink, vcov)`**

  Compute the average marginal effect of a continuous variable `x`. Numerically approximates $\partial\mu/\partial x$ at each observation’s own covariates and averages across the sample. Returns an `AME` struct containing:

  * `ame`, `se`, `grad`, `n`, `η_base`, `μ_base`.

* **`ame_interaction_continuous(df, model, x, z, z_val; δ, typical, invlink, dinvlink, d2invlink, vcov)`**

  Compute the AME of a continuous `x` at a fixed moderator `z = z_val`, holding all other covariates at a `typical` value (default: mean). Returns an `AME` struct.

* **`ame_discrete_contrast(df, model, x; values, invlink, dinvlink, vcov)`**

  Compute the average discrete-change effect of flipping a binary (or two-level) variable `x` from `values[1]` → `values[2]`. Returns an `AMEContrast` struct with fields:

  * `ame_low`, `se_low`, `ame_high`, `se_high`, `ame_diff`, `se_diff`, `grad_low`, `grad_high`, `n`, `names`.

## Curve Helpers

* **`marginal_effect_curve_z(df, model, x, z; K, δ, typical, invlink, dinvlink, d2invlink, vcov)`**

  Sweep the moderator `z` over `K` evenly spaced values (from `min(z)` to `max(z)`) and compute $\mathrm{AME}(x\mid z_j)$ at each slice.  Returns a `DataFrame` with columns:

  * `z_grid`, `ame`, `se_me`.

* **`marginal_effect_curve_x(df, model, x; K, δ, typical, invlink, dinvlink, d2invlink, vcov)`**

  Sweep the focal variable `x` over `K` values (from `min(x)` to `max(x)`), holding all other covariates at `typical`. Returns a `DataFrame` with columns:

  * `x_grid`, `ame`, `se`.

* **`discrete_effect_curve(df, model, x, values; invlink, dinvlink, vcov)`**

  For a discrete `x` with multiple target values, computes the contrast from the baseline `values[1]` → each `values[j]`. Returns a `DataFrame` with columns:

  * `x_target`, `ame_diff`, `se_diff`.

* **`ame_factor_contrasts(df, model, x; invlink, dinvlink, vcov)`**

  For a categorical predictor `x` with levels $ℓ_1,ℓ_2,\dots,ℓ_k$, computes all pairwise AME contrasts ($ℓ_i \to ℓ_j$, $i<j$). Returns a `DataFrame` with:

  * `from`, `to`, `ame_diff`, `se_diff`.

## Data Structures

* **`struct AME`**

  * `ame::Float64`
  * `se::Float64`
  * `grad::Vector{Float64}`
  * `n::Int`
  * `η_base::Vector{Float64}`
  * `μ_base::Vector{Float64}`

* **`struct AMEContrast`**

  * `ame_low::Float64`, `se_low::Float64`
  * `ame_high::Float64`, `se_high::Float64`
  * `ame_diff::Float64`, `se_diff::Float64`
  * `grad_low::Vector{Float64}`, `grad_high::Vector{Float64}`
  * `n::Int`, `names::Vector{String}`

## Example Usage

```julia
using DataFrames, MixedModels

# Fit a Bernoulli GLMM
df = DataFrame(y = rand(Bool,200), x = randn(200), z = randn(200), g = rand(1:5,200))
model = fit!(GeneralizedLinearMixedModel(@formula(y ~ x*z + (1|g)), df, Bernoulli()))

# Continuous AME of x
out = ame_continuous(df, model, :x)
println(out)

# AME of x at z = 0
out_z = ame_interaction_continuous(df, model, :x, :z, 0.0)

# Discrete contrast of binary x
res_disc = ame_discrete_contrast(df, model, :x, values=(0,1))

# Curve of AME vs z
df_curve = marginal_effect_curve_z(df, model, :x, :z; K=30)
```

## Automatic Differentiation

This package uses **Forward‐mode AD** (ForwardDiff.jl) to differentiate a function

$$
f: \mathbb{R} \;\to\; \mathbb{R}\quad(\text{or}\;\mathbb{R}\to\mathbb{R}^p)
$$

with respect to **one scalar input** at a time:

1. **Cost scales with #inputs**. Forward mode computes all needed partials in one “dual‐number” pass at essentially **O(cost of f)** work when there’s only one AD input.  Reverse mode would record an entire computation graph (“tape”) and then backpropagate through it—even though you only care about one input—incurring both a larger memory footprint and extra traversal time.

2. **Simplicity & stability**.  With ForwardDiff you just wrap your scalar in a `Dual` and call `derivative`.  ReverseDiff (or Zygote) requires building and managing a tape or source‐to‐source transforms, which tend to be heavier weight, can be brittle if your code mutates or uses unsupported language features, and often have longer compile times.

3. **StatsModels plays nice with dual numbers**.  Because we’re injecting a single dual into the design‐matrix machinery for each observation, **every** transform (`log`, `^2`, interactions, splines, etc.) automatically propagates that dual.  A reverse‐mode API would need to trace the *entire* modelmatrix construction—overkill when you only need ∂η/∂x, not ∂η/∂all inputs.

> **When *would* reverse‐mode make sense here?**
> If you ever wanted the gradient of a **scalar loss** (say, a log-likelihood) with respect to a **high-dimensional** parameter vector β, reverse‐mode shines.  But for “one covariate → linear predictor” many times over, forward‐mode is both simpler and faster.
