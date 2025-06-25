# Margins

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

This Julia package provides a suite of functions to compute
1. **Adjusted predictions at the mean** and **marginal effects at the mean**
2. **Average Marginal Effects** (AMEs) and related contrasts for continuous and categorical predictors in GLM/GLMM models.

## Core Functions

- effects2()
- effectsΔyΔx()
- ame()

## Example Usage

```julia
using RDatasets # work with iris dataset
using DataFrames, CategoricalArrays
using Margins # development


# Load data
iris = dataset("datasets", "iris") |> DataFrame        # 150 × 5
iris.Species = categorical(iris.Species);


# 1. No interactions – several covariates

form1 = @formula(SepalLength ~ SepalWidth + PetalLength + PetalWidth)

m = lm(form1, iris) # linear regression

ame1  = ame(m, :SepalWidth, iris)
ame2  = ame(m, [:SepalWidth, :PetalLength, :PetalWidth], iris)
```

## Automatic Differentiation

Margins.jl relies on **Forward‐mode AD** ([ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)) to differentiate functions

$$
f: \mathbb{R} \;\to\; \mathbb{R}\quad(\text{or}\;\mathbb{R}\to\mathbb{R}^p)
$$

with respect to **one scalar input** at a time:

1. **Cost scales with num. of inputs**. Forward mode computes all needed partials in one “dual‐number” pass at essentially **O(cost of f)** work when there’s only one AD input.  Reverse mode would record an entire computation graph (“tape”) and then backpropagate through it—even though you only care about one input—incurring both a larger memory footprint and extra traversal time.

2. **Simplicity & stability**.  With ForwardDiff you just wrap your scalar in a `Dual` and call `derivative`.  ReverseDiff (or Zygote) requires building and managing a tape or source‐to‐source transforms, which tend to be heavier weight, can be brittle if your code mutates or uses unsupported language features, and often have longer compile times.

3. **StatsModels with dual numbers**.  Because we’re injecting a single dual into the design‐matrix machinery for each observation, **every** transform (`log`, `^2`, interactions, splines, etc.) automatically propagates that dual.  A reverse‐mode API would need to trace the *entire* modelmatrix construction—overkill when you only need ∂η/∂x, not ∂η/∂all inputs.
