# Margins.jl

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

with respect to **one scalar input** at a time.

Forward mode computes all needed partials in one “dual‐number” pass at essentially _O(cost of f)_ work when there’s only one AD input. With ForwardDiff we wrap scalars in a `Dual` and call `derivative`. Because we inject a single dual into the design‐matrix machinery for each observation, _every_ transform (`log`, `^2`, interactions, splines, etc.) automatically propagates that dual. Furthermore, this strategy integrates easily with the [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) framework, automatically propagating derivatives through data transformations (_e.g._, log(x), x^2, interactions with other variables) without requiring complex tracing of the entire model matrix construction.
