# Margins.jl

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

This Julia package provides a suite of functions to compute:
1. **Adjusted predictions at the mean** and **marginal effects at the mean**
2. **Average Marginal Effects** (AMEs) and related contrasts for predictors in GLM/GLMM models.

As it stands, marginal effect calculations, and AME calculations in particular, are a huge gap in statistical modeling in Julia that really limits the ways researchers can take advantage of packages like [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl). Furthermore, other packages that seek to convert models estimated in Julia into R objects (which can then be used with the mature modeling ecosystem) ultimately feed into another two-language problem (though this strategy may be the best current option in many situations).

Also, note that this package is similar in spirit to [Effects.jl](https://github.com/beacon-biosignals/Effects.jl), and borrows directly from it for the APM calculations.

## Development

- This package is in early stages of development.
- See the "demo_ame_tests.jl" file for the complete set of cases that I have tested.
- So far, I have only tested a limited set of GLM models.

## Core functions

- `effects2()`
- `effectsΔyΔx()`
- `ame()`

## Example usage for AMEs

```julia
using RDatasets # work with iris dataset
using DataFrames, CategoricalArrays
using Margins # development


# Load data (150 × 5)
iris = dataset("datasets", "iris") |> DataFrame
iris.Species = categorical(iris.Species);


# 1. No interactions – several covariates

form1 = @formula(SepalLength ~ SepalWidth + PetalLength + PetalWidth)

m = lm(form1, iris) # linear regression

ame1  = ame(m, :SepalWidth, iris)
ame2  = ame(m, [:SepalWidth, :PetalLength, :PetalWidth], iris)
```

**More cases to come...**

(cf. "demo_ame_tests.jl")

## Example usage for APMs

```julia
using DataFrames, CategoricalArrays, GLM

# Simulate data
df = DataFrame(
    y = randn(100) .+ 2 .* (rand(100) .> 0.5),
    x = rand(100),
    g = categorical(rand(["A","B"], 100))
)

# Fit a linear model
m = lm(@formula(y ~ x + g), df)

# Build a reference grid over x
dct = Dict(x => range(extrema(df.x)..., 5))
grid = expand_grid(dct)

# Compute effects at the mean of g
effects2!(grid, m, df; eff_col=:pred, err_col=:se_pred)
# grid now contains :pred and :se_pred columns
```

### APM contrasts

```julia
effectsΔyΔx(...)
```

(Example to come)

## Mixed models

**Current limitation**: All predictions and summaries are based solely on the fixed‐effects. Random effects are treated as zero (_i.e._, we do not marginalize or integrate over their distribution in GLMMs).

## Automatic differentiation

Margins.jl relies on **Forward‐mode AD** ([ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)) to differentiate functions with respect to one scalar input at a time.

Forward mode computes all needed partials in one “dual‐number” pass at essentially $O(\text{cost of f})$ work when there’s only one AD input. With ForwardDiff we wrap scalars in a `Dual` and call `derivative`. Because we inject a single dual into the design‐matrix machinery for each observation, _every_ transform (`log`, `^2`, interactions, splines, etc.) automatically propagates that dual. Furthermore, this strategy integrates easily with the [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) framework, automatically propagating derivatives through data transformations (_e.g._, log(x), x^2, interactions with other variables) without requiring complex tracing of the entire model matrix construction.

## Other issues

Possibly planned.

- [ ] inspect APM workflow, possibly change UI
- [ ] elasticities
- [ ] efficiency for larger data
- [ ] DataFrame construction (from `AMEResult`, `ContrastResult`)
- [ ] plotting integration
