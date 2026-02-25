# Margins.jl

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)
[![Docs Build](https://github.com/emfeltham/Margins.jl/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/emfeltham/Margins.jl/actions/workflows/docs.yml)
[![Docs: Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://emfeltham.github.io/Margins.jl/stable/)
[![Docs: Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://emfeltham.github.io/Margins.jl/dev/)

Marginal effects and adjusted predictions for Julia statistical models.

## Quick Start

```julia
using Margins, DataFrames, GLM

# Fit a model
n = 100_000
data = DataFrame(y = randn(n), x1 = randn(n), x2 = randn(n), x3 = randn(n))
model = lm(@formula(y ~ x1 + x2 + x3), data)

# Population average marginal effects (AME)
ame_result = population_margins(model, data; type = :effects)

# Population average marginal predictions (AAP)
aap_result = population_margins(model, data; type = :predictions)

# Marginal effects at sample means (MEM)
mem_result = profile_margins(
    model, data, means_grid(data); type = :effects
)

# Marginal predictions at sample means (APM)
apm_result = profile_margins(
    model, data, means_grid(data); type = :predictions
)

# Convert to DataFrame for analysis
DataFrame(ame_result)

DataFrame(aap_result)

DataFrame(mem_result)

DataFrame(apm_result)
```

```julia
grid = cartesian_grid(
    x1 = collect(range(extrema(data.x1)...; length = 3)),
    x2 = collect(range(extrema(data.x2)...; length = 3))
)

mem_result_ = profile_margins(model, data, grid; type = :effects);

apm_result_ = profile_margins(model, data, grid; type = :predictions);

DataFrame(mem_result_)
DataFrame(apm_result_)
```

## Conceptual Framework

### Two-by-Two Analysis Structure

Margins.jl organizes marginal effects analysis around two orthogonal dimensions:

**Population vs Profile Analysis:**
- Population: Average effects/predictions across the observed sample distribution
- Profile: Effects/predictions evaluated at specific covariate combinations

**Effects vs Predictions:**
- Effects: Marginal effects (derivatives for continuous, contrasts for boolean/categorical)
- Predictions: Adjusted predictions (fitted values at specified conditions)

### Core Functions

```julia
# Population analysis
population_margins(model, data; type = :effects) # Average Marginal Effects
population_margins(model, data; type = :predictions) # Average Adjusted Predictions

# Profile analysis
# Effects at Representative Points
profile_margins(model, data, means_grid(data); type = :effects)
# Predictions at Representative Points
profile_margins(model, data, means_grid(data); type = :predictions)
```

## Advanced Features

### Population Scenarios (at) and Groups
```julia
# Effects at counterfactual values (continuous + boolean)
population_margins(
    model, data;
    type = :effects, vars = [:x], scenarios = (z = 0.5,)
)

# Predictions under scenario grids (Cartesian expansion)
population_margins(
    model, data;
    type = :predictions, scenarios = (x = [-1, 0, 1], treated = [true, false])
)

# Weighted contexts (e.g., survey weights)
population_margins(
    model, data;
    type = :effects, vars = [:g], scenarios = (x = 0.0,), weights = :w
)

# Grouped analysis (analogue to Stata's `over()`)
population_margins(
    model, data; type = :effects, vars = [:x], groups = (:z, 4)
) # quartiles of z
population_margins(
    model, data; type = :effects, vars = [:g], groups = :region => :gender
)

# N.B., profile_margins uses explicit reference grids instead of scenarios and groups
ref = cartesian_grid(x=[-2.0, 0.0, 2.0])
profile_margins(model, data, ref; type = :effects)
```

### Effect Measures
```julia
# Standard marginal effects
population_margins(model, data; type = :effects, measure = :effect)

# Elasticity measures
population_margins(model, data; type = :effects, measure = :elasticity)
population_margins(model, data; type = :effects, measure = :semielasticity_dyex) # d(y)/d(ln x)
population_margins(model, data; type = :effects, measure = :semielasticity_eydx) # d(ln y)/dx
```

### Reference Grids
```julia
# Sample means (single-row grid)
means_grid(data)

# Cartesian product of specified values
cartesian_grid(x1 = [0, 1, 2], x2 = [1.5])

# Balanced grid: typical values for unspecified variables, all levels for categoricals
balanced_grid(data; education = :all, income = [30000, 50000])

# Quantile grid: evaluate at quantiles of continuous variables
quantile_grid(data; income = [0.25, 0.5, 0.75])
```

### Computation Backend
```julia
# Automatic differentiation (default) — exact derivatives
population_margins(model, data; type = :effects, backend = :ad)

# Finite differences — zero allocations in hot path
population_margins(model, data; type = :effects, backend = :fd)
```

### Stratified Analysis
```julia
# Group-wise analysis
population_margins(model, data; type = :effects, groups = :education)
population_margins(model, data; type = :effects, groups = [:education, :gender])

# Hierarchical structures
population_margins(model, data; type = :effects, groups = :region => :education)

# Continuous stratification
population_margins(model, data; type = :effects, groups = (:income, 4))
```

## Example: Logistic Regression

```julia
using Margins, DataFrames, GLM, CategoricalArrays

# Sample data
n = 1000
df = DataFrame(
    age = rand(25:65, n),
    education = categorical(rand(["HS", "College", "Graduate"], n)),
    income = exp.(randn(n) * 0.5 + 4),
    treatment = rand([true, false], n)
)

# Create outcome
linear_pred = -2.0 + 0.05*df.age + 0.5*(df.education .== "College") + 
    1.0*(df.education .== "Graduate") + 0.3*log.(df.income) + 1.5*df.treatment
df.outcome = [rand() < 1/(1+exp(-lp)) for lp in linear_pred]

# Fit model
fx = @formula(outcome ~ age + education + log(income) + treatment);
model = glm(fx, df, Binomial(), LogitLink())

# Average marginal effects
ame = population_margins(model, df; type = :effects, scale = :response)
DataFrame(ame)

# Treatment effects by education
reference_grid = cartesian_grid(
    treatment = [true, false], education = ["HS", "College", "Graduate"]
)
treatment_by_edu = profile_margins(
    model, df, reference_grid; type = :predictions, scale = :response
)
DataFrame(treatment_by_edu)
```

## Installation

```julia
using Pkg
Pkg.add("Margins")
```

**Requirements**: Julia ≥ 1.10

## Implementation Details

### Statistical Foundation
- Delta-method standard errors
- Comprehensive validation through manual counterfactual computation tests
- Contrast-coding invariance guaranteed across dummy/effects/helmert schemes

### Performance Characteristics
- O(1) allocation scaling: Constant memory usage regardless of dataset size
- Zero-allocation kernels: All computational cores achieve 0 bytes after warmup
- Profile margins: Constant-time complexity independent of data size

### System Integration
- Support for GLM.jl and MixedModels.jl fitted models
- Tables.jl interface for flexible data input and result output
- [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) integration for robust standard error computation
- [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) backend for high-performance evaluation

## Migration Reference

### From Stata

| Stata Command | Margins.jl Equivalent |
|---------------|----------------------|
| `margins, dydx(*)` | `population_margins(model, data; type = :effects)` |
| `margins, at(means) dydx(*)` | `profile_margins(model, data, means_grid(data); type = :effects)` |
| `margins, at(x=0 1 2)` | `profile_margins(model, data, cartesian_grid(x=[0,1,2]); type = :predictions)` |
| `margins, at(x=0 1 2) dydx(*)` | `profile_margins(model, data, cartesian_grid(x=[0,1,2]); type = :effects)` |
| `margins` | `population_margins(model, data; type = :predictions)` |
| `margins, at(means)` | `profile_margins(model, data, means_grid(data); type = :predictions)` |

## Citation

If you use Margins.jl in your research, please cite:

```bibtex
@misc{feltham_formulacompilerjl_2026,
  title = {{FormulaCompiler}.jl and {Margins}.jl: {Efficient} Marginal Effects in {Julia}},
  shorttitle = {{FormulaCompiler}.jl and {Margins}.jl},
  author = {Feltham, Eric},
  year = {2026},
  month = jan,
  number = {arXiv:2601.07065},
  eprint = {2601.07065},
  primaryclass = {stat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2601.07065},
  urldate = {2026-01-13},
  abstract = {Marginal effects analysis is fundamental to interpreting statistical models, yet existing implementations face computational constraints that limit analysis at scale. We introduce two Julia packages that address this gap. Margins.jl provides a clean two-function API organizing analysis around a 2-by-2 framework: evaluation context (population vs profile) by analytical target (effects vs predictions). The package supports interaction analysis through second differences, elasticity measures, categorical mixtures for representative profiles, and robust standard errors. FormulaCompiler.jl provides the computational foundation, transforming statistical formulas into zero-allocation, type-specialized evaluators that enable O(p) per-row computation independent of dataset size. Together, these packages achieve 622x average speedup and 460x memory reduction compared to R's marginaleffects package, with successful computation of average marginal effects and delta-method standard errors on 500,000 observations where R fails due to memory exhaustion, providing the first comprehensive and efficient marginal effects implementation for Julia's statistical ecosystem.},
  archiveprefix = {arXiv},
  keywords = {Statistics - Computation},
}
```
