# Margins.jl

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

Marginal effects and adjusted predictions for Julia statistical models.

## Quick Start

```julia
using Margins, DataFrames, GLM

# Fit your model
n = 1_000_00
data = DataFrame(y = randn(n), x1 = randn(n), x2 = randn(n), x3 = x2 = randn(n))
model = lm(@formula(y ~ x1 + x2 + x3), data)

# Population average marginal effects (AME)
ame_result = population_margins(model, data; type=:effects)

# Population average marginal predictions (AAP)
ame_result = population_margins(model, data; type=:predictions)

# Population average marginal predictions (AAP)
ame_result = population_margins(model, data; scenarios=nothing, type=:predictions)

# Marginal effects at sample means (MEM)
mem_result = profile_margins(
    model, data, means_grid(data); type=:effects
)

# Marginal predictions at sample means (APM)
apm_result = profile_margins(
    model, data, means_grid(data); type=:predictions
)

# Convert to DataFrame for analysis
DataFrame(ame_result)

DataFrame(mem_result)

DataFrame(apm_result)
```

```julia
grid = cartesian_grid(
    x1 = collect(range(extrema(data.x1)...; length = 3)),
    x2 = collect(range(extrema(data.x2)...; length = 3))
)

mem_result_ = profile_margins(
    model, data, grid; type=:effects
);

apm_result_ = profile_margins(
    model, data, grid; type=:predictions
);

DataFrame(mem_result_)
DataFrame(apm_result_)
```

## Conceptual Framework

### Two-by-Two Analysis Structure

Margins.jl organizes marginal effects analysis around two orthogonal dimensions:

**Population vs Profile Analysis:**
- **Population**: Average effects/predictions across the observed sample distribution
- **Profile**: Effects/predictions evaluated at specific covariate combinations

**Effects vs Predictions:**
- **Effects**: Marginal effects (derivatives for continuous, contrasts for categorical)
- **Predictions**: Adjusted predictions (fitted values at specified conditions)

### Core Functions

```julia
# Population analysis
population_margins(model, data; type=:effects)      # Average Marginal Effects
population_margins(model, data; type=:predictions)  # Average Adjusted Predictions

# Profile analysis
profile_margins(model, data, means_grid(data); type=:effects)         # Effects at Representative Points
profile_margins(model, data, means_grid(data); type=:predictions)     # Predictions at Representative Points
```

## Advanced Features

### Population Scenarios (at) and Groups
```julia
# Effects at counterfactual values (continuous + boolean)
population_margins(model, data; type=:effects, vars=[:x], scenarios=Dict(:z => 0.5))

# Predictions under scenario grids (Cartesian expansion)
population_margins(model, data; type=:predictions, scenarios=Dict(:x => [-1, 0, 1], :treated => [true, false]))

# Weighted contexts (e.g., survey weights)
population_margins(model, data; type=:effects, vars=[:g], scenarios=Dict(:x => 0.0), weights=:w)

# Grouped analysis (Stata `over()` analogue)
population_margins(model, data; type=:effects, vars=[:x], groups=(:z, 4))     # quartiles of z
population_margins(model, data; type=:effects, vars=[:g], groups=:region => :gender)

# Note: profile_margins uses explicit reference grids and does not accept scenarios
ref = cartesian_grid(x=[-2.0, 0.0, 2.0])
profile_margins(model, data, ref; type=:effects)
```

### Effect Measures
```julia
# Standard marginal effects
population_margins(model, data; type=:effects, measure=:effect)

# Elasticity measures
population_margins(model, data; type=:effects, measure=:elasticity)
population_margins(model, data; measure=:semielasticity_dyex)  # d(y)/d(ln x)
population_margins(model, data; measure=:semielasticity_eydx)  # d(ln y)/dx
```

### Profile Specifications
```julia
# Representative points
profile_margins(model, data, means_grid(data); type=:effects)
profile_margins(model, data, cartesian_grid(x1=[0, 1, 2], x2=[1.5]); type=:effects)

# Categorical compositions via explicit reference grids (profile analysis only)
reference_grid = DataFrame(group=["A", "B"])
profile_margins(model, data, reference_grid; type=:effects)

# Population scenarios (Stata `at()` analogue)
population_margins(model, data; type=:effects, scenarios=Dict(:x1 => 0.0, :treatment => true))
population_margins(model, data; type=:predictions, scenarios=Dict(:x1 => [-2.0, 0.0, 2.0]))
```

### Stratified Analysis
```julia
# Group-wise analysis
population_margins(model, data; type=:effects, groups=:education)
population_margins(model, data; type=:effects, groups=[:education, :gender])

# Hierarchical structures
population_margins(model, data; type=:effects, groups=:region => :education)

# Continuous stratification
population_margins(model, data; type=:effects, groups=(:income, 4))
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
model = glm(@formula(outcome ~ age + education + log(income) + treatment), 
            df, Binomial(), LogitLink())

# Average marginal effects
ame = population_margins(model, df; type=:effects, scale=:response)
DataFrame(ame)

# Treatment effects by education
reference_grid = cartesian_grid(treatment=[true, false], education=["HS", "College", "Graduate"])
treatment_by_edu = profile_margins(model, df, reference_grid; type=:predictions, scale=:response)
DataFrame(treatment_by_edu)
```

## Installation

```julia
using Pkg
Pkg.add("Margins")
```

**Requirements**: Julia ≥ 1.9

## Implementation Details

### Statistical Foundation
- Delta-method standard errors computed using full covariance matrices
- Comprehensive validation through manual counterfactual computation tests
- Zero-tolerance policy for statistical approximations
- Contrast-coding invariance guaranteed across dummy/effects/helmert schemes

### Performance Characteristics
- **O(1) allocation scaling**: Constant memory usage regardless of dataset size
- **Zero-allocation kernels**: All computational cores achieve 0 bytes after warmup
- **Profile margins**: Constant-time complexity independent of data size

### System Integration
- Native support for GLM.jl and MixedModels.jl fitted models
- Tables.jl interface for flexible data input and result output
- CovarianceMatrices.jl integration for robust standard error computation
- FormulaCompiler.jl backend for high-performance evaluation

## Migration Reference

### From Stata

| Stata Command | Margins.jl Equivalent |
|---------------|----------------------|
| `margins, dydx(*)` | `population_margins(model, data; type=:effects)` |
| `margins, at(means) dydx(*)` | `profile_margins(model, data, means_grid(data); type=:effects)` |
| `margins, at(x=0 1 2)` | `profile_margins(model, data, cartesian_grid(x=[0,1,2]); type=:effects)` |
| `margins` | `population_margins(model, data; type=:predictions)` |
| `margins, at(means)` | `profile_margins(model, data, means_grid(data); type=:predictions)` |
