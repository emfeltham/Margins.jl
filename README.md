# Margins.jl

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

Marginal effects and adjusted predictions for Julia statistical models.

## Quick Start

```julia
using Margins, DataFrames, GLM

# Fit your model
data = DataFrame(y = randn(1000), x1 = randn(1000), x2 = randn(1000))
model = lm(@formula(y ~ x1 + x2), data)

# Population average marginal effects (AME)
ame_result = population_margins(model, data; type=:effects)

# Marginal effects at sample means (MEM) 
mem_result = profile_margins(model, data; at=:means, type=:effects)

# Convert to DataFrame for analysis
DataFrame(ame_result)
```

## Core API

Margins.jl provides two main functions with a clean conceptual framework:

**Population vs Profile:**
- **`population_margins()`** - Average effects/predictions across observed sample
- **`profile_margins()`** - Effects/predictions at specific evaluation points

**Effects vs Predictions:**
- **`type=:effects`** - Marginal effects (derivatives/contrasts)
- **`type=:predictions`** - Adjusted predictions (fitted values)

```julia
# Four core combinations:
population_margins(model, data; type=:effects)      # Average Marginal Effects (AME)
population_margins(model, data; type=:predictions)  # Average Adjusted Predictions
profile_margins(model, data; type=:effects)         # Marginal Effects at Means (MEM)
profile_margins(model, data; type=:predictions)     # Adjusted Predictions at Means
```

## Key Features

### Elasticities
```julia
# Average elasticities
population_margins(model, data; type=:effects, measure=:elasticity)

# Semi-elasticities
population_margins(model, data; measure=:semielasticity_x)
```

### Profile Specifications
```julia
# At sample means
profile_margins(model, data; at=:means, type=:effects)

# At specific values
profile_margins(model, data; at=Dict(:x1 => [0, 1, 2], :x2 => mean), type=:effects)

# Categorical mixtures
profile_margins(model, data; at=Dict(:group => mix("A" => 0.3, "B" => 0.7)), type=:effects)
```

### Grouping and Stratification
```julia
# Basic grouping
population_margins(model, data; type=:effects, groups=:education)

# Cross-tabulation
population_margins(model, data; type=:effects, groups=[:education, :gender])

# Hierarchical grouping
population_margins(model, data; type=:effects, groups=:region => :education)

# Continuous binning
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
ame = population_margins(model, df; type=:effects, target=:mu)
DataFrame(ame)

# Treatment effects by education
treatment_by_edu = profile_margins(model, df;
    at=Dict(:treatment => [true, false], :education => ["HS", "College", "Graduate"]),
    type=:predictions, target=:mu
)
DataFrame(treatment_by_edu)
```

## Installation

```julia
using Pkg
Pkg.add("Margins")
```

**Requirements**: Julia ≥ 1.9

## Technical Details

**Statistical Methods:**
- Delta-method standard errors with full covariance matrices
- Bootstrap-validated across GLM families

**Integration:**
- GLM.jl and MixedModels.jl support
- Tables.jl interface for data input/output
- CovarianceMatrices.jl for robust standard errors

## Stata Compatibility

| Stata | Margins.jl |
|-------|------------|
| `margins, dydx(*)` | `population_margins(model, data; type=:effects)` |
| `margins, at(means) dydx(*)` | `profile_margins(model, data; at=:means, type=:effects)` |
| `margins, at(x=0 1 2)` | `profile_margins(model, data; at=Dict(:x => [0,1,2]), type=:effects)` |
| `margins` | `population_margins(model, data; type=:predictions)` |
| `margins, at(means)` | `profile_margins(model, data; at=:means, type=:predictions)` |

