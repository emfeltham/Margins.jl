# Margins.jl

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

A Julia package for computing marginal effects with Stata-compatible functionality.

Built on the JuliaStats ecosystem:
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) - Formulas and model specification
- [GLM.jl](https://github.com/JuliaStats/GLM.jl) - Generalized linear models and link functions  
- [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) - Zero-allocation high-performance evaluation
- [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) - Robust/clustered standard errors

## Overview

**Computational Characteristics:**
- Profile margins: O(1) constant time complexity
- Population margins: O(n) linear scaling
- Tested on datasets ranging from 1k to 1M+ observations

**Statistical Properties:**
- Delta-method standard errors using full covariance matrices
- Bootstrap-validated estimates across GLM families
- Designed for econometric research applications

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

## Core API: Clean 2×2 Framework

Margins.jl implements a 2×2 computational framework:

### **Population vs Profile**
- **`population_margins()`**: Effects/predictions averaged over your observed sample (AME/AAP equivalent)
- **`profile_margins()`**: Effects/predictions at specific evaluation points (MEM/APM equivalent)

### **Effects vs Predictions**  
- **`type=:effects`**: Marginal effects (derivatives for continuous, contrasts for categorical)
- **`type=:predictions`**: Adjusted predictions (fitted values at specified points)

```julia
# The four core combinations:
population_margins(model, data; type=:effects)      # AME: Average Marginal Effects
population_margins(model, data; type=:predictions)  # AAP: Average Adjusted Predictions  
profile_margins(model, data; type=:effects)         # MEM: Marginal Effects at Means
profile_margins(model, data; type=:predictions)     # APM: Adjusted Predictions at Means
```

## Additional Features

### **Elasticities** 
```julia
# Population average elasticities
population_margins(model, data; type=:effects, measure=:elasticity)

# Semi-elasticities (x-elasticity, y-elasticity)  
population_margins(model, data; measure=:semielasticity_x)
profile_margins(model, data; at=:means, measure=:semielasticity_y)
```

### **Profile Specifications**
```julia
# Effects at sample means
profile_margins(model, data; at=:means, type=:effects)

# Effects at specific values
profile_margins(model, data; at=Dict(:x1 => [0, 1, 2], :x2 => mean), type=:effects)

# Categorical mixtures for realistic scenarios
using CategoricalArrays
profile_margins(model, data; at=Dict(:group => mix("A" => 0.3, "B" => 0.7)), type=:effects)
```

### **Grouping and Stratification**
```julia
# Compute effects within groups
population_margins(model, data; type=:effects, groups=:region)

# Cross-tabulated grouping
population_margins(model, data; type=:effects, groups=[:region, :year])
```

## Complete Example: Logistic Regression

```julia
using Margins, DataFrames, GLM, CategoricalArrays

# Create sample data
n = 5000
df = DataFrame(
    age = rand(18:65, n),
    education = categorical(rand(["High School", "College", "Graduate"], n)),
    income = exp.(randn(n) * 0.5 + 3),  # Log-normal income
    treatment = rand([true, false], n)
)

# Create outcome with realistic relationships
linear_pred = -2.0 + 0.05*df.age + 0.3*(df.education .== "College") + 
              0.6*(df.education .== "Graduate") + 0.2*log.(df.income) + 1.2*df.treatment
df.outcome = [rand() < 1/(1+exp(-lp)) for lp in linear_pred]

# Fit logistic model
model = glm(@formula(outcome ~ age + education + log(income) + treatment), 
            df, Binomial(), LogitLink())

# Average marginal effects (population-level)
ame = population_margins(model, df; type=:effects, target=:mu)
println(DataFrame(ame))

# Effects at representative profiles
mem = profile_margins(model, df; at=:means, type=:effects, target=:mu)
println(DataFrame(mem))

# Predicted probabilities at specific scenarios
scenarios = profile_margins(model, df; 
    at=Dict(:age => [25, 45, 65], :treatment => [true, false]),
    type=:predictions, target=:mu
)
println(DataFrame(scenarios))

# Treatment effect across education levels
treatment_effects = profile_margins(model, df;
    at=Dict(:treatment => [true, false], :education => ["High School", "College", "Graduate"]),
    type=:predictions, target=:mu
)
println(DataFrame(treatment_effects))
```

## Statistical Implementation

Margins.jl implements standard econometric methods:

- Standard errors computed using the delta method with full covariance matrices
- No independence assumptions beyond those of the underlying model
- Statistical computations validated against bootstrap estimates
- Designed for econometric research applications  

## JuliaStats Integration

Integration with the JuliaStats ecosystem:
- **Models**: Complete support for GLM.jl (lm, glm) and MixedModels.jl (LinearMixedModel, GeneralizedLinearMixedModel)
- **Data**: Accepts any Tables.jl-compatible data (DataFrame, CSV, etc.)  
- **Covariance**: Uses `vcov(model)` by default, supports CovarianceMatrices.jl for robust/clustered SEs
- **Results**: `MarginsResult` implements Tables.jl interface for easy DataFrame conversion

## Installation

```julia
using Pkg
Pkg.add("Margins")
```

**Requirements**: Julia ≥ 1.9

## Support Stata-like Workflows

For users familiar with Stata's `margins` command:

| Stata | Margins.jl |
|-------|------------|
| `margins, dydx(*)` | `population_margins(model, data; type=:effects)` |
| `margins, at(means) dydx(*)` | `profile_margins(model, data; at=:means, type=:effects)` |
| `margins, at(x=0 1 2)` | `profile_margins(model, data; at=Dict(:x => [0,1,2]), type=:effects)` |
| `margins` | `population_margins(model, data; type=:predictions)` |
| `margins, at(means)` | `profile_margins(model, data; at=:means, type=:predictions)` |

## Performance Notes

**Profile margins**: Constant time complexity, approximately 100-200μs per analysis.

**Population margins**: Linear time complexity with low per-row computational cost.

## Acknowledgements

Margins.jl builds upon decades of research and development in marginal effects computation. I gratefully acknowledge the foundational work of:

**Related Work:**
- The Stata development team for their pioneering `margins` command, which established the conceptual framework and computational standards that inspired this package
- Vincent Arel-Bundock and colleagues for the [`marginaleffects`](https://marginaleffects.com/) R package, whose comprehensive approach to effect computation and cross-language consistency influenced our design
- Russell Lenth and colleagues for the [`emmeans`](https://rvlenth.github.io/emmeans/) R package, whose treatment of estimated marginal means provided valuable insights for profile-based computations

**Julia Ecosystem Contributors:**
- The maintainers of [`Effects.jl`](https://github.com/beacon-biosignals/Effects.jl) for their early work on marginal effects in Julia and establishing Julia-specific conventions
- The [JuliaStats](https://github.com/JuliaStats) community, particularly the authors of:
  - [`GLM.jl`](https://github.com/JuliaStats/GLM.jl) for robust GLM implementation and the prediction interface that Margins.jl extends
  - [`MixedModels.jl`](https://github.com/JuliaData/MixedModels.jl) for comprehensive mixed-effects modeling support
  - [`StatsModels.jl`](https://github.com/JuliaStats/StatsModels.jl) for the formula interface and model abstraction that enables seamless integration

Margins.jl builds on this foundation with efficient computation and comprehensive statistical validation for the Julia ecosystem.

---

*Developed by Eric Feltham (eric.feltham@aya.yale.edu)*