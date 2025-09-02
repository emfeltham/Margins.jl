# Margins.jl

[![Build Status](https://github.com/emfeltham/Margins.jl/workflows/CI/badge.svg)](https://github.com/emfeltham/Margins.jl/actions)

**Marginal effects for Julia with Stata-like functionality and efficient computation.**

Built on the JuliaStats ecosystem:
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) - Formulas and model specification
- [GLM.jl](https://github.com/JuliaStats/GLM.jl) - Generalized linear models and link functions  
- [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) - Zero-allocation high-performance evaluation
- [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) - Robust/clustered standard errors

## Performance Highlights

**Performance:**
- **Profile margins**: O(1) constant time scaling
- **Population margins**: O(n) scaling with minimal allocations
- **Zero-allocation core**: FormulaCompiler.jl foundation provides efficient computation
- **Scalability**: Tested on datasets from 1k to 1M+ observations

**Statistical Correctness:**
- Delta-method standard errors with full covariance matrix integration
- Bootstrap-validated statistical correctness across all GLM families
- Suitable for econometric research and academic publication

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

Margins.jl uses a **conceptually clear 2×2 framework** that replaces confusing statistical acronyms:

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

## Advanced Features

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
population_margins(model, data; type=:effects, over=:region)

# Nested grouping
population_margins(model, data; type=:effects, over=[:region, :year])
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

## Statistical Correctness Guarantees

Margins.jl follows a **zero-tolerance policy for statistical errors**:

- All standard errors use proper delta-method with full covariance matrix  
- No independence assumptions unless theoretically justified  
- Error-first policy: Package errors rather than providing invalid results  
- Bootstrap validated: All statistical computations verified against bootstrap estimates  
- Suitable for econometric research and academic publication  

## JuliaStats Integration

**Seamless integration with the broader ecosystem:**
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

## Performance Comparison

**Profile margins performance** (constant time regardless of dataset size):
- Approximately 100-200μs per analysis across different dataset sizes
- Substantial performance improvement over naive implementations

**Population margins performance** (O(n) scaling):
- Low per-row computational cost for effects and predictions
- Minimal memory allocation footprint

---

*Margins.jl: Statistical rigor meets Julia performance for econometric analysis.*