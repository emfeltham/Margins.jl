# Margins.jl

*Marginal effects for Julia*

## Overview

Margins.jl provides marginal effects computation, featuring:

- **Clean 2×2 Framework**: Population vs Profile × Effects vs Predictions
- **Efficient Performance**: O(1) constant-time profile analysis  
- **Statistical Rigor**: Comprehensive testing for statistical correctness with delta-method standard errors
- **Stata Compatibility**: Direct migration path for economists familiar with Stata's `margins` command
- **Scalability**: Tested on datasets from 1k to 1M+ observations

Built for compatibility with JuliaStats foundation:
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) for model specification
- [GLM.jl](https://github.com/JuliaStats/GLM.jl) for generalized linear models  
- [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for robust standard errors

Build on top of [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) for efficient and flexible marginal effect computation.

## Quick Start

```julia
using CategoricalArrays, DataFrames, GLM, Margins

# Generate sample data
n = 1000
df = DataFrame(
    y = randn(n),
    x1 = randn(n), 
    x2 = randn(n),
    group = categorical(rand(["A", "B", "C"], n)) # Raw string input not (yet) supported
)

# Fit model
model = lm(@formula(y ~ x1 + x2 + group), df)

# Population average marginal effects (AME)
ame_result = population_margins(model, df; type=:effects)
DataFrame(ame_result)

# Marginal effects at sample means (MEM)  
mem_result = profile_margins(model, df; at=:means, type=:effects)
DataFrame(mem_result)
```

## The 2×2 Framework

Margins.jl uses a conceptually clean framework that attempts to minimize terminological confusion:

### Two Fundamental Choices

1. **Where to evaluate**:
   - **Population**: Average effects across your observed sample distribution  
   - **Profile**: Effects at specific covariate scenarios (e.g., sample means)

2. **What to compute**:
   - **Effects**: Marginal effects (derivatives for continuous, contrasts for categorical)
   - **Predictions**: Adjusted predictions (fitted values at specified points)

### The Complete Framework

```julia
# Population Analysis (AME/AAP equivalent)
population_margins(model, data; type=:effects)      # Average Marginal Effects
population_margins(model, data; type=:predictions)  # Average Adjusted Predictions

# Profile Analysis (MEM/APM equivalent)  
profile_margins(model, data; at=:means, type=:effects)      # Effects at Sample Means
profile_margins(model, data; at=:means, type=:predictions)  # Predictions at Sample Means
```

## Key Features

### Performance
- **Profile margins**: O(1) constant time regardless of dataset size
- **Population margins**: O(n) scaling with low per-row computational cost  
- **Zero-allocation core**: FormulaCompiler.jl foundation for efficient computation
- **Scalability**: Tested on datasets from 1k to 1M+ observations
- See [Performance Guide](performance.md) for more information

### Statistical Correctness
- **Rigorous validation**: Error rather than approximate when statistical validity compromised
- **Delta-method standard errors**: Full covariance matrix integration
- **Bootstrap validated**: All computations verified against bootstrap estimates
- **Academic standards**: Suitable for econometric research and academic publication

### Advanced Features
- **Elasticities**: Full support via `measure` parameter (`:elasticity`, `:semielasticity_x`, `:semielasticity_y`)
- **Categorical mixtures**: Realistic population composition for policy analysis
- **Profile scenarios**: Complex scenario specification with Dict-based and table-based approaches  
- **Robust standard errors**: CovarianceMatrices.jl integration for robust/clustered SEs
- **Flexible grouping**: Subgroup analysis via `over` parameter
- See [Advanced Features](advanced.md) for coverage of elasticities and robust inference

## Advanced Usage

### Profile Specification

Multiple ways to specify evaluation points:

```julia
# At sample means (most common)
profile_margins(model, data; at=:means, type=:effects)

# Custom scenarios
profile_margins(model, data; 
    at=Dict(:x1 => [0, 1, 2], :group => ["A", "B"]), 
    type=:effects
)

# Complex realistic scenarios with categorical mixtures
using CategoricalArrays
profile_margins(model, data;
    at=Dict(:group => mix("A" => 0.5, "B" => 0.3, "C" => 0.2)),
    type=:predictions
)

# Pre-built reference grids (maximum control)
reference_grid = DataFrame(x1=[0, 1], x2=[0, 0], group=["A", "A"])  
profile_margins(model, reference_grid; type=:effects)
```

### Elasticity Analysis

```julia
# Population average elasticities
population_margins(model, data; type=:effects, measure=:elasticity)

# Elasticities at representative scenarios  
profile_margins(model, data; at=:means, type=:effects, measure=:elasticity)

# Semi-elasticities  
population_margins(model, data; measure=:semielasticity_x)  # % change Y per unit X
population_margins(model, data; measure=:semielasticity_y)  # unit change Y per % X
```

### Subgroup Analysis

```julia
# Effects by categorical groups
population_margins(model, data; type=:effects, groups=:region)

# Multiple grouping variables  
population_margins(model, data; type=:effects, groups=[:region, :year])

# Complex nested grouping
population_margins(model, data; type=:effects, groups=[:region, :income_quartile])
```

## Statistical Guarantees

Margins.jl follows uses:

- Delta-method standard errors: use full covariance matrix integration  
- Error-first policy: Package errors rather than providing invalid results  
- Validation: All statistical computations verified  
- Academic standards: Inteded to be suitable for econometric research and academic work  

## Integration with JuliaStats

### Model Compatibility

Works with StatsModels.jl-compatible models:

- GLM.jl: Linear models, logistic regression, Poisson models, etc.
- MixedModels.jl: Linear and generalized linear mixed models  
- Custom model types supporting `coef()` and `vcov()` methods

### Data Integration

- Tables.jl interface: Works with DataFrames, CSV files, database results
- `MarginsResult` type: Implements Tables.jl for seamless DataFrame conversion
- Flexible input: Accepts any Tables.jl-compatible data source

### Robust Standard Errors
```julia
# Using CovarianceMatrices.jl
using CovarianceMatrices

# Robust standard errors
robust_model = glm(formula, data, family, vcov=HC1())
population_margins(robust_model, data)

# Clustered standard errors  
cluster_model = glm(formula, data, family, vcov=Clustered(:firm_id))
population_margins(cluster_model, data)
```

## Performance Characteristics

### Constant-Time Profile Analysis
```julia
# Profile margins scale O(1) - same time regardless of dataset size
@time profile_margins(model, small_data; at=:means)    # constant time
@time profile_margins(model, large_data; at=:means)    # same time complexity

# Complex scenarios also O(1)
scenarios = Dict(:x1 => [0,1,2], :x2 => [10,20,30], :group => ["A","B"])  # 18 profiles
@time profile_margins(model, huge_data; at=scenarios)  # still constant time
```

### Optimized Population Analysis
```julia
# Population margins scale O(n) with low per-row computational cost
@time population_margins(model, data_1k)    # scales linearly with dataset size
@time population_margins(model, data_10k)   # with efficient per-row processing
@time population_margins(model, data_100k)  # minimal allocation overhead
```

## Getting Help

- **Documentation**: Complete API reference and mathematical foundation
- **Examples**: Executable workflows in `examples/` directory
- **Issues**: Report bugs at [GitHub Issues](https://github.com/emfeltham/Margins.jl/issues)
- **Migration**: Comparison to other packages

## Installation

Since Margins.jl is not yet registered, install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/emfeltham/Margins.jl")
```

**Requirements**: Julia ≥ 1.9

---

*For conceptual background on the 2×2 framework, see [Mathematical Foundation](mathematical_foundation.md). For comprehensive function reference, see [API Reference](api.md).*
