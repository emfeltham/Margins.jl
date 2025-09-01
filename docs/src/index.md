# Margins.jl

*Production-ready marginal effects for Julia with Stata-like functionality and superior performance*

## Overview

Margins.jl provides **publication-grade marginal effects computation** for the JuliaStats ecosystem, featuring:

- **Clean 2×2 Framework**: Population vs Profile × Effects vs Predictions
- **Superior Performance**: 250-500x speedup with O(1) constant-time profile analysis  
- **Statistical Rigor**: Zero-tolerance policy for statistical errors with delta-method standard errors
- **Stata Compatibility**: Direct migration path for economists familiar with Stata's `margins` command
- **Production Ready**: Handles datasets from 1k to 1M+ observations seamlessly

Built on the JuliaStats foundation:
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) for model specification
- [GLM.jl](https://github.com/JuliaStats/GLM.jl) for generalized linear models  
- [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) for zero-allocation evaluation
- [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for robust standard errors

## Quick Start

```julia
using Margins, DataFrames, GLM

# Generate sample data
n = 1000
df = DataFrame(
    y = randn(n),
    x1 = randn(n), 
    x2 = randn(n),
    group = rand(["A", "B", "C"], n)
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

Margins.jl uses a **conceptually clean framework** that eliminates statistical acronym confusion:

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

### **🚀 Superior Performance**
- **Profile margins**: O(1) constant time regardless of dataset size (~100-200μs)
- **Population margins**: Optimized O(n) scaling (~150ns per row for effects)  
- **Zero-allocation core**: FormulaCompiler.jl foundation for maximum efficiency
- **Production scaling**: 1k to 1M+ observations supported
- *See [Performance Guide](performance.md) for detailed benchmarks and optimization strategies*

### **📊 Statistical Correctness**
- **Zero-tolerance policy**: Error rather than approximate when statistical validity compromised
- **Delta-method standard errors**: Full covariance matrix integration
- **Bootstrap validated**: All computations verified against bootstrap estimates
- **Publication grade**: Ready for econometric research and academic publication

### **🔧 Advanced Features**  
- **Elasticities**: Full support via `measure` parameter (`:elasticity`, `:semielasticity_x`, `:semielasticity_y`)
- **Categorical mixtures**: Realistic population composition for policy analysis
- **Profile scenarios**: Complex scenario specification with Dict-based and table-based approaches  
- **Robust standard errors**: CovarianceMatrices.jl integration for robust/clustered SEs
- **Flexible grouping**: Subgroup analysis via `over` parameter
- *See [Advanced Features](advanced.md) for detailed coverage of elasticities and robust inference*

### **🔄 Stata Migration**  
Direct command equivalency for economists:

| Stata | Margins.jl |
|-------|------------|
| `margins, dydx(*)` | `population_margins(model, data; type=:effects)` |
| `margins, at(means) dydx(*)` | `profile_margins(model, data; at=:means, type=:effects)` |
| `margins, at(x=0 1 2)` | `profile_margins(model, data; at=Dict(:x=>[0,1,2]), type=:effects)` |
| `margins` | `population_margins(model, data; type=:predictions)` |

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
population_margins(model, data; type=:effects, over=:region)

# Multiple grouping variables
population_margins(model, data; type=:effects, over=[:region, :year])

# Complex nested grouping
population_margins(model, data; type=:effects, over=[:region, :income_quartile])
```

## Statistical Guarantees

Margins.jl follows **zero-tolerance principles** for statistical correctness:

✅ **Delta-method standard errors** use full covariance matrix integration  
✅ **No independence assumptions** unless theoretically justified  
✅ **Error-first policy**: Package errors rather than providing invalid results  
✅ **Bootstrap validation**: All statistical computations verified  
✅ **Publication standards**: Ready for econometric research and academic work  

## Integration with JuliaStats

### Model Compatibility
Works with any StatsModels.jl-compatible model:
- **GLM.jl**: Linear models, logistic regression, Poisson models, etc.
- **MixedModels.jl**: Linear and generalized linear mixed models  
- Custom model types supporting `coef()` and `vcov()` methods

### Data Integration
- **Tables.jl interface**: Works with DataFrames, CSV files, database results
- **MarginsResult type**: Implements Tables.jl for seamless DataFrame conversion
- **Flexible input**: Accepts any Tables.jl-compatible data source

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
@time profile_margins(model, small_data; at=:means)    # ~200μs
@time profile_margins(model, large_data; at=:means)    # ~200μs (same!)

# Complex scenarios also O(1)
scenarios = Dict(:x1 => [0,1,2], :x2 => [10,20,30], :group => ["A","B"])  # 18 profiles
@time profile_margins(model, huge_data; at=scenarios)  # ~400μs
```

### Optimized Population Analysis
```julia
# Population margins scale O(n) with optimized per-row cost
@time population_margins(model, data_1k)    # ~0.2ms  (~150ns/row)
@time population_margins(model, data_10k)   # ~1.5ms  (~150ns/row)  
@time population_margins(model, data_100k)  # ~15ms   (~150ns/row)
```

## Getting Help

- **Documentation**: Complete API reference and mathematical foundation
- **Examples**: Executable workflows in `examples/` directory
- **Issues**: Report bugs at [GitHub Issues](https://github.com/emfeltham/Margins.jl/issues)
- **Migration**: Stata compatibility guide for economists

## Installation

```julia
using Pkg
Pkg.add("Margins")
```

**Requirements**: Julia ≥ 1.9

---

*For conceptual background on the 2×2 framework, see [Mathematical Foundation](mathematical_foundation.md). For performance optimization, see [Performance Guide](performance.md). For comprehensive function reference, see [API Reference](api.md).*