# Examples Guide

This directory contains comprehensive examples demonstrating the capabilities of Margins.jl across different use cases and complexity levels. Each example is designed to be self-contained and executable.

## Overview

Margins.jl provides a clean **2×2 framework** for marginal analysis:

- **Population vs Profile**: Average effects across sample vs. effects at specific scenarios
- **Effects vs Predictions**: Marginal effects (derivatives/contrasts) vs. adjusted predictions (fitted values)

## Example Files

### 1. **basic_usage.jl** - Getting Started
**Purpose**: Introduction to the fundamental 2×2 framework with simple examples.

**Key Features Demonstrated**:
- Population vs Profile comparison using linear models
- Basic marginal effects and predictions
- GLM (logistic regression) examples
- Converting results to DataFrames

**Best For**: New users learning the core concepts.

**Run Time**: ~10 seconds

```julia
# Population marginal effects (AME)
result = population_margins(model, data; type=:effects, vars=[:x1, :x2])

# Profile marginal effects at means (MEM) 
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1, :x2])
```

### 2. **basic_workflow.jl** - Essential Patterns
**Purpose**: Complete workflow demonstrating practical analysis patterns.

**Key Features Demonstrated**:
- Data setup and model fitting
- The 2×2 framework in practice (AME, AAP, MEM, APM)
- Reference grid specification patterns
- Subgroup analysis with grouping parameters
- Results handling and export

**Best For**: Users planning their first real analysis.

**Run Time**: ~15 seconds

```julia
# Subgroup analysis
result = population_margins(model, data; type=:effects, groups=:education)

# Custom scenarios
result = profile_margins(model, data, cartesian_grid(age=[30, 45, 60], treatment=[0, 1]); type=:predictions)
```

### 3. **margins_glm.jl** - GLM Applications
**Purpose**: Comprehensive examples for generalized linear models.

**Key Features Demonstrated**:
- Logistic regression marginal effects
- Link vs response scale effects
- Binary outcome interpretation
- Population vs profile framework for GLMs

**Best For**: Users working with binary/count outcomes.

**Run Time**: ~8 seconds

```julia
# GLM effects on response scale (probabilities)
result = population_margins(logit_model, data; type=:effects, scale=:response)

# GLM effects on link scale (log-odds)  
result = population_margins(logit_model, data; type=:effects, scale=:link)
```

### 4. **stata_migration.jl** - Stata User Guide
**Purpose**: Direct command translations for Stata users migrating to Julia.

**Key Features Demonstrated**:
- Exact Stata command equivalencies
- Marginal effects at means translation
- Subgroup analysis (`over` parameter equivalents)
- Post-estimation analysis patterns
- Migration checklist and best practices

**Best For**: Economists and social scientists familiar with Stata.

**Run Time**: ~25 seconds

```julia
# Stata: margins, dydx(*)
julia_equivalent = population_margins(model, data; type=:effects)

# Stata: margins, at(means) dydx(age experience)
julia_equivalent = profile_margins(model, data, means_grid(data); type=:effects, vars=[:age, :experience])
```

### 5. **performance_comparison.jl** - Optimization Guide
**Purpose**: Performance benchmarking and optimization strategies.

**Key Features Demonstrated**:
- O(1) profile vs O(n) population scaling verification
- Memory allocation analysis
- Backend comparison (:fd vs :ad)
- Production optimization strategies
- Large dataset handling patterns

**Best For**: Users working with large datasets or performance-critical applications.

**Run Time**: ~2-3 minutes (includes benchmarking)

```julia
# Verify O(1) profile scaling
@benchmark profile_margins($model, $data, means_grid($data); type=:effects)

# Compare backends
population_margins(model, data; backend=:fd, type=:effects)  # Zero allocation
population_margins(model, data; backend=:ad, type=:effects)  # Higher accuracy
```

### 6. **economic_analysis.jl** - Advanced Applications
**Purpose**: Publication-ready econometric analysis workflows.

**Key Features Demonstrated**:
- Realistic economic datasets with complex interactions
- Wage determination analysis (Mincer equations)
- Elasticity analysis (`:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx`)
- Policy counterfactual analysis using categorical mixtures
- Binary outcome analysis (promotion probability)
- Robust standard errors framework

**Best For**: Researchers conducting publication-grade econometric analysis.

**Run Time**: ~20 seconds

```julia
# Elasticity analysis
result = population_margins(model, data; type=:effects, measure=:elasticity, vars=[:age, :experience])

# Policy counterfactuals
result = population_margins(model, data;
    scenarios=(education=mix("Bachelor's" => 0.60, "Graduate" => 0.40),),
    type=:predictions
)
```

## Reference Grid Builders

All profile examples demonstrate the reference grid system:

```julia
# Sample means with population composition for categoricals
means_grid(data)  

# Cartesian product of specified values
cartesian_grid(age=[25, 35, 45], education=["College", "Graduate"])

# Balanced factorial design
balanced_grid(data; education=:all, region=:all)  

# Quantile-based scenarios
quantile_grid(data; income=[0.25, 0.5, 0.75])

# Custom DataFrame
custom_grid = DataFrame(age=[30, 40], education=["College", "Graduate"])
```

## Running Examples

### Prerequisites
```julia
using Pkg
Pkg.add(["Margins", "DataFrames", "GLM", "StatsModels", "CategoricalArrays", "Random", "Statistics"])
```

### Execution
```bash
# Run individual examples
julia --project=. examples/basic_usage.jl
julia --project=. examples/economic_analysis.jl

# Or from Julia REPL
julia> include("examples/basic_workflow.jl")
```

### Dependencies by Example

| Example | Core Dependencies | Optional Dependencies |
|---------|-------------------|----------------------|
| basic_usage.jl | DataFrames, GLM, StatsModels, CategoricalArrays | - |
| basic_workflow.jl | DataFrames, GLM, Random, Statistics | - |
| margins_glm.jl | DataFrames, GLM, CategoricalArrays | CovarianceMatrices |
| stata_migration.jl | DataFrames, GLM, CategoricalArrays, Printf | - |
| performance_comparison.jl | DataFrames, GLM, BenchmarkTools | - |
| economic_analysis.jl | DataFrames, GLM, CategoricalArrays, Distributions | CovarianceMatrices, StatsPlots |

## Learning Path

**Recommended progression for new users**:

1. **Start here**: `basic_usage.jl` - Learn the 2×2 framework
2. **Build workflow**: `basic_workflow.jl` - See practical patterns  
3. **Choose specialization**:
   - **GLM focus**: `margins_glm.jl` - Binary/count outcomes
   - **Stata background**: `stata_migration.jl` - Direct translations
   - **Performance needs**: `performance_comparison.jl` - Optimization
   - **Advanced research**: `economic_analysis.jl` - Publication methods

## Getting Help

- **Quick questions**: Check the relevant example file
- **API reference**: See function docstrings (`?population_margins`)
- **Statistical methods**: See [Mathematical Foundation](../docs/src/mathematical_foundation.md) and [Computational Architecture](../docs/src/computational_architecture.md)
- **Performance**: Review performance_comparison.jl benchmarks or see [Performance Guide](../docs/src/performance.md)

## Contributing Examples

When adding new examples:

1. **Follow naming convention**: `topic_description.jl`
2. **Include purpose statement** and key features at the top
3. **Use clear documentation style** with proper explanations and comments
4. **Verify statistical correctness** before submission
5. **Test execution** on clean environment
6. **Update this EXAMPLES.md** with new entry

Examples should demonstrate **real-world workflows** rather than isolated features, helping users understand how to conduct complete analyses with Margins.jl.