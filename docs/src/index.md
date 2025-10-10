# Margins.jl

*Marginal effects for Julia*

## Overview

Margins.jl provides a rigorous computational framework for marginal effects analysis in econometric applications. The package implements a systematic approach to marginal effects computation through a conceptually unified framework that distinguishes between population-level and profile-specific analyses. The computational architecture achieves constant-time performance for profile analysis while maintaining statistical validity through comprehensive delta-method standard error computation.

The package integrates seamlessly with the established JuliaStats ecosystem, providing compatibility with [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) for model specification, [GLM.jl](https://github.com/JuliaStats/GLM.jl) for generalized linear models, and [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for robust standard errors. The implementation builds upon [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) to achieve efficient and mathematically precise marginal effect computation suitable for publication-grade econometric analysis.

## Implementation Overview

```julia
using Random
using CategoricalArrays, DataFrames, GLM, Margins

# Generate sample data
n = 1000
Random.seed!(06515)
df = DataFrame(
    y = randn(n),
    x1 = randn(n), 
    x2 = randn(n),
    group = categorical(rand(["A", "B", "C"], n))
)

# Fit model
model = lm(@formula(y ~ x1 + x2 + group), df)

# Population analysis: effects averaged across sample distribution
ame_result = population_margins(model, df; type=:effects)
DataFrame(ame_result)

# Profile analysis: effects at representative points
mem_result = profile_margins(model, df, means_grid(df); type=:effects)
DataFrame(mem_result)
```

## Methodological Framework

**Marginal Effects Framework Overview** 
This package addresses two core questions in marginal effects analysis:
1. **Effects**: "How much does Y change when I change X?" (like: "How much do wages increase per year of education?")
2. **Predictions**: "What value of Y should I expect for specific values of X?" (like: "What wage should I expect for someone with 16 years of education?")

The methodological foundation of Margins.jl rests upon a two-dimensional analytical framework that systematically addresses the fundamental questions arising in marginal effects analysis. This framework distinguishes between the evaluation context and the analytical target, thereby providing a comprehensive approach to econometric inference.

### Evaluation Context

The choice of evaluation context determines the distributional properties of the marginal effects estimates. Population analysis computes effects averaged across the observed sample distribution, yielding estimates that reflect the heterogeneity present in the data generating process. Profile analysis evaluates effects at specific covariate combinations, providing inference at representative or theoretically meaningful points in the covariate space.

### Analytical Targets  

The analytical target specifies the statistical quantity of interest within the chosen evaluation context. Effects analysis computes marginal effects through appropriate differentiation of the conditional expectation function, utilizing analytical derivatives for continuous variables and discrete contrasts for categorical variables. Predictions analysis evaluates adjusted predictions, providing fitted values that incorporate the full uncertainty structure of the estimated model.

### Analytical Framework Implementation

```julia
# Population Analysis: Sample Distribution Averaging
population_margins(model, data; type=:effects)      # Average Marginal Effects
population_margins(model, data; type=:predictions)  # Average Adjusted Predictions

# Profile Analysis: Representative Point Evaluation
profile_margins(model, data, means_grid(data); type=:effects)      # Effects at Sample Means
profile_margins(model, data, means_grid(data); type=:predictions)  # Predictions at Sample Means
```

## Computational Architecture and Statistical Properties

### Performance Characteristics

The computational implementation achieves constant-time complexity for profile analysis through optimized evaluation algorithms that scale independently of dataset size. Population analysis exhibits linear scaling with respect to sample size while maintaining minimal per-observation computational overhead through zero-allocation implementations built upon the FormulaCompiler.jl foundation. The architecture has been empirically validated across datasets ranging from small-scale experimental studies to large administrative datasets exceeding one million observations. Detailed performance analysis is provided in the [Performance Guide](performance.md).

### Statistical Inference Framework

Statistical inference employs rigorous delta-method standard error computation with full integration of the model's covariance matrix structure. The implementation prioritizes statistical validity over computational convenience, implementing an error-first policy whereby invalid statistical operations generate explicit errors rather than approximate results. All statistical computations have been validated through bootstrap comparison studies to ensure coverage probability accuracy suitable for econometric research applications and academic publication standards.

### Extended Analytical Capabilities

The package supports comprehensive elasticity analysis through parametric specification of effect measures, including standard elasticities and semi-elasticity variants for both dependent and independent variable transformations. Second differences (interaction effects) enable analysis of effect heterogeneity across moderator levels, addressing whether marginal effects vary with other covariates. Policy analysis applications are supported through categorical mixture specifications that enable realistic population composition modeling. The inference framework accommodates robust and clustered standard error computation through integration with CovarianceMatrices.jl, while flexible subgroup analysis capabilities facilitate stratified inference across multiple dimensions of heterogeneity. Comprehensive coverage of these advanced methodological features is provided in [Advanced Features](advanced.md) and [Second Differences](second_differences.md).

## Implementation Examples

### Profile Specification

Multiple ways to specify evaluation points:

```julia
# At sample means (most common)
profile_margins(model, data, means_grid(data); type=:effects)

# Custom scenarios
scenarios = cartesian_grid(x1=[0, 1, 2], group=["A", "B"]) 
profile_margins(model, data, scenarios; type=:effects)

# Complex realistic scenarios with categorical mixtures
using CategoricalArrays
mixture_grid = DataFrame(group=[mix("A" => 0.5, "B" => 0.3, "C" => 0.2)])
profile_margins(model, data, mixture_grid; type=:predictions)

# Pre-built reference grids (maximum control)
reference_grid = DataFrame(x1=[0, 1], x2=[0, 0], group=["A", "A"])  
profile_margins(model, data, reference_grid; type=:effects)
```

### Elasticity Analysis

```julia
# Population average elasticities
population_margins(model, data; type=:effects, measure=:elasticity)

# Elasticities at representative scenarios  
profile_margins(model, data, means_grid(data); type=:effects, measure=:elasticity)

# Semi-elasticities  
population_margins(model, data; measure=:semielasticity_dyex)  # change Y per % change X
population_margins(model, data; measure=:semielasticity_eydx)  # % change Y per unit X
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

### Second Differences (Interaction Effects)

```julia
# Compute AMEs across modifier levels
ames = population_margins(model, data; scenarios=(treated=[0,1],), type=:effects)

# Calculate second differences (does X's effect depend on treatment?)
sd = second_differences(ames, :age, :treated, vcov(model))
DataFrame(sd)

# Categorical moderators with multiple levels
ames = population_margins(model, data; scenarios=(education=["hs","college","grad"],), type=:effects)
sd = second_differences(ames, :income, :education, vcov(model))
```

## Statistical Validity and Methodological Standards

The package implements rigorous statistical principles designed to meet the standards required for econometric research and academic publication. Delta-method standard error computation incorporates the complete covariance matrix structure of the underlying statistical model, ensuring appropriate propagation of parameter uncertainty through the marginal effects calculations. 

The implementation follows an error-first philosophy whereby statistical operations that cannot guarantee validity generate explicit errors rather than potentially misleading approximate results. This approach prioritizes statistical correctness over computational convenience, reflecting the fundamental requirement that econometric software produce results suitable for peer-reviewed research applications.

All statistical procedures have undergone comprehensive validation through bootstrap comparison studies and theoretical verification, ensuring that confidence intervals, hypothesis tests, and other inferential procedures maintain appropriate coverage probabilities and Type I error rates across diverse model specifications and data characteristics.  

## Integration with JuliaStats

### Model Compatibility

The package provides comprehensive compatibility with models following the StatsModels.jl interface specifications. Support encompasses linear models, logistic regression, Poisson models, and other generalized linear model families through GLM.jl integration. Mixed effects modeling is accommodated through MixedModels.jl compatibility for both linear and generalized linear mixed model specifications. Extensions to custom model types are supported provided they implement the standard `coef()` and `vcov()` accessor methods.

### Data Integration Framework

Data handling utilizes the Tables.jl interface to ensure compatibility with diverse data sources including DataFrames, CSV files, and database result sets. The specialized result types (`EffectsResult` and `PredictionsResult`) implement the Tables.jl protocol to enable seamless conversion to DataFrame format for downstream analysis and reporting. This design provides maximum flexibility in data pipeline integration while maintaining type stability and computational efficiency.

### Robust Standard Errors
```julia
# Using CovarianceMatrices.jl
using CovarianceMatrices

# Robust standard errors (HC1)
population_margins(model, data; vcov=CovarianceMatrices.HC1)

# Clustered standard errors  
population_margins(model, data; vcov=CovarianceMatrices.Clustered(:firm_id))
```

## Computational Performance Analysis

### Constant-Time Profile Evaluation

Profile analysis achieves computational complexity independent of dataset size through optimized algorithms that evaluate marginal effects at specified covariate combinations without full dataset traversal. This constant-time property holds across diverse scenario specifications, enabling efficient analysis of complex policy counterfactuals and sensitivity analyses regardless of the underlying sample size.

```julia
# Profile margins exhibit O(1) complexity characteristics
@time profile_margins(model, small_data, means_grid(small_data))    # baseline timing
@time profile_margins(model, large_data, means_grid(large_data))    # identical complexity

# Complex scenario specifications maintain constant-time properties
scenarios = (x1=[0,1,2], x2=[10,20,30], group=["A","B"])  # 18 profiles
scenarios = cartesian_grid(x1=[0,1,2], x2=[10,20,30], group=["A","B"])  # 18 profiles
@time profile_margins(model, huge_data, scenarios)  # remains constant time
```

### Linear Scaling in Population Analysis

Population analysis exhibits optimal linear scaling characteristics with respect to sample size while maintaining minimal per-observation computational overhead through zero-allocation implementations. The computational architecture ensures predictable performance scaling suitable for large-scale econometric applications and administrative dataset analysis.

```julia
# Population margins demonstrate O(n) scaling with optimized per-row processing
@time population_margins(model, data_1k)    # baseline linear scaling
@time population_margins(model, data_10k)   # proportional computational cost
@time population_margins(model, data_100k)  # maintained efficiency at scale
```

## Documentation Organization

### Conceptual Foundation
- **[Mathematical Foundation](mathematical_foundation.md)**: Theoretical basis and statistical properties
- **[Second Differences](second_differences.md)**: Interaction effects and effect heterogeneity
- **[Comparison Guide](comparison.md)**: Methodological comparison with alternative approaches

### Implementation Reference
- **[API Reference](api.md)**: Complete function specifications and parameters
- **[Performance Guide](performance.md)**: Computational characteristics and benchmarks
- **[Examples](examples.md)**: Executable workflows and application demonstrations

### Migration and Integration
- **[Stata Migration](stata_migration.md)**: Command equivalence and workflow translation
- **[Advanced Features](advanced.md)**: Extended analytical capabilities

Technical support and bug reports should be directed to the [GitHub Issues](https://github.com/emfeltham/Margins.jl/issues) repository.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/emfeltham/Margins.jl")
```

**Requirements**: Julia ≥ 1.10

Note on documentation versions:
- The “dev” docs reflect the latest commits on `main` and update after each successful docs build.
- The “stable” docs update when a new tagged release is published. If you don’t see your recent changes on “stable”, check the “dev” docs or tag a release to promote changes to “stable”.

## Citation

If you use Margins.jl in your research, please cite:

```bibtex
@software{margins_jl,
  author = {Feltham, Eric M.},
  title = {Margins.jl: Marginal Effects and Adjusted Predictions for Julia Statistical Models},
  url = {https://github.com/emfeltham/Margins.jl},
  version = {2.0.0},
  year = {2025}
}
```

Margins.jl builds upon FormulaCompiler.jl for high-performance statistical computation. Please also cite:

```bibtex
@software{formulacompiler_jl,
  author = {Feltham, Eric M.},
  title = {FormulaCompiler.jl: High-Performance Formula Evaluation and Automatic Differentiation for Julia},
  url = {https://github.com/emfeltham/FormulaCompiler.jl},
  version = {1.0.0},
  year = {2025}
}
```
