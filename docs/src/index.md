# Margins.jl

*Marginal effects for Julia*

## Overview

Margins.jl provides a general and high-performance framework for marginal effects analysis in julia. The package is organized around both population- and profile-based approaches. The cost of profile analysis is dominated by the size of the reference grid rather than the dataset, and every result carries delta-method standard errors.

The package integrates seamlessly with the established JuliaStats ecosystem, providing compatibility with [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) for model specification, [GLM.jl](https://github.com/JuliaStats/GLM.jl) for generalized linear models, [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) for high-dimensional fixed effects, and [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for robust standard errors. The implementation builds upon [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl) to achieve efficient and accurate results.

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

The methodological foundation of Margins.jl rests upon a two-dimensional framework that hinges on the evaluation context and the analytical target.

### Evaluation Context

The choice of evaluation context determines the distributional properties of the marginal effects estimates. Population analysis computes effects averaged across the observed sample distribution, yielding estimates that reflect the heterogeneity present in the data generating process. Profile analysis evaluates effects at specific covariate combinations, providing inference at representative or theoretically meaningful points in the covariate space.

### Analytical Targets

The analytical target specifies the statistical quantity of interest within the chosen evaluation context. Effects analysis computes marginal effects through appropriate differentiation of the conditional expectation function, utilizing analytical derivatives for continuous variables and discrete contrasts for categorical variables. Predictions analysis evaluates adjusted predictions, providing fitted values that incorporate the full uncertainty structure of the estimated model.

### Analytical Framework Implementation

```julia
# Population Analysis: Sample Distribution Averaging
population_margins(model, data; type=:effects) # Average Marginal Effects
population_margins(model, data; type=:predictions) # Average Adjusted Predictions

# Profile Analysis: Representative Point Evaluation
profile_margins(model, data, means_grid(data); type=:effects) # Effects at Sample Means
profile_margins(model, data, means_grid(data); type=:predictions) # Predictions at Sample Means
```

## Computational Architecture and Statistical Properties

### Performance Characteristics

The cost of profile analysis is dominated by the number of reference-grid rows; each call also does a small amount of O(n) work (converting integer columns to `Float64`, checking for precomputed transforms of the variables it changes, and computing typical values over the estimation sample for variables the grid leaves out). Population analysis exhibits linear scaling with respect to sample size while maintaining minimal per-observation computational overhead through zero-allocation implementations built upon [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl). The architecture has been empirically validated across datasets ranging from small-scale experimental studies to large administrative datasets exceeding one million observations. Detailed performance analysis is provided in the [Performance Guide](performance.md).

### Statistical Inference Framework

Standard errors are calculated via the delta-method with full integration of the model's covariance matrix structure. Statistical computations have been validated with tests.

### Extended Analytical Capabilities

We include methods for second differences (interaction analyses) that allow researchers to smoothly analyze effect heterogeneity across moderator levels. The package supports elasticity analysis through specification of effect measures, and includes both standard elasticities and semi-elasticity variants for both dependent and independent variable transformations. We also support categorical mixture specifications for handling typical values of categorical and binary variables. Furthermore, the framework accommodates robust and clustered standard error computation through integration with CovarianceMatrices.jl. while flexible subgroup analysis capabilities facilitate stratified inference across multiple dimensions of heterogeneity.

cf. [Advanced Features](advanced.md) and [Second Differences](second_differences.md) for these and other features.

## Implementation Examples

### Profile Specification

Multiple ways to specify evaluation points:

```julia
# At sample means (most common)
profile_margins(model, data, means_grid(data); type = :effects)

# Custom scenarios
scenarios = cartesian_grid(x1 = [0, 1, 2], group = ["A", "B"]) 
profile_margins(model, data, scenarios; type=:effects)

# Complex realistic scenarios with categorical mixtures
using CategoricalArrays
mixture_grid = DataFrame(group = [mix("A" => 0.5, "B" => 0.3, "C" => 0.2)])
profile_margins(model, data, mixture_grid; type = :predictions)

# Pre-built reference grids (maximum control)
reference_grid = DataFrame(
  x1 = [0, 1], x2 = [0, 0], group = ["A", "A"]
)
profile_margins(model, data, reference_grid; type = :effects)
```

### Elasticity Analysis

```julia
# Population average elasticities
population_margins(model, data; type = :effects, measure = :elasticity)

# Elasticities at representative scenarios  
profile_margins(
  model, data, means_grid(data); type = :effects, measure = :elasticity
)

# Semi-elasticities:
# change Y per % change X
population_margins(model, data; type = :effects, measure = :semielasticity_dyex)
# % change Y per unit X
population_margins(model, data; type = :effects, measure = :semielasticity_eydx)  
```

### Subgroup Analysis

```julia
# Effects by categorical groups
population_margins(model, data; type = :effects, groups = :region)

# Multiple grouping variables
population_margins(
  model, data; type = :effects, groups = [:region, :year]
)

# Complex nested grouping
population_margins(
  model, data; type = :effects, groups = :region => (:income, 4)
)
```

### Second Differences (Interaction Effects)

```julia
# Does the effect of age depend on treatment?
sd = second_differences(model, data, :age, :treated)

# Categorical moderators with multiple levels: all pairwise comparisons
sd = second_differences(model, data, :income, :education)

# Continuous moderator: average ∂AME/∂income over the observed values, or at chosen values
sd = second_differences(model, data, :age, :income)
sd = second_differences(model, data, :age, :income; at = [20_000, 50_000])
```

## Integration with JuliaStats

### Model Compatibility

The package provides comprehensive compatibility with models following the StatsModels.jl interface specifications. Support encompasses linear models, logistic regression, Poisson models, and other generalized linear model families through GLM.jl integration. Mixed effects modeling is accommodated through MixedModels.jl compatibility for both linear and generalized linear mixed model specifications, and models with absorbed fixed effects through FixedEffectModels.jl (1.13.2 or later, and 2.x; see [Fixed Effects Models](fixed_effects.md)). Extensions to custom model types are supported provided they implement the standard `coef()` and `vcov()` accessor methods and rely on the `@formula` approach.

### Data Integration Framework

Data handling utilizes the Tables.jl interface to ensure compatibility with diverse data sources including DataFrames, CSV files, and database result sets. The specialized result types (`EffectsResult` and `PredictionsResult`) implement the Tables.jl protocol with conversion to DataFrame format for downstream analysis and reporting.

### Robust Standard Errors
```julia
# Using CovarianceMatrices.jl
using CovarianceMatrices

# Robust standard errors (HC1)
population_margins(model, data; vcov = HC1())

# Clustered standard errors
population_margins(
  model, data; vcov = CR1(data.firm_id)
)
```

## Computational Performance Analysis

### Profile Evaluation

Profile analysis evaluates the model only at the reference-grid rows, so its cost is dominated by the grid size. Each call still does a small amount of O(n) work on the data: integer columns are copied to `Float64`, the derived-column check scans the columns, and variables the grid leaves out are filled in with their (model-weighted) typical values over the estimation sample. On a 65-coefficient logit (Apple M4 Pro), profile predictions at 4 grid rows take 0.27, 0.59 and 1.8 ms at 5,000, 50,000 and 500,000 rows.

```julia
# Cost grows with the number of grid rows (here 18 profiles)
grid = cartesian_grid(x1 = [0, 1, 2], x2 = [10, 20, 30], group = ["A", "B"])
profile_margins(model, data, grid)
```

### Linear Scaling in Population Analysis

Population analysis exhibits linear scaling with respect to sample size; the per-row loops allocate nothing, so a call's allocations do not grow with n apart from per-call copies of columns that need conversion. On the same logit, the AME of all variables takes 18 ms, 182 ms and 1.8 s at 5,000, 50,000 and 500,000 rows. The computational architecture ensures predictable performance scaling suitable for large-scale statistical applications and administrative dataset analysis.

```julia
# Population margins demonstrate O(n) scaling with optimized
# per-row processing:
@time population_margins(model, data_1k) # baseline linear scaling
@time population_margins(model, data_10k) # proportional computational cost
@time population_margins(model, data_100k) # maintained efficiency at scale
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
Pkg.add("Margins")
```

**Requirements**: Julia ≥ 1.10

Note on documentation versions:
- The “dev” docs reflect the latest commits on `main` and update after each successful docs build.
- The “stable” docs update when a new tagged release is published. If you don’t see your recent changes on “stable”, check the “dev” docs or tag a release to promote changes to “stable”.

## Citation

If you use Margins.jl in your research, please cite:

```bibtex
@misc{feltham_formulacompilerjl_2026,
  title = {{FormulaCompiler}.Jl and {Margins}.Jl: {Efficient Marginal Effects} in {Julia}},
  shorttitle = {{FormulaCompiler}.Jl and {Margins}.Jl},
  author = {Feltham, Eric},
  year = {2026},
  month = jan,
  number = {arXiv:2601.07065},
  eprint = {2601.07065},
  primaryclass = {stat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2601.07065},
  archiveprefix = {arXiv},
  keywords = {Statistics - Computation},
}
```
