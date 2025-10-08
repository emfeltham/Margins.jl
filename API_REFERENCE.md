# Margins.jl API Reference

**Complete reference for the two-function analytical framework**

---

## Conceptual Overview

### Analysis Framework

Margins.jl structures marginal effects analysis around two orthogonal dimensions:

- **Population Analysis**: Estimates averaged across observed sample distribution
- **Profile Analysis**: Estimates evaluated at specific covariate combinations via explicit reference grids
- **Effects**: Marginal effects (derivatives and contrasts)
- **Predictions**: Adjusted predictions (fitted values)

### Function Organization

| Analysis Type | Effects | Predictions |
|---------------|---------|-------------|
| **Population** | Average Marginal Effects (AME) | Average Adjusted Predictions (AAP) |
| **Profile** | Effects at Representative Points (MEM/MER) | Predictions at Representative Points (APM/APR) |

---

## Function Reference

### `population_margins(model, data; kwargs...)`

**Population-level marginal effects and adjusted predictions**

Computes effects or predictions averaged across the observed sample distribution. This approach provides population-level estimates that reflect the actual composition and covariate distribution of the analysis sample.

**Arguments:**
- `model`: Fitted statistical model (GLM, LM, MixedModel, or any model supporting `coef()` and `vcov()`)
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)

**Keyword Arguments:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | `:effects` | `:effects` (AME) or `:predictions` (AAP) |
| `vars` | `nothing` | Variables for effects analysis (Symbol, Vector{Symbol}, or `:all_continuous`) |
| `scale` | `:response` | `:response` (response scale) or `:link` (linear predictor scale) |
| `backend` | `:ad` | `:ad` (automatic differentiation) or `:fd` (finite differences) |
| `measure` | `:effect` | `:effect`, `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx` |
| `scenarios` | `nothing` | Counterfactual scenarios (NamedTuple mapping variables to values) |
| `groups` | `nothing` | Subgroup analysis (Symbol, Vector{Symbol}, Tuple, or Pair) |
| `contrasts` | `:baseline` | `:baseline` or `:pairwise` contrasts for categorical variables |
| `ci_alpha` | `0.05` | Confidence interval alpha level |
| `vcov` | `GLM.vcov` | Covariance matrix function |
| `weights` | `nothing` | Sampling/frequency weights (Symbol, Vector, or nothing) |

**Returns:** `EffectsResult` or `PredictionsResult` with estimates, standard errors, and metadata

---

### `profile_margins(model, data, reference_grid; kwargs...)`

**Profile marginal effects and adjusted predictions**

Computes effects or predictions at specific covariate combinations defined by an explicit reference grid. This approach evaluates estimates at representative points or scenarios of substantive interest, facilitating interpretation at concrete covariate values.

**Arguments:**
- `model`: Fitted statistical model (GLM, LM, MixedModel, or any model supporting `coef()` and `vcov()`)
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)
- `reference_grid`: DataFrame defining covariate combinations for evaluation

**Keyword Arguments:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | `:effects` | `:effects` (MEM/MER) or `:predictions` (APM/APR) |
| `vars` | `nothing` | Variables for effects analysis (Symbol, Vector{Symbol}, or `:all_continuous`) |
| `scale` | `:response` | `:response` (response scale) or `:link` (linear predictor scale) |
| `backend` | `:ad` | `:ad` (automatic differentiation) or `:fd` (finite differences) |
| `measure` | `:effect` | `:effect`, `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx` |
| `contrasts` | `:baseline` | `:baseline` or `:pairwise` contrasts for categorical variables |
| `ci_alpha` | `0.05` | Confidence interval alpha level |
| `vcov` | `GLM.vcov` | Covariance matrix function |

**Returns:** `EffectsResult` or `PredictionsResult` with estimates, standard errors, profile information, and metadata

---

## Reference Grid Builders

### Grid Construction Functions

Margins.jl provides several functions to construct reference grids for `profile_margins()`:

#### `means_grid(data; typical=mean)`

**Sample means for continuous variables, frequency-weighted mixtures for categorical**

```julia
# Basic usage - sample means/frequencies
grid = means_grid(data)
result = profile_margins(model, data, grid)

# Custom typical value function
grid = means_grid(data; typical=median)
```

#### `cartesian_grid(; kwargs...)`

**Cartesian product of specified variable values**

```julia
# Explicit variable specifications
grid = cartesian_grid(x1=[0, 1, 2], x2=[10, 20])
result = profile_margins(model, data, grid)

# Mixed specifications
grid = cartesian_grid(treatment=[false, true], age=40:10:70)
```

#### `balanced_grid(data; kwargs...)`

**Equal probability mixtures for categorical variables**

```julia
# Balanced treatment of specified variables
grid = balanced_grid(data; region=:all, education=:all)
result = profile_margins(model, data, grid)

# Selective balancing
grid = balanced_grid(data; treatment=:all)
```

#### `quantile_grid(data; kwargs...)`

**Quantile-based grids for continuous variables**

```julia
# Quartile analysis
grid = quantile_grid(data; income=[0.25, 0.5, 0.75])
result = profile_margins(model, data, grid)

# Multiple variables
grid = quantile_grid(data; age=[0.1, 0.9], income=[0.25, 0.75])
```

#### `hierarchical_grid(data, spec; max_depth=5, warn_large=true)`

**Complex hierarchical reference grid specifications**

```julia
# Hierarchical scenarios
spec = Dict(:treatment => [false, true], 
           :age => [30, 50], 
           :region => :all)
grid = hierarchical_grid(data, spec)
result = profile_margins(model, data, grid)
```

#### `complete_reference_grid(reference_grid, model, data; typical=mean)`

**Complete partial reference grids with missing variables**

```julia
# Partial grid completion
partial_grid = DataFrame(treatment=[false, true])
complete_grid = complete_reference_grid(partial_grid, model, data)
result = profile_margins(model, data, complete_grid)
```

---

## Result Types

### `EffectsResult` and `PredictionsResult`

**Containers for marginal effects and prediction results with Tables.jl interface**

**Common Fields:**
- `estimates::Vector{Float64}`: Computed estimates (effects or predictions)
- `standard_errors::Vector{Float64}`: Delta-method standard errors
- `profile_values::Union{Nothing, NamedTuple}`: Reference grid values (profile_margins only)
- `group_values::Union{Nothing, NamedTuple}`: Grouping values (when using groups parameter)
- `gradients::Matrix{Float64}`: Parameter gradients for delta-method computation
- `metadata::Dict{Symbol, Any}`: Analysis options, model information, and technical details

**EffectsResult Additional Fields:**
- `variables::Vector{String}`: Variable names (the "x" in dy/dx)
- `terms::Vector{String}`: Contrast descriptions for categorical variables

**Methods:**
- `DataFrame(result)`: Convert to DataFrame for analysis and export
- Tables.jl interface: Works with CSV.write, DataFrames operations, etc.

**DataFrame Output Columns:**
- `estimate`: Computed marginal effect or prediction
- `se`: Standard error (delta-method)
- `variable`: Variable name (EffectsResult only)
- `term`: Contrast description (EffectsResult only) 
- Profile columns: Named according to reference grid variables (profile_margins only)
- Group columns: Named according to grouping variables (when using groups parameter)

### Categorical Variables and Contrasts

**Baseline Contrasts (default):**
For categorical variables, marginal effects represent contrasts relative to the baseline level automatically detected from the model's contrast coding.

**Pairwise Contrasts:**
When `contrasts=:pairwise` is specified, all pairwise comparisons between categorical levels are computed.

**Example:**
```julia
# Multi-level categorical with baseline contrasts
grid = cartesian_grid(age=[30, 50], education=["high_school", "college", "graduate"])
result = profile_margins(model, data, grid; type=:effects, vars=[:education])

# Computes baseline contrasts at each age:
# college vs high_school (baseline) at age 30 and 50
# graduate vs high_school (baseline) at age 30 and 50
```

---

## Parameter Guide

### Variable Selection (`vars` parameter)

| Value | Behavior |
|-------|----------|
| `nothing` | Auto-detect continuous variables (Int64, Float64, excluding Bool) |
| `:all_continuous` | Explicit specification of all continuous variables |
| `:x1` | Single variable |
| `[:x1, :x2]` | Multiple specific variables |

### Effect Measures (`measure` parameter)

| Measure | Description | Formula |
|---------|-------------|---------|
| `:effect` | Marginal effect (default) | `∂E[y]/∂x` |
| `:elasticity` | Elasticity | `(∂E[y]/∂x) × (x/E[y])` |
| `:semielasticity_dyex` | Semi-elasticity: d(y)/d(ln x) | `(∂E[y]/∂x) × x` |
| `:semielasticity_eydx` | Semi-elasticity: d(ln y)/dx | `(∂E[y]/∂x) × (1/E[y])` |

### Computational Backends (`backend` parameter)

| Backend | Performance | Accuracy | Use Case |
|---------|-------------|----------|----------|
| `:ad` | Small allocation | Highest | Default (recommended) |
| `:fd` | Zero allocation | High | Production, large datasets |

### Scale Parameters (`scale` parameter)

| Scale | Description | GLM Behavior |
|-------|-------------|--------------|
| `:response` | Response scale (default) | Applies inverse link function |
| `:link` | Linear predictor scale | Direct linear predictor effects |

### Contrast Types (`contrasts` parameter)

| Type | Description | Example Output |
|------|-------------|----------------|
| `:baseline` | Contrasts vs reference level (default) | college - high_school, graduate - high_school |
| `:pairwise` | All pairwise comparisons | college - high_school, graduate - high_school, graduate - college |

### Grouping Specifications (`groups` parameter)

| Specification | Description | Example |
|---------------|-------------|---------|
| `Symbol` | Single grouping variable | `groups=:region` |
| `Vector{Symbol}` | Multiple grouping variables | `groups=[:region, :year]` |
| `Tuple` | Nested grouping | `groups=(:region, :year)` |
| `Pair` | Named grouping | `groups=:analysis => :region` |

---

## Usage Patterns

### Basic Workflow

```julia
using Margins, DataFrames, GLM

# 1. Fit model
model = lm(@formula(y ~ x1 + x2 + group), data)

# 2. Population analysis (most common)
ame = population_margins(model, data)                        # Average marginal effects
aap = population_margins(model, data; type=:predictions)     # Average predictions

# 3. Profile analysis using reference grids
grid = means_grid(data)                                      # Sample means/frequencies
mem = profile_margins(model, data, grid)                     # Effects at sample means
apm = profile_margins(model, data, grid; type=:predictions)  # Predictions at means

# 4. Convert results
DataFrame(ame)  # Standard DataFrame with results
```

### Reference Grid Workflows

```julia
# Sample means and frequencies
grid = means_grid(data)
result = profile_margins(model, data, grid)

# Explicit scenarios
grid = cartesian_grid(treatment=[false, true], age=[30, 50, 70])
result = profile_margins(model, data, grid)

# Quantile analysis
grid = quantile_grid(data; income=[0.25, 0.5, 0.75])
result = profile_margins(model, data, grid)

# Balanced categorical combinations
grid = balanced_grid(data; region=:all, education=:all)
result = profile_margins(model, data, grid)

# Custom reference grid
grid = DataFrame(x1=[0, 1, 2], x2=repeat([10], 3), treatment=[false, true, false])
result = profile_margins(model, data, grid)
```

### Advanced Analysis Patterns

```julia
# Elasticity analysis
elasticities = population_margins(model, data; measure=:elasticity)
grid = means_grid(data)
profile_elasticities = profile_margins(model, data, grid; measure=:elasticity)

# Scenario analysis with multiple variables
grid = cartesian_grid(treatment=[false, true], income=[25000, 50000, 75000])
scenarios = profile_margins(model, data, grid)

# Subgroup analysis (population only)
by_region = population_margins(model, data; groups=:region)  
by_multiple = population_margins(model, data; groups=[:region, :year])

# High-performance production use
fast_ame = population_margins(model, large_data; backend=:fd, scale=:link)
```

### Categorical Variable Handling

```julia
# Automatic frequency-weighted defaults in means_grid
grid = means_grid(data)
result = profile_margins(model, data, grid)
# → Continuous vars: sample mean
# → Categorical vars: frequency-weighted mixtures based on data composition

# Explicit categorical scenarios
grid = cartesian_grid(treatment=[false, true], region=["urban", "rural"])
scenario = profile_margins(model, data, grid)

# Mixed with CategoricalMixture (if implemented)
# grid = cartesian_grid(group=mix("control" => 0.3, "treatment" => 0.7))
# mixed_scenario = profile_margins(model, data, grid)
```

---

## Performance Guide

### Dataset Size Recommendations

| Size | Population Margins | Profile Margins | Recommended Backend |
|------|-------------------|-----------------|---------------------|
| < 10k | Excellent | Excellent | `:ad` (default) |
| 10k-100k | Production-ready | Excellent | `:ad` or `:fd` |
| 100k-1M | Good | Excellent | `:ad` or `:fd` |
| > 1M | Scales appropriately | Constant time | `:ad` or `:fd` |

### Performance Characteristics

- **Profile margins**: O(1) constant time regardless of dataset size (~100-200μs)
- **Population margins**: O(n) scaling with optimized per-row costs (~150ns/row effects, ~10ns/row predictions)
- **Memory usage**: Constant allocation footprint for production workloads

### Optimization Tips

```julia
# Maximum performance for large datasets
result = population_margins(model, data; backend=:fd, target=:eta)

# Profile analysis is always fast
result = profile_margins(model, data; at=Dict(...))  # ~200μs regardless of data size

# Pre-build reference grids for repeated analysis
grid = DataFrame(x1=[0,1,2], x2=[10,20,30])
result = profile_margins(model, grid; type=:predictions)
```

---

## Statistical Guarantees

### **Statistical Correctness**
- **Delta-method standard errors**: Full covariance matrix integration
- **Zero tolerance policy**: Errors rather than invalid approximations  
- **Bootstrap validated**: All computations verified against bootstrap estimates
- **Publication-grade**: Ready for econometric research and academic publication

### **Categorical Effects**
- **Baseline contrasts**: Proper reference level comparisons
- **Frequency weighting**: Realistic population composition in profiles
- **Mixed scenarios**: Support for fractional categorical values

### **GLM Integration**  
- **All families**: Gaussian, Binomial, Poisson, Gamma support
- **Link functions**: Automatic scale handling (μ vs η)
- **Robust covariance**: CovarianceMatrices.jl integration

---

## Integration Examples

### Stata Migration

| Stata Command | Margins.jl Equivalent |
|---------------|----------------------|
| `margins, dydx(*)` | `population_margins(model, data)` |
| `margins, at(means) dydx(*)` | `profile_margins(model, data, means_grid(data))` |
| `margins, at(x1=(0 1 2))` | `profile_margins(model, data, cartesian_grid(x1=[0,1,2]))` |
| `margins` | `population_margins(model, data; type=:predictions)` |
| `margins, at(means)` | `profile_margins(model, data, means_grid(data); type=:predictions)` |
| `margins, over(group)` | `population_margins(model, data; groups=:group)` |

### Workflow Integration

```julia
# With CSV.jl
using CSV
data = CSV.read("mydata.csv", DataFrame)
results = population_margins(model, data)
CSV.write("margins_results.csv", DataFrame(results))

# With CovarianceMatrices.jl for robust SEs
using CovarianceMatrices
robust_model = glm(formula, data, family, vcov=HC1())
robust_results = population_margins(robust_model, data)

# With plotting
using Plots
grid = cartesian_grid(x1=-2:0.1:2)
results_df = DataFrame(profile_margins(model, data, grid))
plot(results_df.x1, results_df.estimate)

# Mixed model integration
using MixedModels
mixed_model = fit(MixedModel, @formula(y ~ x1 + x2 + (1|group)), data)
mixed_results = population_margins(mixed_model, data)
```

---

*Margins.jl v1.0 - Statistical rigor meets Julia performance*