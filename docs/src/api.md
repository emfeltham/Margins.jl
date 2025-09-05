# API Reference

*Comprehensive technical specification for Margins.jl functions and types*

## Conceptual Foundation

### Two-Function Architecture

The package implements a systematic two-function API that operationalizes the unified analytical framework through distinct computational pathways for population-level and profile-specific marginal effects analysis.

### Analysis Type Distinction

- **Population Analysis**: Integration over empirical covariate distributions
- **Profile Analysis**: Evaluation at specified covariate combinations

## Function Specifications

### Population Analysis

### `population_margins`

Computes population-level marginal effects or adjusted predictions through integration over the empirical distribution of observed covariates.

The function implements population-averaged inference by computing marginal quantities for each observation in the sample and subsequently averaging these quantities according to the empirical distribution. This approach yields population parameters that reflect the heterogeneity present in the data generating process while providing appropriate standard errors through delta-method computation with full covariance matrix integration.

**Methodological Applications:**
Population analysis provides unbiased estimates of population parameters suitable for policy evaluation requiring external validity to similar populations. The approach proves particularly valuable when sample heterogeneity represents important features of the underlying population, and when analytical applications affect diverse demographic or economic groups requiring representative inference.

**Computational Characteristics**: Linear scaling with respect to sample size while maintaining minimal per-observation computational overhead through optimized implementations.
*Detailed performance analysis and computational complexity comparisons are provided in the [Performance Guide](performance.md)*

### Profile Analysis

### `profile_margins`

Computes marginal effects or adjusted predictions evaluated at specified covariate combinations within the covariate space.

The function implements profile-specific inference through evaluation of marginal quantities at predetermined points in the covariate space, typically at sample means or theoretically motivated scenario specifications. This approach yields concrete, interpretable estimates for specific covariate combinations while maintaining appropriate uncertainty quantification through delta-method standard error computation.

**Methodological Applications:**  
Profile analysis provides representative case inference suitable for policy targeting specific demographic or economic profiles. The approach facilitates clear communication of quantitative results through concrete scenario interpretation, making it particularly valuable for stakeholder communication and policy applications requiring specific target group analysis.

**Computational Characteristics**: Constant-time complexity independent of dataset size through optimized evaluation algorithms that avoid full dataset traversal.
*Comprehensive performance benchmarking and memory allocation analysis are detailed in the [Performance Guide](performance.md)*

## Result Type Specifications

### `MarginsResult`

Structured container for marginal effects analysis results implementing the Tables.jl interface protocol.

The result type encapsulates computed marginal effects or adjusted predictions along with associated statistical inference quantities including standard errors, confidence intervals, and hypothesis test statistics. The implementation adheres to the Tables.jl protocol to ensure seamless integration with the broader data analysis ecosystem while maintaining type stability and computational efficiency.

**Data Integration Framework:**
```julia
# Tables.jl protocol implementation enables direct conversion
result = population_margins(model, data)
df = DataFrame(result)

# Compatible with all Tables.jl-compliant output formats
CSV.write("results.csv", result)
```

## Extended Analytical Capabilities

### Categorical Mixture Specifications

The package implements sophisticated categorical mixture functionality to enable realistic policy scenario analysis through fractional category specifications. The `CategoricalMixture` type facilitates the specification of probability-weighted categorical distributions that reflect realistic population compositions rather than arbitrary baseline categories.

**Policy Counterfactual Analysis:**
```julia
# Current population educational composition
baseline = profile_margins(model, data; 
    at=Dict(:education => mix("HS" => 0.4, "College" => 0.4, "Graduate" => 0.2)))

# Policy counterfactual: educational attainment improvement
intervention = profile_margins(model, data;
    at=Dict(:education => mix("HS" => 0.2, "College" => 0.5, "Graduate" => 0.3)))
```

## Parameter Reference

### Common Parameters

All main functions support these core parameters:

#### Analysis Type (`type`)
- `:effects` - Marginal effects (derivatives for continuous, contrasts for categorical)
- `:predictions` - Adjusted predictions (fitted values)

#### Variable Selection (`vars`)
- `nothing` - Auto-detect continuous variables (default for effects)
- `:all_continuous` - Explicit selection of all continuous variables
- `:variable_name` - Single variable
- `[:var1, :var2]` - Multiple specific variables

#### Target Scale (`target`)
- `:mu` - Response scale (default, applies inverse link function)
- `:eta` - Linear predictor scale (link scale)

#### Computational Backend (`backend`)
- `:auto` - Context-dependent selection (default)
- `:fd` - Finite differences (zero allocation, production-ready)
- `:ad` - Automatic differentiation (higher accuracy, small allocation)

#### Effect Measures (`measure`)
- `:effect` - Standard marginal effects (default)
- `:elasticity` - Elasticities (% change in Y per % change in X)
- `:semielasticity_dyex` - Semielasticity d(y)/d(ln x) (change in Y per % change in X)
- `:semielasticity_eydx` - Semielasticity d(ln y)/dx (% change in Y per unit change in X)

### Profile-Specific Parameters

#### Profile Specification (`at`)
- `:means` - Effects/predictions at sample means
- `Dict(:var => [val1, val2])` - Cartesian product specification
- `[Dict(:var => val1), Dict(:var => val2)]` - Explicit profile list
- `DataFrame` - Pre-built reference grid (maximum control)

**Examples:**
```julia
# At sample means (most common)
profile_margins(model, data, means_grid(data))

# Cartesian product: 6 scenarios (3×2)
profile_margins(model, data; at=Dict(:x => [0,1,2], :group => ["A","B"]))

# Explicit profiles  
profiles = [Dict(:x => 0, :group => "A"), Dict(:x => 1, :group => "B")]
profile_margins(model, data; at=profiles)

# DataFrame grid (full control)
grid = DataFrame(x=[0,1,2], group=["A","A","B"])
profile_margins(model, grid)
```

### Population-Specific Parameters

#### Grouping (`over`)
- `Symbol` - Single grouping variable
- `Vector{Symbol}` - Multiple grouping variables  
- `NamedTuple` - Advanced grouping with value specifications

**Examples:**
```julia
# By single categorical variable
population_margins(model, data; groups=:region)

# Multiple grouping
population_margins(model, data; groups=[:region, :year])

# Advanced grouping (unified syntax)
population_margins(model, data; groups=(:income, [20000, 50000, 80000]))
```

#### Counterfactual Analysis (`scenarios`)
```julia
# Effects when treatment is set to 1 vs 0 for entire population
population_margins(model, data; scenarios=Dict(:treatment => [0, 1]), type=:effects)
```

## Usage Patterns

### Basic Workflow
```julia
# 1. Fit model
model = lm(@formula(y ~ x1 + x2 + group), data)

# 2. Population analysis (most common starting point)
ame = population_margins(model, data)
aap = population_margins(model, data; type=:predictions)

# 3. Profile analysis for specific scenarios
mem = profile_margins(model, data; at=:means)
scenarios = profile_margins(model, data; at=Dict(:x1 => [0,1,2]))

# 4. Convert to DataFrame for analysis
DataFrame(ame)
```

### Performance Optimization
```julia
# Maximum performance configuration
fast_result = population_margins(model, data; backend=:fd, scale=:link)

# Profile analysis is O(1) - efficient regardless of data size
scenarios = Dict(:var1 => [-2,-1,0,1,2], :var2 => ["A","B","C"])  # 15 scenarios
profile_result = profile_margins(model, huge_data; at=scenarios)  # ~300μs regardless of data size
```

### Advanced Analysis Patterns
```julia
# Elasticity analysis across scenarios
elasticities = profile_margins(model, data; 
    at=Dict(:x1 => [0, 1, 2]), 
    measure=:elasticity,
    vars=[:x2])

# Robust standard errors (with CovarianceMatrices.jl)
robust_model = glm(formula, data, family, vcov=HC1())
robust_effects = population_margins(robust_model, data)

# Complex categorical scenarios
policy_scenario = profile_margins(model, data;
    at=Dict(
        :treatment => mix(0 => 0.3, 1 => 0.7),           # 70% treatment rate
        :education => mix("HS" => 0.3, "College" => 0.7)  # Education composition
    ),
    type=:predictions
)
```

## Error Handling

### Common Error Patterns

#### Variable Specification Errors
```julia
# Error: Variable not found
population_margins(model, data; vars=[:nonexistent_var])
# → Clear error message with available variables

# Error: Wrong variable type for effects  
population_margins(model, data; vars=[:categorical_var], type=:effects)
# → Suggests using categorical contrasts or predictions
```

#### Profile Specification Errors
```julia  
# Error: Invalid at parameter
profile_margins(model, data; at="invalid")
# → Clear guidance on valid at specifications

# Error: Reference grid missing model variables
incomplete_grid = DataFrame(x1=[0,1])  # Missing x2, group from model
profile_margins(model, incomplete_grid)
# → Error with list of missing variables
```

#### Statistical Validity Errors
```julia
# Error: Insufficient data for robust estimation
tiny_data = data[1:5, :]
population_margins(model, tiny_data)
# → Warning about statistical reliability with small samples
```

### Error Recovery Patterns
```julia
# Backend fallback
function robust_margins(model, data; kwargs...)
    try
        return population_margins(model, data; backend=:ad, kwargs...)
    catch e
        @warn "AD backend failed, falling back to FD"
        return population_margins(model, data; backend=:fd, kwargs...)
    end
end

# Input validation
function validated_margins(model, data; vars=nothing, kwargs...)
    # Validate variable existence
    if vars !== nothing
        data_vars = names(data)
        missing_vars = setdiff(vars, Symbol.(data_vars))
        if !isempty(missing_vars)
            throw(ArgumentError("Variables not found in data: $missing_vars"))
        end
    end
    
    return population_margins(model, data; vars=vars, kwargs...)
end
```

## Integration Examples

### With GLM.jl Ecosystem
```julia
using GLM, CategoricalArrays

# Logistic regression
model = glm(@formula(outcome ~ x1 + x2 + group), data, Binomial(), LogitLink())

# Effects on probability scale
prob_effects = population_margins(model, data; scale=:response, type=:effects)

# Effects on log-odds scale  
logodds_effects = population_margins(model, data; scale=:link, type=:effects)
```

### With CovarianceMatrices.jl
```julia
using CovarianceMatrices

# Various robust estimators
models = [
    glm(formula, data, family, vcov=HC1()),      # Heteroskedasticity-robust
    glm(formula, data, family, vcov=HC3()),      # High-leverage robust  
    glm(formula, data, family, vcov=Clustered(:cluster_var)),  # Clustered
    glm(formula, data, family, vcov=HAC(kernel=:bartlett))     # HAC
]

# All margin computations automatically use model's covariance
results = [population_margins(m, data) for m in models]
```

### With DataFrames Ecosystem
```julia
using DataFrames, CSV, Chain

# Complete analysis pipeline
results_df = @chain begin
    population_margins(model, data; type=:effects)
    DataFrame(_)
    select(_, :term, :estimate, :se, :p_value)
    filter(row -> row.p_value < 0.05, _)  # Significant effects only
end

# Export results
CSV.write("significant_effects.csv", results_df)
```

---

*This API reference provides complete documentation for all Margins.jl functionality. For conceptual background on the 2×2 framework, see [Mathematical Foundation](mathematical_foundation.md). For performance optimization guidance, see [Performance Guide](performance.md). For advanced features including elasticities and robust inference, see [Advanced Features](advanced.md).*