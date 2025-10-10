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

See also: [Population Scenarios](population_scenarios.md) for counterfactual analysis and [Weights](weights.md) for sampling/frequency weights.

### Profile Analysis

### `profile_margins`

Computes marginal effects or adjusted predictions evaluated at specified covariate combinations within the covariate space.

The function implements profile-specific inference through evaluation of marginal quantities at predetermined points in the covariate space, typically at sample means or theoretically motivated scenario specifications. This approach yields concrete, interpretable estimates for specific covariate combinations while maintaining appropriate uncertainty quantification through delta-method standard error computation.

**Methodological Applications:**  
Profile analysis provides representative case inference suitable for policy targeting specific demographic or economic profiles. The approach facilitates clear communication of quantitative results through concrete scenario interpretation, making it particularly valuable for stakeholder communication and policy applications requiring specific target group analysis.

**Computational Characteristics**: Constant-time complexity independent of dataset size through optimized evaluation algorithms that avoid full dataset traversal.
*Comprehensive performance benchmarking and memory allocation analysis are detailed in the [Performance Guide](performance.md)*

## Result Type Specifications

### Type-Safe Result System (v2.0)

Margins.jl implements a specialized type system that provides type safety and optimized DataFrame formatting through distinct result containers for different analysis types.

### `EffectsResult`

Structured container for marginal effects analysis (AME, MEM, MER) implementing the Tables.jl interface protocol.

The `EffectsResult` type encapsulates computed marginal effects along with associated statistical inference quantities including standard errors, confidence intervals, and hypothesis test statistics. The type contains variable identification fields (`variables`, `terms`) essential for effects interpretation and supports multiple DataFrame formatting options.

**Fields:**
- `estimates::Vector{Float64}` - Point estimates of marginal effects
- `standard_errors::Vector{Float64}` - Delta-method standard errors
- `variables::Vector{String}` - The "x" in dy/dx (which variable each row represents)
- `terms::Vector{String}` - Contrast descriptions (e.g., "continuous", "treated vs control")
- `profile_values::Union{Nothing, NamedTuple}` - Reference grid values (for profile effects MEM/MER; `nothing` for population effects AME)
- `group_values::Union{Nothing, NamedTuple}` - Grouping variable values (when using `groups` parameter; `nothing` otherwise)
- `gradients::Matrix{Float64}` - Parameter gradients (G matrix) for delta-method computation
- `metadata::Dict{Symbol, Any}` - Analysis metadata (model info, options used, sample size, etc.)

**Key Features:**
- Multiple DataFrame formats: `:standard`, `:compact`, `:confidence`, `:profile`, `:stata`
- Auto-detects appropriate format based on analysis type
- `profile_values` populated only for `profile_margins()` (MEM/MER)
- `group_values` populated only when using `groups` parameter

### `PredictionsResult`

Streamlined container for predictions analysis (AAP, APM, APR) implementing the Tables.jl interface protocol.

The `PredictionsResult` type focuses specifically on predicted values without variable/contrast concepts, providing a clean interface optimized for predictions analysis. The streamlined design reflects that predictions represent "fitted values at scenarios" rather than "effects of variables."

**Fields:**
- `estimates::Vector{Float64}` - Point estimates (predicted values)
- `standard_errors::Vector{Float64}` - Delta-method standard errors
- `profile_values::Union{Nothing, NamedTuple}` - Reference grid values (for profile predictions APM/APR; `nothing` for population predictions AAP)
- `group_values::Union{Nothing, NamedTuple}` - Grouping variable values (when using `groups` parameter; `nothing` otherwise)
- `gradients::Matrix{Float64}` - Parameter gradients (G matrix) for delta-method computation
- `metadata::Dict{Symbol, Any}` - Analysis metadata (model info, options used, sample size, etc.)

**Key Features:**
- Omits variable/contrast fields (not applicable to predictions - predictions don't have "x" or "dy/dx" concepts)
- Single optimized DataFrame format for predictions display
- Clean tabular output focused on prediction values and statistics
- `profile_values` populated only for `profile_margins()` (APM/APR)
- `group_values` populated only when using `groups` parameter

**Data Integration Framework:**
```julia
# Type-specific result containers with Tables.jl protocol
effects_result = population_margins(model, data; type=:effects)  # Returns EffectsResult
predictions_result = population_margins(model, data; type=:predictions)  # Returns PredictionsResult

# Accessing fields directly
effects_result.estimates          # Vector{Float64} of marginal effects
effects_result.standard_errors    # Vector{Float64} of standard errors
effects_result.variables          # Vector{String} of variable names
effects_result.profile_values     # Nothing (for population) or NamedTuple (for profile)
effects_result.group_values       # Nothing (no groups) or NamedTuple (with groups)
effects_result.metadata           # Dict{Symbol, Any} with analysis info

# Profile margins have profile_values populated
profile_result = profile_margins(model, data, means_grid(data); type=:effects)
profile_result.profile_values     # NamedTuple(x1=[...], x2=[...], ...)

# Grouped analysis has group_values populated
grouped_result = population_margins(model, data; type=:effects, groups=:region)
grouped_result.group_values       # NamedTuple(region=["North", "South", ...])

# Type-specific DataFrame conversion
effects_df = DataFrame(effects_result)  # Includes variable/contrast columns
predictions_df = DataFrame(predictions_result)  # Streamlined predictions format

# Multiple format options for effects
DataFrame(effects_result; format=:compact)  # Minimal columns
DataFrame(effects_result; format=:stata)    # Stata-style column names

# Compatible with all Tables.jl-compliant output formats
CSV.write("effects.csv", effects_result)
CSV.write("predictions.csv", predictions_result)
```

## Second Differences (Interaction Effects)

Margins.jl provides comprehensive support for computing second differences—interaction effects on the predicted outcome scale. Second differences quantify how marginal effects vary across levels of a moderating variable, addressing the fundamental question: **"Does the effect of X depend on Z?"**

### Quick Start

```julia
# Step 1: Compute AMEs across modifier levels
ames = population_margins(model, data;
                         scenarios=(treated=[0, 1],),
                         type=:effects)

# Step 2: Calculate second differences
sd = second_differences(ames, :age, :treated, vcov(model))
DataFrame(sd)
```

### Available Functions

**Discrete Contrast Approach** (Population-based):
- **`second_differences()`**: Unified interface (recommended) - handles binary, categorical, and continuous moderators
- **`second_difference()`**: Binary moderators only (backward compatibility)
- **`second_differences_pairwise()`**: All pairwise modifier comparisons
- **`second_differences_all_contrasts()`**: All focal contrasts × all modifier pairs

**Local Derivative Approach** (Profile-based):
- **`second_differences_at()`**: Compute ∂AME/∂modifier at specific evaluation points via finite differences

For comprehensive coverage including methodological foundation, usage patterns, and interpretation guidance, see [Second Differences](second_differences.md).

## Extended Analytical Capabilities

### Categorical Mixture Specifications

The package implements sophisticated categorical mixture functionality to enable realistic policy scenario analysis through fractional category specifications. The `CategoricalMixture` type facilitates the specification of probability-weighted categorical distributions that reflect realistic population compositions rather than arbitrary baseline categories.

**Policy Counterfactual Analysis:**
```julia
# Current population educational composition (predictions at a mixture)
baseline_grid = DataFrame(education=[mix("HS" => 0.4, "College" => 0.4, "Graduate" => 0.2)])
baseline = profile_margins(model, data, baseline_grid; type=:predictions)

# Policy counterfactual: educational attainment improvement (new mixture)
intervention_grid = DataFrame(education=[mix("HS" => 0.2, "College" => 0.5, "Graduate" => 0.3)])
intervention = profile_margins(model, data, intervention_grid; type=:predictions)
```

## Parameter Reference

### Common Parameters

**Quick Start Examples**:
- `type=:effects` → "How much does the outcome change?" (most common)  
- `type=:predictions` → "What outcome value should I expect?"
- `measure=:elasticity` → "What's the percentage effect?" (useful for proportional changes)
- `backend=:ad` → Use default (most accurate, zero allocation)
- `backend=:fd` → Alternative backend (legacy compatibility)

All main functions support these core parameters:

#### Analysis Type (`type`)
- `:effects` - Marginal effects (derivatives for continuous, contrasts for categorical)
- `:predictions` - Adjusted predictions (fitted values)

#### Variable Selection (`vars`)
- `nothing` - Auto-detect continuous variables (default for effects)
- `:all_continuous` - Explicit selection of all continuous variables
- `:variable_name` - Single variable
- `[:var1, :var2]` - Multiple specific variables

#### Target Scale (`scale`)
- `:response` - Response scale (default, applies inverse link function)
- `:link` - Linear predictor scale (link scale)

#### Computational Backend (`backend`)
- `:ad` - Automatic differentiation (default; higher accuracy, zero allocation after warmup)
- `:fd` - Finite differences (zero allocation, production-ready)

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
profile_margins(model, data, cartesian_grid(x=[0,1,2], group=["A","B"]))

# Hierarchical grid construction using group grammar
reference_spec = :region => [(:income, :quartiles), (:age, :mean)]
profile_margins(model, data, hierarchical_grid(data, reference_spec))

# Deep hierarchical nesting for complex policy analysis
policy_spec = :country => (:region => (:education => [(:income, :quintiles), (:age, :mean)]))
profile_margins(model, data, hierarchical_grid(data, policy_spec; max_depth=4))

# DataFrame grid (full control)
grid = DataFrame(x=[0,1,2], group=["A","A","B"])
profile_margins(model, data, grid)
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
population_margins(model, data; scenarios=(treatment=[0, 1]), type=:effects)
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
mem = profile_margins(model, data, means_grid(data))
scenarios = profile_margins(model, data, cartesian_grid(x1=[0,1,2]))

# 4. Convert to DataFrame for analysis
DataFrame(ame)
```

### Performance Optimization
```julia
# Maximum performance configuration
fast_result = population_margins(model, data; backend=:fd, scale=:link)

# Profile analysis is O(1) - efficient regardless of data size
scenarios = (var1=[-2,-1,0,1,2], var2=["A","B","C"])  # 15 scenarios
scenarios = cartesian_grid(x1=[0,1,2])
profile_result = profile_margins(model, huge_data, scenarios)  # ~300μs regardless of data size
```

### Advanced Analysis Patterns
```julia
# Elasticity analysis across scenarios (profile)
scenarios = cartesian_grid(x1=[0, 1, 2])
elasticities = profile_margins(model, data, scenarios; 
    measure=:elasticity, vars=[:x2])

# Robust standard errors (with CovarianceMatrices.jl)
using CovarianceMatrices
robust_effects = population_margins(model, data; vcov=CovarianceMatrices.HC1)

# Complex categorical scenarios via reference grid
policy_grid = DataFrame(
    treatment=[mix(0 => 0.3, 1 => 0.7)],           # 70% treatment rate
    education=[mix("HS" => 0.3, "College" => 0.7)] # Education composition
)
policy_scenario = profile_margins(model, data, policy_grid; type=:predictions)
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
# Error: Invalid reference grid argument (must be DataFrame or a grid builder output)
profile_margins(model, data, "invalid")
# → Clear guidance on valid reference grid specifications

# Error: Reference grid missing model variables
incomplete_grid = DataFrame(x1=[0,1])  # Missing x2, group from model
profile_margins(model, data, incomplete_grid)
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

# Apply different estimators via vcov parameter
ame_hc1 = population_margins(model, data; vcov=CovarianceMatrices.HC1)
ame_hc3 = population_margins(model, data; vcov=CovarianceMatrices.HC3)
ame_clustered = population_margins(model, data; vcov=CovarianceMatrices.Clustered(:cluster_var))
ame_hac = population_margins(model, data; vcov=CovarianceMatrices.HAC(kernel=:bartlett))
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
