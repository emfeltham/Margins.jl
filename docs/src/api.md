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
- `terms::Vector{String}` - Contrast descriptions (e.g., "dy/dx", "treated - control")
- `profile_values::Union{Nothing, NamedTuple}` - Per-row values of the context columns: the reference-grid variables for `profile_margins()` (MEM/MER), and the `scenarios=` and `groups=` variables for `population_margins()`. `nothing` for a plain population result (AME). `metadata[:groups_vars]` and `metadata[:scenarios_vars]` record which of these columns are groups and which are scenarios (`Margins.context_columns(result)` returns them)
- `group_values::Union{Nothing, NamedTuple}` - Reserved; currently always `nothing`. Group columns from the `groups` parameter are stored in `profile_values`
- `gradients::Matrix{Float64}` - Parameter gradients (G matrix) for delta-method computation
- `metadata::Dict{Symbol, Any}` - Analysis metadata (model info, options used, sample size, etc.)

**Key Features:**
- Multiple DataFrame formats: `:standard`, `:compact`, `:confidence`, `:profile`, `:stata`
- Auto-detects appropriate format based on analysis type
- `profile_values` holds the reference grid (`profile_margins()`) or the scenario and group columns (`population_margins()` with `scenarios=` or `groups=`); `nothing` for a plain population result
- Group columns are in `profile_values` (not `group_values`); `metadata[:groups_vars]` identifies them

### `PredictionsResult`

Streamlined container for predictions analysis (AAP, APM, APR) implementing the Tables.jl interface protocol.

The `PredictionsResult` type focuses specifically on predicted values without variable/contrast concepts, providing a clean interface optimized for predictions analysis. The streamlined design reflects that predictions represent "fitted values at scenarios" rather than "effects of variables."

**Fields:**
- `estimates::Vector{Float64}` - Point estimates (predicted values)
- `standard_errors::Vector{Float64}` - Delta-method standard errors
- `profile_values::Union{Nothing, NamedTuple}` - Per-row values of the context columns: the reference-grid variables for `profile_margins()` (APM/APR), and the `scenarios=` and `groups=` variables for `population_margins()`. `nothing` for a plain population result (AAP); `metadata[:groups_vars]` and `metadata[:scenarios_vars]` say which columns are groups and which are scenarios
- `group_values::Union{Nothing, NamedTuple}` - Reserved; currently always `nothing`. Group columns from the `groups` parameter are stored in `profile_values`
- `gradients::Matrix{Float64}` - Parameter gradients (G matrix) for delta-method computation
- `metadata::Dict{Symbol, Any}` - Analysis metadata (model info, options used, sample size, etc.)

**Key Features:**
- Omits variable/contrast fields (not applicable to predictions - predictions don't have "x" or "dy/dx" concepts)
- Single optimized DataFrame format for predictions display
- Clean tabular output focused on prediction values and statistics
- `profile_values` holds the reference grid (`profile_margins()`) or the scenario and group columns (`population_margins()` with `scenarios=` or `groups=`); `nothing` for a plain population result
- Group columns are in `profile_values` (not `group_values`); `metadata[:groups_vars]` identifies them

**Data Integration Framework:**
```julia
# Type-specific result containers with Tables.jl protocol
effects_result = population_margins(model, data; type=:effects)  # Returns EffectsResult
predictions_result = population_margins(model, data; type=:predictions)  # Returns PredictionsResult

# Accessing fields directly
effects_result.estimates          # Vector{Float64} of marginal effects
effects_result.standard_errors    # Vector{Float64} of standard errors
effects_result.variables          # Vector{String} of variable names
effects_result.profile_values     # Nothing (plain population result) or NamedTuple (profile, scenarios or groups)
effects_result.group_values       # Reserved; always Nothing (group columns are in profile_values)
effects_result.metadata           # Dict{Symbol, Any} with analysis info

# Profile margins have profile_values populated
profile_result = profile_margins(model, data, means_grid(data); type=:effects)
profile_result.profile_values     # NamedTuple(x1=[...], x2=[...], ...)

# Grouped analysis keeps its group columns in profile_values; metadata says which are groups
grouped_result = population_margins(model, data; type=:effects, groups=:region)
grouped_result.profile_values     # NamedTuple(region=["North", "South", ...])
grouped_result.metadata[:groups_vars]       # [:region]
Margins.context_columns(grouped_result)     # ([:region], Symbol[]): (groups, scenarios)

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

## Output Columns and Inference

### Columns of `DataFrame(result)`

Margins 3.0 uses one set of column names across result types:

| result | columns (default format) |
|:-------|:-------------------------|
| population effects (AME) | `type, variable, contrast, estimate, se, statistic, df, p_value, ci_lower, ci_upper, n` |
| population effects with `groups`/`scenarios` | `over_<g>…, at_<s>…,` then as above |
| population predictions (AAP) | `[over_<g>…, at_<s>…,] type, estimate, se, statistic, df, p_value, ci_lower, ci_upper, n` |
| profile effects (MEM/MER) | `<grid vars…>, type, variable, contrast, estimate, se, statistic, df, p_value, ci_lower, ci_upper, n` |
| profile predictions (APM/APR) | `<grid vars…>, type, estimate, se, statistic, df, p_value, ci_lower, ci_upper, n` |
| `DataFrame(contrast(...))` | `contrast, se, statistic, df, p_value, ci_lower, ci_upper, estimate1, estimate2, row1, row2` |
| second differences | see [Second Differences](second_differences.md#Output-Schema) |

- **Group columns** carry an `over_` prefix (`groups=:region` gives `over_region`, as in Stata's `over()`), for plain, binned and nested groups alike.
- **Scenario columns** carry an `at_` prefix (`scenarios=(x=[0, 1],)` gives `at_x`).
- **Profile (reference-grid) columns** keep their bare names.
- `format=:stata` uses Stata's labels: `margin, std_err, t, P_t, N` for t inference and `margin, std_err, z, P_z, N` for z inference.

### Test statistic and degrees of freedom

`statistic = estimate / se`. Its reference distribution, used for `p_value` and the confidence interval, is t with `df` degrees of freedom (`df = Inf` is the standard Normal, i.e. z). The rule depends only on the model and the covariance estimator (`Margins._inference_df(model, vcov)`), and the same `df` is used for every row of a result, for contrasts between rows and for second differences:

| model | `vcov` | `df` |
|:------|:-------|:-----|
| `lm(...)`, or `glm(..., Normal(), IdentityLink())` | default `GLM.vcov` (or a matrix equal to `vcov(model)`) | `dof_residual(model)` (t) |
| the same linear models | robust / clustered / custom (`HC1()`, `CR1(cl)`, a function, another matrix) | `Inf` (z) |
| other GLMs (logit, probit, Poisson, …), mixed models, FixedEffectModels | any | `Inf` (z) |

For a linear model every marginal effect, adjusted prediction and contrast is a linear combination of β, so `estimate / se` has an exact t distribution with the residual degrees of freedom — the reference Stata's `margins` uses after `regress` and that `GLM.coeftable` reports for `lm`. For nonlinear models the delta-method standard errors are asymptotic, so z is used, as in Stata after `logit` and in R's marginaleffects. With an identity link the link and response scales coincide, so `scale` does not change `df`.

Robust and clustered covariance: Stata uses the residual df after `vce(robust)` and `n_clusters − 1` after `vce(cluster)`. The `vcov` argument (a CovarianceMatrices estimator or a function) does not carry that information reliably, so Margins uses z for every non-default covariance estimator. With few clusters this is anti-conservative; compute the t-based p-value yourself if you need Stata's small-sample convention, e.g. `2 .* ccdf.(TDist(G - 1), abs.(df.statistic))`.

The df is stored in `result.metadata[:df]`.

## Second Differences (Interaction Effects)

Margins.jl provides comprehensive support for computing second differences—interaction effects on the predicted outcome scale. Second differences quantify how marginal effects vary across levels of a moderating variable, addressing the fundamental question: **"Does the effect of X depend on Z?"**

### Quick Start

One function covers every case. Give it the fitted model, the data, the focal variable(s) and the
moderator:

```julia
second_differences(model, data, :age, :treated)             # discrete modifier: treated vs not
second_differences(model, data, [:age, :income], :treated)  # several focal variables
second_differences(model, data, :age, :income)              # continuous modifier: average ∂AME/∂income over the observed incomes
second_differences(model, data, :age, :income; at=[20_000, 50_000])      # at chosen values
second_differences(model, data, :age, :income; levels=[20_000, 50_000])  # slope between two values
second_differences(model, data, :age, :treated; profile=(female=[0, 1],)) # separately by female
second_differences(model, data, :age, :treated; vcov=HC1())              # robust covariance
```

The modifier type is detected from the data (Bool, 0/1, two-valued, string or categorical →
discrete; numeric with more than two distinct values → continuous); use
`modifier_type=:categorical` for an integer-coded categorical modifier. Scenarios, AMEs and
delta-method standard errors are handled internally. `vcov` takes the same forms as in
`population_margins` and defaults to `GLM.vcov`.

If you already hold an `EffectsResult` from `population_margins(...; scenarios=..., type=:effects)`,
the same comparison is available on its rows, with `vcov` defaulting to the covariance the result
was computed with:

```julia
ames = population_margins(model, data; scenarios=(treated=[0, 1],), type=:effects)
sd = second_differences(ames, :age, :treated)
```

### Available Functions

- **`second_differences(model, data, variables, modifier; …)`**: the entry point described above.
- **`second_differences(result, variables, modifier[, vcov])`**: the same on a precomputed result.

That is the whole API: the 2.x functions `second_differences_at`, `second_differences_pairwise`,
`second_differences_all_contrasts`, `second_differences_table` and `second_difference` were removed in 3.0;
see [Migrating from Margins 2.x](second_differences.md#Migrating-from-Margins-2.x).

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
- `backend=:ad` → Default backend (highest accuracy, zero allocation)
- `backend=:fd` → Alternative backend (zero allocation, numerical approximation)

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

Measures other than `:effect` are defined for continuous variables only — they scale a
derivative ∂y/∂x by x̄ and/or ȳ, which a discrete level contrast does not have. Naming a
categorical or boolean variable in `vars` together with an elasticity measure raises an
`ArgumentError`; when `vars` is auto-detected, discrete variables are skipped with an
informational message. Request them separately with `measure=:effect`.

### Profile-Specific Parameters

#### Reference Grid (positional argument)

Profile margins take a reference grid as the third positional argument. Use the built-in grid builders or pass a DataFrame directly:

```julia
# At sample means (most common)
profile_margins(model, data, means_grid(data))

# Cartesian product: 6 scenarios (3×2)
profile_margins(model, data, cartesian_grid(x=[0,1,2], group=["A","B"]))

# Balanced grid (sample frequencies for categoricals, means for continuous)
profile_margins(model, data, balanced_grid(data))

# DataFrame grid (full control)
grid = DataFrame(x=[0,1,2], group=["A","A","B"])
profile_margins(model, data, grid)
```

See [Reference Grids](reference_grids.md) for full documentation of grid builders.

### Population-Specific Parameters

#### Grouping (`groups`)
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
population_margins(model, data; scenarios=(treatment=[0, 1],), type=:effects)
```

## Usage Patterns

### Basic Workflow
```julia
# 1. Fit model
model = lm(@formula(y ~ x1 + x2 + group), data)

# 2. Population analysis (most common starting point)
ame = population_margins(model, data; type=:effects)
aap = population_margins(model, data; type=:predictions)

# 3. Profile analysis for specific scenarios
mem = profile_margins(model, data, means_grid(data); type=:effects)
scenario_results = profile_margins(model, data, cartesian_grid(x1=[0,1,2]); type=:effects)

# 4. Convert to DataFrame for analysis
DataFrame(ame)
```

### Performance Optimization
```julia
# Maximum performance configuration
fast_result = population_margins(model, data; backend=:fd, scale=:link)

# Profile analysis is O(1) - efficient regardless of data size
grid = cartesian_grid(x1=[0,1,2])
profile_result = profile_margins(model, huge_data, grid; type=:effects)  # ~300μs regardless of data size
```

### Advanced Analysis Patterns
```julia
# Elasticity analysis across scenarios (profile)
scenarios = cartesian_grid(x1=[0, 1, 2])
elasticities = profile_margins(model, data, scenarios; 
    measure=:elasticity, vars=[:x2])

# Robust standard errors (with CovarianceMatrices.jl); inference is then z (df = Inf)
using CovarianceMatrices
robust_effects = population_margins(model, data; vcov=HC1(), type=:effects)

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
ame_hc1 = population_margins(model, data; vcov=HC1())
ame_hc3 = population_margins(model, data; vcov=HC3())
ame_clustered = population_margins(model, data; vcov=Clustered(:cluster_var))
ame_hac = population_margins(model, data; vcov=HAC(Bartlett()))
```

### With DataFrames Ecosystem
```julia
using DataFrames, CSV, Chain

# Complete analysis pipeline
results_df = @chain begin
    population_margins(model, data; type=:effects)
    DataFrame(_)
    select(_, :variable, :contrast, :estimate, :se, :statistic, :df, :p_value)
    filter(row -> row.p_value < 0.05, _)  # Significant effects only
end

# Export results
CSV.write("significant_effects.csv", results_df)
```

---

*This API reference provides complete documentation for all Margins.jl functionality. For conceptual background on the 2×2 framework, see [Mathematical Foundation](mathematical_foundation.md). For performance optimization guidance, see [Performance Guide](performance.md). For advanced features including elasticities and robust inference, see [Advanced Features](advanced.md).*
