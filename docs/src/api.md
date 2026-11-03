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

**Computational Characteristics**: The model is evaluated only at the reference-grid rows, so the cost is dominated by the grid size. Each call also does a small amount of O(n) work on the data (column conversion, the derived-column check, and typical values over the estimation sample for variables the grid leaves out).
*Comprehensive performance benchmarking and memory allocation analysis are detailed in the [Performance Guide](performance.md)*

## Result Type Specifications

### Type-Safe Result System

Margins.jl implements a specialized type system that provides type safety and optimized DataFrame formatting through distinct result containers for different analysis types.

### `EffectsResult`

Structured container for marginal effects analysis (AME, MEM, MER) implementing the Tables.jl interface protocol.

The `EffectsResult` type encapsulates computed marginal effects along with associated statistical inference quantities including standard errors, confidence intervals, and hypothesis test statistics. The type contains variable identification fields (`variables`, `terms`) essential for effects interpretation and supports multiple DataFrame formatting options.

**Fields:**
- `estimates::Vector{Float64}` - Point estimates of marginal effects
- `standard_errors::Vector{Float64}` - Delta-method standard errors
- `variables::Vector{String}` - The "x" in dy/dx (which variable each row represents)
- `terms::Vector{String}` - Contrast descriptions (e.g., "dy/dx", "treated - control")
- `context::Union{Nothing, ResultContext}` - Per-row values of the context columns, in output order, with the kind of each column (see [`ResultContext`](#ResultContext)): the reference-grid variables for `profile_margins()` (MEM/MER), the `groups=` and `scenarios=` variables for `population_margins()`. `nothing` for a plain population result (AME)
- `gradients::Matrix{Float64}` - Parameter gradients (G matrix) for delta-method computation
- `inference::Inference` - The covariance matrix, degrees of freedom and `alpha` behind the standard errors and intervals (see [`Inference`](#Inference))
- `metadata::Dict{Symbol, Any}` - Analysis metadata (model info, options used, sample size, etc.)

**Key Features:**
- Multiple DataFrame formats: `:standard`, `:compact`, `:confidence`, `:profile`, `:stata`
- Auto-detects appropriate format based on analysis type

### `PredictionsResult`

Streamlined container for predictions analysis (AAP, APM, APR) implementing the Tables.jl interface protocol.

The `PredictionsResult` type focuses specifically on predicted values without variable/contrast concepts, providing a clean interface optimized for predictions analysis. The streamlined design reflects that predictions represent "fitted values at scenarios" rather than "effects of variables."

**Fields:**
- `estimates::Vector{Float64}` - Point estimates (predicted values)
- `standard_errors::Vector{Float64}` - Delta-method standard errors
- `context::Union{Nothing, ResultContext}` - Per-row values of the context columns, as for `EffectsResult`: the reference-grid variables for `profile_margins()` (APM/APR), the `groups=` and `scenarios=` variables for `population_margins()`; `nothing` for a plain population result (AAP)
- `gradients::Matrix{Float64}` - Parameter gradients (G matrix) for delta-method computation
- `inference::Inference` - The covariance matrix, degrees of freedom and `alpha` behind the standard errors and intervals
- `metadata::Dict{Symbol, Any}` - Analysis metadata (model info, options used, sample size, etc.)

**Key Features:**
- Omits variable/contrast fields (not applicable to predictions - predictions don't have "x" or "dy/dx" concepts)
- Single optimized DataFrame format for predictions display
- Clean tabular output focused on prediction values and statistics

### `Inference`

`result.inference`: how a result's standard errors, tests and intervals were computed, set from the call that produced it.

- `vcov::Matrix{Float64}` - The parameter covariance matrix Σ the delta method used (the `vcov=` keyword; `GLM.vcov(model)` by default). With `gᵢ = result.gradients[i, :]`, the standard error of row `i` is `√(gᵢ'Σgᵢ)` and the covariance of rows `i` and `j` is `gᵢ'Σgⱼ`
- `df::Float64` - Degrees of freedom of the reference distribution (`Inf` = z; see [Test statistic and degrees of freedom](#Test-statistic-and-degrees-of-freedom))
- `alpha::Float64` - Significance level of the intervals

`DataFrame(result)`, `contrast(result, …)` and `second_differences(result, …)` all read it, so they cannot disagree with the result. To use another covariance estimator, pass `vcov=` to the call that computes the result.

### `ResultContext`

`result.context`: the per-row context columns, or `nothing`.

- `columns::NamedTuple` - One vector per context variable, one entry per result row, in output order: the reference grid's columns in the grid's order (`profile_margins`), or the `groups=` variables, then the `scenarios=` variables, each in the order given (`population_margins`)
- `kinds::Vector{Symbol}` - The kind of each column: `:grid` (bare name in `DataFrame(result)`), `:group` (`over_<var>`) or `:scenario` (`at_<var>`)

`keys(ctx)`, `ctx[name]` and `pairs(ctx)` read the columns; `Margins.context_columns(result)` returns the group and scenario names.

**Data Integration Framework:**
```julia
# Type-specific result containers with Tables.jl protocol
effects_result = population_margins(model, data; type=:effects)  # Returns EffectsResult
predictions_result = population_margins(model, data; type=:predictions)  # Returns PredictionsResult

# Accessing fields directly
effects_result.estimates          # Vector{Float64} of marginal effects
effects_result.standard_errors    # Vector{Float64} of standard errors
effects_result.variables          # Vector{String} of variable names
effects_result.context            # nothing (plain population result) or a ResultContext
effects_result.inference          # Inference(vcov, df, alpha)
effects_result.inference.df       # Inf (z) or the residual df (t)
effects_result.metadata           # Dict{Symbol, Any} with analysis info

# Profile margins have a context: the reference grid's columns, in the grid's order
profile_result = profile_margins(model, data, means_grid(data); type=:effects)
profile_result.context.columns    # (x1=[...], x2=[...], ...)
profile_result.context.kinds      # [:grid, :grid, ...]

# Grouped analysis: the context holds the group columns, marked as groups
grouped_result = population_margins(model, data; type=:effects, groups=:region)
grouped_result.context[:region]             # ["North", "South", ...]
grouped_result.context.kinds                # [:group]
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
- **Order:** group columns, then scenario columns, each in the order given in `groups` / `scenarios`; reference-grid columns in the grid's column order. `show(result)` uses the same order.
- `format=:stata` uses Stata's labels: `margin, std_err, t, P_t, N` for t inference and `margin, std_err, z, P_z, N` for z inference.

### Test statistic and degrees of freedom

`statistic = estimate / se`. Its reference distribution, used for `p_value` and the confidence interval, is t with `df` degrees of freedom (`df = Inf` is the standard Normal, i.e. z). `df` is a property of the model class, whatever `vcov` is passed (`Margins._inference_df(model)`), and the same `df` is used for every row of a result, for contrasts between rows and for second differences:

| model | `df` |
|:------|:-----|
| `lm(...)`, or `glm(..., Normal(), IdentityLink())`, with any `vcov` (default, robust, clustered, a function, a matrix) | `dof_residual(model)` (t) |
| FixedEffectModels `reg(...)` | `dof_residual(model)` (t): FixedEffectModels' own value, which is G − 1 with a cluster `vcov`, as its `coeftable` uses |
| other GLMs (logit, probit, Poisson, …), mixed models | `Inf` (z) |

The `df` keyword of `population_margins`, `profile_margins` and the model-level `second_differences` overrides the rule: a positive number, or `Inf` for z.

For a linear model every marginal effect, adjusted prediction and contrast is a linear combination of β, so `estimate / se` has an exact t distribution with the residual degrees of freedom — the reference Stata's `margins` uses after `regress` and that `GLM.coeftable` reports for `lm`. For nonlinear models the delta-method standard errors are asymptotic, so z is used, as in Stata after `logit` and in R's marginaleffects. With an identity link the link and response scales coincide, so `scale` does not change `df`.

Robust and clustered covariance: for a linear model Margins keeps the residual df with any `vcov`, as Stata does after `regress, vce(robust)` and as R's `coeftest(m, vcovHC(m))` does. After `vce(cluster c)` Stata uses `n_clusters − 1`; the `vcov` argument does not carry the cluster count, so pass it: `population_margins(m, data; vcov=CR1(data.c), df=G - 1)`. (Before 3.0, any covariance other than `GLM.vcov` gave z.)

The df is stored in `result.inference.df`.

### Offsets

A model's offset enters the linear predictor, η = x′β + offset, and so every prediction and every derivative of the inverse link: GLM.jl `glm(...; offset=log.(exposure))`, a MixedModels GLMM's `offset=`, and for FixedEffectModels the estimated absorbed fixed effects (`reg(...; save=:fe)`).

- **`population_margins`** uses each row's own offset, also within `groups` and under `scenarios` (where it stays at its observed value). The offset must have one entry per row of `data`. For a Poisson model with an intercept and an exposure offset the average adjusted prediction equals `mean(y)`, as `mean(predict(model))` does.
- **`profile_margins`** needs the offset at each grid row when the requested quantity depends on it (predictions, and response-scale effects of a non-identity link or measures that divide by ŷ): pass `offset=` (a number, or one value per grid row, on the offset's scale, e.g. `offset=log(1000)`) or add an `offset` column to the grid. Without it the call is an error; the offset is never silently 0.

Before 3.0 offsets were ignored (predictions at η = x′β).

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
second_differences(model, data, :age, :treated; profile=(female=[0, 1],)) # everyone's female set to 0, then 1
second_differences(model, data, :age, :treated; groups=:region)           # within each region (over_region)
second_differences(model, data, :age, :treated; vcov=HC1())              # robust covariance
```

The modifier type is detected from the data (Bool, 0/1, two-valued, string or categorical →
discrete; numeric with more than two distinct values → continuous); use
`modifier_type=:categorical` for an integer-coded categorical modifier. Scenarios, AMEs and
delta-method standard errors are handled internally. `vcov` takes the same forms as in
`population_margins` and defaults to `GLM.vcov`.

If you already hold an `EffectsResult` from `population_margins(...; scenarios=..., type=:effects)`,
the same comparison is available on its rows, using the covariance matrix and df the result was
computed with (`result.inference`; for another estimator, compute the result with `vcov=…`):

```julia
ames = population_margins(model, data; scenarios=(treated=[0, 1],), type=:effects)
sd = second_differences(ames, :age, :treated)
```

### Available Functions

- **`second_differences(model, data, variables, modifier; …)`**: the entry point described above.
- **`second_differences(result, variables, modifier)`**: the same on a precomputed result, with its covariance matrix and df (`result.inference`).

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
derivative ∂y/∂x by x and/or 1/ŷ (row by row; population measures average the unit-level
quantities), which a discrete level contrast does not have. Naming a
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
- `Vector{Symbol}` - Multiple grouping variables (crossed)
- `(:var, n)` / `(:var, [cuts])` - Bins of a continuous variable (quantile bins or thresholds)
- `outer => inner` - Nested grouping

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

# Profile analysis: cost dominated by the grid size, plus a small per-call O(n) part
grid = cartesian_grid(x1=[0,1,2])
profile_result = profile_margins(model, huge_data, grid; type=:effects)
```

### Advanced Analysis Patterns
```julia
# Elasticity analysis across scenarios (profile)
scenarios = cartesian_grid(x1=[0, 1, 2])
elasticities = profile_margins(model, data, scenarios; 
    measure=:elasticity, vars=[:x2])

# Robust standard errors (with CovarianceMatrices.jl); df is unchanged (t for this lm, z for other GLMs)
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
```

#### Profile Specification Errors
```julia
# Error: Invalid reference grid argument (must be DataFrame or a grid builder output)
profile_margins(model, data, "invalid")
# → Clear guidance on valid reference grid specifications

# Not an error: model variables missing from the grid are filled in
incomplete_grid = DataFrame(x1=[0,1])  # Missing x2, group from model
profile_margins(model, data, incomplete_grid)
# → x2 at its mean, group at its frequency mixture (model-weighted, estimation sample)
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
ame_clustered = population_margins(model, data; vcov=CR1(data.cluster_var))
ame_hac = population_margins(model, data; vcov=Bartlett{Andrews}())  # HAC, Andrews bandwidth
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
