# Advanced Features

*Elasticities, robust standard errors, and specialized analysis techniques*

## Elasticities and Semi-Elasticities

Margins.jl provides comprehensive elasticity support through the `measure` parameter, following the same **Population vs Profile** framework as standard marginal effects. *For conceptual background on the 2×2 framework, see [Mathematical Foundation](mathematical_foundation.md).*

### Types of Elasticity Measures

#### Standard Elasticity
Definition: Percent change in Y per percent change in X
Formula: `(∂Y/∂X) × (X/Y)`
Interpretation: "A 1% increase in X leads to an ε% change in Y"

```julia
# Population average elasticities
population_margins(model, data; type=:effects, measure=:elasticity)

# Elasticities at sample means
profile_margins(model, data, means_grid(data); type=:effects, measure=:elasticity)
```

#### Semi-Elasticity with respect to X  
Definition: Percent change in Y per unit change in X
Formula: `(∂Y/∂X) × (1/Y)`
Interpretation: "A 1-unit increase in X leads to a (100×ε)% change in Y"

```julia
# Population average semi-elasticities (X)
population_margins(model, data; type=:effects, measure=:semielasticity_dyex)

# Semi-elasticities at specific scenarios
profile_margins(model, data, cartesian_grid(x1=[0,1,2]); type=:effects, measure=:semielasticity_dyex)
```

#### Semi-Elasticity with respect to Y
Definition: Unit change in Y per percent change in X  
Formula: `(∂Y/∂X) × X`
Interpretation: "A 1% increase in X leads to an ε-unit change in Y"

```julia
# Population average semi-elasticities (Y)
population_margins(model, data; type=:effects, measure=:semielasticity_eydx)
```

### Elasticity Framework Application

Elasticities follow the same **Population vs Profile** distinction as marginal effects (*see [Mathematical Foundation](mathematical_foundation.md) for detailed framework explanation*):

| Measure | Population Approach | Profile Approach |
|---------|-------------------|------------------|
| **Elasticity** | Average of `(∂Y/∂X) × (Xᵢ/Yᵢ)` across sample | `(∂Y/∂X) × (X̄/Ȳ)` at representative values |
| **Semi-elasticity (X)** | Average of `(∂Y/∂X) × (1/Yᵢ)` across sample | `(∂Y/∂X) × (1/Ȳ)` at representative values |
| **Semi-elasticity (Y)** | Average of `(∂Y/∂X) × Xᵢ` across sample | `(∂Y/∂X) × X̄` at representative values |

### Practical Example: Wage Elasticities

```julia
using Margins, DataFrames, GLM

# Economic data: wages, education, experience
df = DataFrame(
    log_wage = randn(1000) .+ 2.5, # Log wages
    education = rand(12:20, 1000), # Years of education  
    experience = rand(0:30, 1000), # Years of experience
    age = rand(25:55, 1000)
)

model = lm(@formula(log_wage ~ education + experience + age), df)

# Education elasticity of wages (population average)
edu_elasticity = population_margins(model, df; 
                                   vars=[:education], 
                                   measure=:elasticity)
println("Population average education elasticity: ", DataFrame(edu_elasticity))

# Education elasticity at different experience levels (profile analysis)  
exp_scenarios = profile_margins(
    model, df,
    cartesian_grid(experience=[0, 10, 20, 30]);
    vars = [:education],
    measure = :elasticity
)
println("Education elasticity by experience level:")
println(DataFrame(exp_scenarios))

# Semi-elasticity: percent wage change per year of education
edu_semielast = population_margins(
    model, df;
    vars = [:education],
    measure = :semielasticity_dyex
)
println("Education semi-elasticity: ", DataFrame(edu_semielast))
```

### When Profile ≠ Population for Elasticities

In GLMs with non-identity links, population and profile elasticities can differ substantially:

```julia
# Logistic model example
logit_model = glm(@formula(employed ~ education + experience), df, Binomial(), LogitLink())

# Population average employment elasticity w.r.t. education
pop_elastic = population_margins(logit_model, df; vars=[:education], measure=:elasticity)

# Employment elasticity at sample means
prof_elastic = profile_margins(logit_model, df, means_grid(df); vars=[:education], measure=:elasticity)

# These will differ because logistic function is nonlinear
println("Population elasticity: ", DataFrame(pop_elastic).estimate[1])
println("Profile elasticity: ", DataFrame(prof_elastic).estimate[1])
```

## Robust Standard Errors

Margins.jl integrates [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for robust standard error computation.

### Basic Robust Standard Errors

#### Heteroskedasticity-Robust (White/Huber-White)
```julia
using CovarianceMatrices

# Apply robust covariance via vcov parameter
robust_effects = population_margins(
    model, data; vcov=HC1(), type=:effects
)
```

#### Available Robust Estimators
```julia
# Different heteroskedasticity-robust variants
HC0()  # Basic White estimator
HC1()  # Degrees-of-freedom adjusted (most common)
HC2()  # Leverage-adjusted  
HC3()  # Jackknife-type
HC4()  # High-leverage robust
HC5()  # Outlier-robust

# Example with HC3
result = population_margins(model, data; vcov=HC3())
```

### Clustered Standard Errors

#### Single-Level Clustering
```julia
# Cluster by firm ID
clustered_effects = population_margins(model, data;
    vcov=Clustered(:firm_id), type=:effects)
```

#### Multi-Level Clustering  
```julia
# Two-way clustering (firm and year)
result = population_margins(model, data; vcov=Clustered([:firm_id, :year]))
```

### HAC (Heteroskedasticity and Autocorrelation Consistent) Standard Errors

```julia
# Newey-West HAC estimator
effects_hac = population_margins(model, data;
    vcov=HAC(Bartlett()), type=:effects)
```

### Custom Covariance Providers

#### Function-Based Covariance
```julia
# Custom covariance function (must return an AbstractMatrix)
function my_robust_vcov(model)
    # ... compute covariance from model ...
    return Σ::AbstractMatrix
end

# Use custom function directly
result = population_margins(model, data; vcov=my_robust_vcov)
```

### Robust Standard Errors with Elasticities

Robust standard errors work seamlessly with all elasticity measures:

```julia
# Robust elasticity estimates
robust_elasticities = population_margins(model, data;
    vcov=HC1(),
    measure=:elasticity, type=:effects)

# Profile elasticities with clustered SEs
profile_elasticities = profile_margins(model, data,
    means_grid(data); vcov = Clustered(:cluster_var),
    measure = :elasticity)
```

### Standard Errors for Elasticities: Delta Method vs Bootstrap

!!! warning "Conditional vs Unconditional Inference"
    Standard errors for elasticity measures computed via the delta method (default) represent conditional inference; they assume the observed data (X, Y) are fixed and only account for uncertainty in parameter estimates β̂. Bootstrap standard errors represent unconditional inference and may be larger because they also account for sampling variation in the data.

#### Understanding the Difference

For elasticity measures like `ε = (x̄/ȳ) × (∂y/∂x)`, the transformation involves sample moments (x̄, ȳ) that are treated differently by different inference methods:

Delta Method (Default):
- Assumption: Observed data X and Y are fixed constants
- Uncertainty source: Only parameter estimates β̂
- Variance: `Var[ε(β̂) | X, Y]`
- Advantages: Fast, analytically exact (given the conditional assumption)
- Implementation: Uses the quotient rule to account for mean(ŷ) depending on β

Bootstrap:
- Assumption: Data are sampled from a population
- Uncertainty sources: Both β̂ and the sample moments x̄, ȳ
- Variance: `Var[ε(β̂, x̄, ȳ)]`
- Advantages: Accounts for full sampling variation
- Trade-off: Computationally intensive (requires refitting model many times)

#### Why They Differ

The key distinction is that elasticity formulas involve ratios of sample statistics:

```julia
# For elasticity: ε = (x̄ / mean(ŷ)) × AME
# Where:
#   x̄ = sample mean of predictor (varies across bootstrap samples)
#   mean(ŷ) = sample mean of predictions (varies with both β and resampled data)
#   AME = average marginal effect (varies with β)
```

When you bootstrap:
1. Different observations are sampled → x̄ changes
2. Model is refit → β̂ changes → AME changes
3. Predictions are recomputed → mean(ŷ) changes

The delta method only captures variation from (2), treating (1) and parts of (3) as fixed.

!!! note "This is Not a Bug"
    This behavior matches other statistical software:
    - R's marginaleffects: Documentation states *"For nonlinear models, the delta method is only an approximation"* and recommends bootstrap for transformations
    - Stata's margins: Documents that delta method *"assumes that the values at which the covariates are evaluated are fixed"*

#### Example: Comparing Methods

```julia
using Margins, GLM, DataFrames, Bootstrap

# Fit model
model = lm(@formula(y ~ x + z), data)

# Delta method SEs (default - fast, conditional)
delta_result = population_margins(model, data;
    vars=[:x],
    measure=:elasticity)
println("Delta method SE: ", DataFrame(delta_result).se[1])

# Bootstrap SEs (slower, unconditional)
# Note: Built-in bootstrap support coming soon
# For now, use manual bootstrap:
function boot_elasticity(data, indices)
    boot_data = data[indices, :]
    boot_model = lm(@formula(y ~ x + z), boot_data)
    boot_result = population_margins(boot_model, boot_data;
        vars=[:x], measure=:elasticity)
    return DataFrame(boot_result).estimate[1]
end

bs = bootstrap(boot_elasticity, data, BasicSampling(1000))
boot_se = std(bs.t[1])
println("Bootstrap SE: ", boot_se)

# Bootstrap SE will typically be larger, especially in small samples
```

#### Technical Details

Margins.jl implements the full quotient rule for elasticity derivatives:

```math
\frac{\partial \varepsilon}{\partial \beta} = \frac{\bar{x}}{\bar{y}} \frac{\partial \text{AME}}{\partial \beta} - \frac{\varepsilon}{\bar{y}} \frac{\partial \bar{y}}{\partial \beta}
```

Here, the second term accounts for mean(ŷ) depending on β.

!!! info "References"
    - Krinsky, I., & Robb, A. L. (1986). "On Approximating the Statistical Properties of Elasticities." *Review of Economics and Statistics*, 68(4), 715-719.
    - Arel-Bundock, V. (2023). "marginaleffects: Predictions, Comparisons, Slopes, Marginal Means, and Hypothesis Tests." R package. [Documentation on inference](https://marginaleffects.com/chapters/uncertainty.html)
    - Greene, W. H. (2018). *Econometric Analysis* (8th ed.), Section 3.6 on the Delta Method.

## Standardized Predictors

Margins.jl seamlessly integrates with [StandardizedPredictors.jl](https://github.com/beacon-biosignals/StandardizedPredictors.jl) for models fit with standardized (z-scored) variables. Marginal effects are automatically reported on the original (raw) scale, requiring no manual back-transformation.

### Why Standardize Predictors?

Standardization transforms variables to have mean 0 and standard deviation 1:
```julia
x_std = (x - mean(x)) / std(x)
```

### Integration with Margins.jl

```julia
using Margins, GLM, StandardizedPredictors, DataFrames

# Fit model with standardized income
df = DataFrame(
    sales = randn(1000) .* 10000 .+ 50000,
    income = randn(1000) .* 20000 .+ 50000,  # mean ≈ \$50k, std ≈ \$20k
    age = randn(1000) .* 10 .+ 40
)

model = lm(@formula(sales ~ income + age), df,
           contrasts = Dict(:income => ZScore(), :age => ZScore()))

# Marginal effects are automatically on ORIGINAL scale
result = population_margins(model, df; type=:effects, vars=[:income, :age])
DataFrame(result)

# income effect: change in sales per \$1 increase in income (not per SD!)
# age effect: change in sales per 1-year increase in age (not per SD!)
```

### How Automatic Back-Transformation Works

When computing marginal effects, Margins.jl uses FormulaCompiler.jl's derivative system, which automatically applies the chain rule through the standardization transformation:

Mathematical detail (with `ZScore`):
- Model uses: `x_std = (x - μ) / σ`
- Derivative computation: `∂η/∂x_raw = ∂η/∂x_std × ∂x_std/∂x_raw = β × (1/σ)`
- Result: Marginal effect per unit of original variable

Both finite differences (FD) and automatic differentiation (AD) backends handle this automatically:
- FD: Perturbs raw values → standardization applied during evaluation → derivative includes 1/σ
- AD: Dual arithmetic propagates through (x - μ)/σ → derivative includes 1/σ

### Comparison: Raw vs Standardized Models

```julia
# Fit both raw and standardized models
model_raw = lm(@formula(sales ~ income), df)
model_std = lm(@formula(sales ~ income), df, contrasts=Dict(:income => ZScore()))

# Marginal effects are IDENTICAL (both on original scale)
me_raw = population_margins(model_raw, df; vars=[:income])
me_std = population_margins(model_std, df; vars=[:income])

# Both give same result: effect per dollar of income
@assert DataFrame(me_raw).estimate ≈ DataFrame(me_std).estimate
```

Why they match:
- Raw model: `∂sales/∂income_dollars = β₁`
- Standardized model: `∂sales/∂income_dollars = β₁_std / σ_income`
- The σ in the denominator is automatically included by the chain rule

### Elasticities with Standardized Predictors

Elasticities are invariant to standardization because they are scale-free:

```julia
# Elasticity with standardized predictors
model = lm(@formula(sales ~ income + age), df,
           contrasts = Dict(:income => ZScore()))

# Elasticity uses raw values of X and Y
result = population_margins(model, df; vars=[:income], measure=:elasticity)

# Interpretation: % change in sales per % change in income
# (same whether predictors are standardized or not)
```

### Profile Analysis with Standardized Predictors

Reference grids work directly with raw values:

```julia
# Specify scenarios in original units
using Margins: cartesian_grid

grid = cartesian_grid(
    income = [40000, 60000, 80000],  # Raw dollar amounts
    age = [30, 40, 50]                # Raw years
)

result = profile_margins(model, df, grid; type=:effects)

# Effects are per dollar of income, per year of age
# Standardization is handled automatically during evaluation
```

### Technical Notes

1. Model coefficients (`coef(model)`) are on the _standardized_ scale
   - β₁ represents effect per SD change in x

2. Jacobian from FormulaCompiler is on the _raw_ scale
   - Includes 1/σ factor from chain rule automatically

3. Marginal effects: `g = J' × β`
   - The 1/σ in J combines with standardized β to give raw-scale effects

This behavior is validated with tests that compare the raw and standardized models, ensuring that both produce identical marginal effects on the original measurement scales.


## Categorical Mixtures for Policy Analysis

Margins.jl supports **categorical mixtures** for scenario analysis, which enables the specification of population compositions as an alternative to (the observed) category levels.

### Motivation: Realistic Population Scenarios

Marginal effects often use arbitrary categorical values (e.g., "set all observations to treatment=1"). Categorical mixtures enable the specification of typical values:

```julia
using CategoricalArrays, Margins

# Instead of: "All treated" (unrealistic)
unrealistic = profile_margins(
    model, data, cartesian_grid(treatment = [1]); type = :predictions
)

# Use: Realistic treatment rate  
realistic = profile_margins(
    model, data, 
    DataFrame(treatment=[mix(0 => 0.3, 1 => 0.7)])
) # 70% treatment rate
```

### Frequency-Weighted Categorical Defaults

When categorical variables are unspecified in profiles, Margins.jl uses actual sample frequencies rather than arbitrary first levels:

```julia
# Data composition:
#     education = 40% HS, 45% College, 15% Graduate
#     region = 60% Urban, 40% Rural

# Effects "at means" uses realistic composition
result = profile_margins(model, data, means_grid(data); type = :effects)
# → Continuous vars: sample means
# → education: mix("HS" => 0.40, "College" => 0.45, "Graduate" => 0.15)  
# → region: mix("Urban" => 0.60, "Rural" => 0.40)
```

### Scenario Analysis

#### Demographic Transition Scenarios
```julia
# Current population composition
current_scenario = profile_margins(model, data,
    DataFrame(education=[mix("HS" => 0.40, "College" => 0.45, "Graduate" => 0.15)]);
    type=:predictions)

# Future scenario: increased college graduation
future_scenario = profile_margins(model, data,
    DataFrame(education=[mix("HS" => 0.25, "College" => 0.60, "Graduate" => 0.15)]);
    type=:predictions)

# Policy impact
impact = DataFrame(future_scenario).estimate[1] - DataFrame(current_scenario).estimate[1]
```

#### Treatment Effect Heterogeneity
```julia
# Treatment effects across population compositions
treatment_scenarios = DataFrame([
    (treatment = 0, education = mix("HS" => 0.5, "College" => 0.5)),
    (treatment = 1, education = mix("HS" => 0.5, "College" => 0.5)),
    (treatment = 0, education = mix("HS" => 0.2, "College" => 0.8)),  
    (treatment = 1, education = mix("HS" => 0.2, "College" => 0.8))
])

results = profile_margins(
    model, data, treatment_scenarios; type = :predictions
)
treatment_effects_df = DataFrame(results)
```

## Advanced Grouping and Stratification

Margins.jl provides a comprehensive grouping framework for population-based marginal effects analysis, supporting hierarchical stratification patterns that extend far beyond traditional approaches.

### Hierarchical Grouping Framework

#### Basic Grouping Patterns
```julia
# Simple categorical grouping
demographic_effects = population_margins(
    model, data;
    type = :effects, vars = [:income], groups = :education
)

# Cross-tabulated grouping
cross_effects = population_margins(
    model, data;
    type = :effects,
    vars = [:income], 
    groups = [:education, :region]
)
```

#### Nested Hierarchical Grouping
```julia
# Hierarchical nesting: region → education within each region
nested_effects = population_margins(
    model, data;
    type = :effects,
    vars = [:income],
    groups = :region => :education
)

# Deep nesting: region → urban → education
deep_nested = population_margins(
    model, data;
    type = :effects,
    groups = :region => (:urban => :education)
)
```

#### Continuous Variable Binning
```julia
# Quartile analysis
income_quartiles = population_margins(
    model, data;
    type = :effects,
    groups = (:income, 4) # Q1, Q2, Q3, Q4
)

# Custom policy-relevant thresholds
policy_thresholds = population_margins(
    model, data;
    type = :effects,
    groups = (:income, [25000, 50000, 75000])
)

# Mixed categorical and continuous
mixed_groups = population_margins(
    model, data;
    type = :effects,
    groups = [:education, (:income, 4)]
)
```

### Counterfactual Scenario Analysis

#### Policy Scenarios with Population Override
```julia
# Binary policy scenarios
policy_analysis = population_margins(
    model, data;
    type = :effects,
    vars = [:outcome],
    scenarios = (:policy => [0, 1])
)

# Multi-variable scenarios
complex_scenarios = population_margins(
    model, data;
    type = :effects,
    scenarios = (:treatment => [0, 1], 
    :policy => ["current", "reform"])
)
```

#### Combined Grouping and Scenarios
```julia
# Comprehensive policy analysis: demographics × policy scenarios
full_analysis = population_margins(
    model, data;
    type = :effects,
    vars = [:outcome],
    groups = [:education, :region],
    scenarios = (:treatment => [0, 1])
)
```

### Complex Nested Patterns

#### Parallel Grouping Within Hierarchy
```julia
# Region → (education levels + income quartiles separately)
parallel_groups = population_margins(
    model, data;
    type = :effects,
    groups = :region => [:education, (:income, 4)]
)
```

#### Advanced Policy Applications
```julia
# Healthcare policy analysis
healthcare_analysis = population_margins(
    health_model, health_data;
    type = :effects,
    groups = :state => (:urban => [:insurance_type, (:income, 3)]),
    scenarios = (:policy_reform => [0, 1], :funding_level => [0.8, 1.2])
)

# Results: State × Urban/Rural × (Insurance×Income-Tertiles) × Policy×Funding scenarios
```

## Second Differences (Interaction Effects)

For comprehensive coverage of second differences—interaction effects on the predicted outcome scale—see the dedicated [Second Differences](second_differences.md) guide. Second differences quantify how marginal effects vary across moderator levels, extending the Margins.jl framework to address effect heterogeneity questions.

**Quick reference**:
```julia
# Compute AMEs across modifier levels
ames = population_margins(
    model, data; scenarios = (treated=[0,1],), type = :effects
)

# Calculate second differences
sd = second_differences(ames, :age, :treated, vcov(model))
```

Available functions: `second_differences()`, `second_difference()`, `second_differences_pairwise()`, `second_differences_all_contrasts()`.

## Error Handling and Diagnostics

### Robust Error Detection
```julia
# Check for statistical validity issues
function validate_margins_result(result::MarginsResult)
    df = DataFrame(result)
    
    # Check for excessive standard errors (potential issues)
    large_se = df[df.se .> 10 * abs.(df.estimate), :]
    if nrow(large_se) > 0
        @warn "Large standard errors detected - potential statistical issues"
        println(large_se)
    end
    
    # Check for missing values
    missing_results = df[ismissing.(df.estimate) .| ismissing.(df.se), :]
    if nrow(missing_results) > 0
        @warn "Missing values in results - check model specification"
    end
    
    return df
end

# Usage
result = population_margins(model, data)
validated_df = validate_margins_result(result)
```

### Covariance Matrix Diagnostics
```julia
# Check covariance matrix properties
function diagnose_vcov(model)
    Σ = vcov(model)
    
    # Check positive definiteness
    eigenvals = eigvals(Σ)
    if any(eigenvals .< 1e-12)
        @warn "Covariance matrix near-singular - standard errors may be unreliable"
    end
    
    # Check condition number
    cond_num = cond(Σ)
    if cond_num > 1e12
        @warn "Poorly conditioned covariance matrix - numerical issues possible"
    end
    
    return (eigenvals=eigenvals, condition_number=cond_num)
end
```
