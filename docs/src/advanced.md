# Advanced Features

*Elasticities, robust standard errors, and specialized analysis techniques*

## Elasticities and Semi-Elasticities

Margins.jl provides comprehensive elasticity support through the `measure` parameter, following the same **Population vs Profile** framework as standard marginal effects. *For conceptual background on the 2×2 framework, see [Mathematical Foundation](mathematical_foundation.md).*

### Types of Elasticity Measures

#### Standard Elasticity
**Definition**: Percent change in Y per percent change in X
**Formula**: `(∂Y/∂X) × (X/Y)`
**Interpretation**: "A 1% increase in X leads to an ε% change in Y"

```julia
# Population average elasticities
population_margins(model, data; type=:effects, measure=:elasticity)

# Elasticities at sample means
profile_margins(model, data, means_grid(data); type=:effects, measure=:elasticity)
```

#### Semi-Elasticity with respect to X  
**Definition**: Percent change in Y per unit change in X
**Formula**: `(∂Y/∂X) × (1/Y)`
**Interpretation**: "A 1-unit increase in X leads to a (100×ε)% change in Y"

```julia
# Population average semi-elasticities (X)
population_margins(model, data; measure=:semielasticity_dyex)

# Semi-elasticities at specific scenarios
profile_margins(model, data, cartesian_grid(x1=[0,1,2]); measure=:semielasticity_dyex)
```

#### Semi-Elasticity with respect to Y
**Definition**: Unit change in Y per percent change in X  
**Formula**: `(∂Y/∂X) × X`
**Interpretation**: "A 1% increase in X leads to an ε-unit change in Y"

```julia
# Population average semi-elasticities (Y)
population_margins(model, data; measure=:semielasticity_eydx)
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
    log_wage = randn(1000) .+ 2.5,  # Log wages
    education = rand(12:20, 1000),   # Years of education  
    experience = rand(0:30, 1000),   # Years of experience
    age = rand(25:55, 1000)
)

model = lm(@formula(log_wage ~ education + experience + age), df)

# Education elasticity of wages (population average)
edu_elasticity = population_margins(model, df; 
                                   vars=[:education], 
                                   measure=:elasticity)
println("Population average education elasticity: ", DataFrame(edu_elasticity))

# Education elasticity at different experience levels (profile analysis)  
exp_scenarios = profile_margins(model, df,
                               cartesian_grid(experience=[0, 10, 20, 30]);
                               vars=[:education],
                               measure=:elasticity)
println("Education elasticity by experience level:")
println(DataFrame(exp_scenarios))

# Semi-elasticity: percent wage change per year of education
edu_semielast = population_margins(model, df;
                                  vars=[:education],
                                  measure=:semielasticity_dyex)
println("Education semi-elasticity: ", DataFrame(edu_semielast))
```

### When Profile ≠ Population for Elasticities

In **GLMs with non-identity links**, population and profile elasticities can differ substantially:

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

Margins.jl integrates seamlessly with Julia's robust standard error ecosystem, particularly [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl).

### Integration Philosophy

- **Responsibility separation**: Margins.jl computes marginal effects; CovarianceMatrices.jl computes robust covariances
- **Delta-method interface**: Margins only needs a parameter covariance matrix `Σ` for standard error computation
- **Model ecosystem compatibility**: Uses same covariance sources as GLM.jl/StatsModels.jl

### Basic Robust Standard Errors

#### Heteroskedasticity-Robust (White/Huber-White)
```julia
using CovarianceMatrices

# Fit model with robust standard errors
robust_model = glm(@formula(y ~ x1 + x2 + group), data, Normal(), vcov=HC1())

# Marginal effects automatically use robust covariance
robust_effects = population_margins(robust_model, data; type=:effects)
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
robust_model = glm(formula, data, family, vcov=HC3())
result = population_margins(robust_model, data)
```

### Clustered Standard Errors

#### Single-Level Clustering
```julia
# Cluster by firm ID
clustered_model = glm(@formula(wages ~ education + experience), data, 
                     Normal(), vcov=Clustered(:firm_id))

# Marginal effects with clustered standard errors
clustered_effects = population_margins(clustered_model, data; type=:effects)
```

#### Multi-Level Clustering  
```julia
# Two-way clustering (firm and year)
twoway_model = glm(formula, data, family, vcov=Clustered([:firm_id, :year]))
result = population_margins(twoway_model, data)
```

### HAC (Heteroskedasticity and Autocorrelation Consistent) Standard Errors

```julia
# Newey-West HAC estimator
hac_model = glm(@formula(returns ~ lagged_returns + market_factor), data,
               Normal(), vcov=HAC(kernel=:bartlett, bandwidth=4))

effects_hac = population_margins(hac_model, data; type=:effects)
```

### Manual Covariance Matrix Specification

#### Direct Matrix Input
```julia
# User-provided covariance matrix
custom_vcov = your_covariance_computation(model, data)

# Specify via vcov parameter (not yet implemented - placeholder)
result = population_margins(model, data; vcov=custom_vcov)
```

#### Function-Based Covariance
```julia
# Custom covariance function
function my_robust_vcov(model)
    # Custom robust covariance computation
    return robust_covariance_matrix
end

# Use custom function (not yet implemented - placeholder)
result = population_margins(model, data; vcov=my_robust_vcov)
```

### Robust Standard Errors with Elasticities

Robust standard errors work seamlessly with all elasticity measures:

```julia
# Robust elasticity estimates
robust_model = glm(formula, data, family, vcov=HC1())

# Population elasticities with robust SEs
robust_elasticities = population_margins(robust_model, data; 
                                        measure=:elasticity, 
                                        type=:effects)

# Profile elasticities with clustered SEs  
clustered_model = glm(formula, data, family, vcov=Clustered(:cluster_var))
profile_elasticities = profile_margins(clustered_model, data,
                                      means_grid(data);
                                      measure=:elasticity)
```

## Categorical Mixtures for Policy Analysis

Margins.jl supports **categorical mixtures** for realistic policy scenario analysis, allowing specification of population compositions rather than arbitrary category levels.

### Motivation: Realistic Population Scenarios

Traditional marginal effects often use arbitrary categorical values (e.g., "set all observations to treatment=1"). Categorical mixtures enable **realistic population compositions**:

```julia
using CategoricalArrays, Margins

# Instead of: "All treated" (unrealistic)
unrealistic = profile_margins(model, data, cartesian_grid(treatment=[1]); type=:predictions)

# Use: Realistic treatment rate  
realistic = profile_margins(model, data, 
                           DataFrame(treatment=[mix(0 => 0.3, 1 => 0.7)]))  # 70% treatment rate
```

### Frequency-Weighted Categorical Defaults

When categorical variables are unspecified in profiles, Margins.jl uses **actual sample frequencies** rather than arbitrary first levels:

```julia
# Data composition: education = 40% HS, 45% College, 15% Graduate
#                   region = 60% Urban, 40% Rural

# Effects "at means" uses realistic composition
result = profile_margins(model, data, means_grid(data); type=:effects)
# → Continuous vars: sample means
# → education: mix("HS" => 0.40, "College" => 0.45, "Graduate" => 0.15)  
# → region: mix("Urban" => 0.60, "Rural" => 0.40)
```

### Policy Scenario Analysis

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
    (treatment=0, education=mix("HS" => 0.5, "College" => 0.5)),
    (treatment=1, education=mix("HS" => 0.5, "College" => 0.5)),
    (treatment=0, education=mix("HS" => 0.2, "College" => 0.8)),  
    (treatment=1, education=mix("HS" => 0.2, "College" => 0.8))
])

results = profile_margins(model, data, treatment_scenarios; type=:predictions)
treatment_effects_df = DataFrame(results)
```

## Advanced Grouping and Stratification

Margins.jl provides a comprehensive grouping framework for population-based marginal effects analysis, supporting hierarchical stratification patterns that extend far beyond traditional approaches.

### Hierarchical Grouping Framework

#### Basic Grouping Patterns
```julia
# Simple categorical grouping
demographic_effects = population_margins(model, data;
                                        type=:effects,
                                        vars=[:income],
                                        groups=:education)

# Cross-tabulated grouping
cross_effects = population_margins(model, data;
                                 type=:effects,
                                 vars=[:income], 
                                 groups=[:education, :region])
```

#### Nested Hierarchical Grouping
```julia
# Hierarchical nesting: region → education within each region
nested_effects = population_margins(model, data;
                                  type=:effects,
                                  vars=[:income],
                                  groups=:region => :education)

# Deep nesting: region → urban → education
deep_nested = population_margins(model, data;
                               type=:effects,
                               groups=:region => (:urban => :education))
```

#### Continuous Variable Binning
```julia
# Quartile analysis
income_quartiles = population_margins(model, data;
                                    type=:effects,
                                    groups=(:income, 4))  # Q1, Q2, Q3, Q4

# Custom policy-relevant thresholds
policy_thresholds = population_margins(model, data;
                                     type=:effects,
                                     groups=(:income, [25000, 50000, 75000]))

# Mixed categorical and continuous
mixed_groups = population_margins(model, data;
                                type=:effects,
                                groups=[:education, (:income, 4)])
```

### Counterfactual Scenario Analysis

#### Policy Scenarios with Population Override
```julia
# Binary policy scenarios
policy_analysis = population_margins(model, data;
                                   type=:effects,
                                   vars=[:outcome],
                                   scenarios=Dict(:policy => [0, 1]))

# Multi-variable scenarios
complex_scenarios = population_margins(model, data;
                                     type=:effects,
                                     scenarios=Dict(:treatment => [0, 1], 
                                                   :policy => ["current", "reform"]))
```

#### Combined Grouping and Scenarios
```julia
# Comprehensive policy analysis: demographics × policy scenarios
full_analysis = population_margins(model, data;
                                 type=:effects,
                                 vars=[:outcome],
                                 groups=[:education, :region],
                                 scenarios=Dict(:treatment => [0, 1]))
```

### Complex Nested Patterns

#### Parallel Grouping Within Hierarchy
```julia
# Region → (education levels + income quartiles separately)
parallel_groups = population_margins(model, data;
                                   type=:effects,
                                   groups=:region => [:education, (:income, 4)])
```

#### Advanced Policy Applications
```julia
# Healthcare policy analysis
healthcare_analysis = population_margins(health_model, health_data;
    type=:effects,
    groups=:state => (:urban => [:insurance_type, (:income, 3)]),
    scenarios=Dict(:policy_reform => [0, 1], :funding_level => [0.8, 1.2])
)

# Results: State × Urban/Rural × (Insurance×Income-Tertiles) × Policy×Funding scenarios
```

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

---

*These advanced features enable sophisticated econometric analysis while maintaining Margins.jl's core principles of statistical correctness and computational efficiency. For conceptual foundation on when to use elasticities vs marginal effects, see [Mathematical Foundation](mathematical_foundation.md). For elasticity performance characteristics, see [Performance Guide](performance.md).*