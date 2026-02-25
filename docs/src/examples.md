# Examples

*Comprehensive workflow examples and implementation patterns*

## Conceptual Overview

### Example Organization

This guide demonstrates practical implementation of the two-dimensional analytical framework through concrete examples. Examples progress from basic usage patterns to advanced specification techniques, illustrating both population and profile analysis approaches across diverse econometric applications.

## Basic Implementation

### Fundamental Usage Pattern

```julia
using Random
using Margins, DataFrames, GLM

# Generate sample data
n = 1000
Random.seed!(06515)
df = DataFrame(
    y = randn(n),
    x1 = randn(n), 
    x2 = randn(n),
    group = rand(["A", "B", "C"], n)
)

# Fit model
model = lm(@formula(y ~ x1 + x2 + group), df)

# Population average marginal effects (AME)
ame_result = population_margins(model, df; type=:effects)
DataFrame(ame_result)

# Marginal effects at sample means (MEM)  
mem_result = profile_margins(model, df, means_grid(df); type=:effects)
DataFrame(mem_result)
```

## Advanced Implementation Patterns

### Profile Specification Methods

Margins.jl provides multiple approaches to specify evaluation profiles for `profile_margins()`, each optimized for different analytical requirements.

### 1. Table-Based Reference Grid (Maximum Control)

For exact control over evaluation points, pass a DataFrame directly:

```julia
using DataFrames

# Custom reference grid
reference_grid = DataFrame(
    x1 = [-1.0, 0.0, 1.0],
    x2 = [10, 15, 20],
    group = ["A", "B", "A"]  # Row order preserved as specified
)

# Predictions at specific points
predictions = profile_margins(model, df, reference_grid; type=:predictions)

# Effects at specific points  
effects = profile_margins(model, df, reference_grid; type=:effects, vars=[:x1, :x2])
```

### 2. Cartesian Product Specification

For systematic scenario construction, use grid builders to specify value combinations:

```julia
# Systematic scenario grid (Cartesian product)
scenarios = cartesian_grid(x1=[-1.0, 0.0, 1.0], x2=[10, 20], group=["A", "B"])  
# Creates 3×2×2 = 12 evaluation points

# Effects across all scenarios
scenario_effects = profile_margins(model, df, scenarios; type=:effects, vars=[:x1])
DataFrame(scenario_effects)
```

### 3. At Sample Means (Most Common)

For representative case analysis:

```julia
# Effects at sample means - most interpretable approach
means_effects = profile_margins(model, df, means_grid(df); type=:effects)

# Predictions at sample means
means_predictions = profile_margins(model, df, means_grid(df); type=:predictions)
```

### 4. Explicit Profile Tables

For irregular or custom evaluation points, pass an explicit DataFrame:

```julia
custom_profiles = DataFrame(
    x1 = [-1.0, 0.0, 1.0],
    x2 = [10, 15, 20],
    group = ["A", "B", "A"]
)

results = profile_margins(model, df, custom_profiles; type=:effects)
```

### 5. Categorical Mixtures for Policy Analysis

For realistic population scenarios using categorical mixtures:

```julia
using CategoricalArrays

# Realistic policy scenarios with population composition
mixture_grid = DataFrame(group=[mix("A" => 0.5, "B" => 0.3, "C" => 0.2)])
policy_scenario = profile_margins(model, df, mixture_grid; type=:predictions)

# Multiple policy scenarios
policy_grid = DataFrame(
    x1 = [0, 1],  # Policy intervention levels
    group = [mix("A" => 0.6, "B" => 0.4), mix("A" => 0.6, "B" => 0.4)]
)
policy_effects = profile_margins(model, df, policy_grid; type=:effects)
```

## Economic Analysis Workflow

### Wage Determination Analysis

Complete econometric workflow using human capital theory:

```julia
using GLM, CategoricalArrays, Random

# Generate realistic econometric dataset
Random.seed!(06515)
n = 2000

data = DataFrame(
    # Demographics
    age = rand(25:65, n),
    female = rand([0, 1], n),
    education = categorical(rand(["HS", "College", "Graduate"], n)),
    
    # Economic variables
    experience = rand(0:40, n),
    urban = rand([0, 1], n),
    unemployment_rate = rand(3.0:0.1:12.0, n)
)

# Generate realistic log wages
education_effects = Dict("HS" => 0.0, "College" => 0.4, "Graduate" => 0.8)
edu_numeric = [education_effects[string(edu)] for edu in data.education]

data.log_wage = 1.5 .+ 
                0.05 .* data.age .+ 
                edu_numeric .+ 
                0.02 .* data.experience .- 
                0.15 .* data.female .+ 
                0.10 .* data.urban .- 
                0.03 .* data.unemployment_rate .+ 
                0.3 .* randn(n)

# Fit wage equation
wage_model = lm(@formula(log_wage ~ age + education + experience + 
                        female + urban + unemployment_rate), data)
```

#### Population Analysis

```julia
# Population average marginal effects
ame_results = population_margins(wage_model, data; type=:effects)
println("Population Average Marginal Effects:")
println(DataFrame(ame_results))

# Effects by gender subgroups  
gender_effects = population_margins(wage_model, data; 
                                  type=:effects, 
                                  groups=:female)
println("Effects by gender:")
println(DataFrame(gender_effects))
```

#### Profile Analysis

```julia
# Effects at sample means (representative person)
mem_results = profile_margins(wage_model, data, means_grid(data); type=:effects)
println("Effects for typical person:")
println(DataFrame(mem_results))

# Policy scenarios: education and unemployment effects
policy_grid = cartesian_grid(education=["HS", "College", "Graduate"],
                             unemployment_rate=[3.0, 6.0, 9.0])
policy_analysis = profile_margins(wage_model, data, policy_grid; type=:predictions)
println("Policy scenario predictions:")
println(DataFrame(policy_analysis))
```

### Logistic Regression Example

Binary outcome analysis with proper probability interpretation:

```julia
# Generate binary outcome data
data.manager = [rand() < (1/(1+exp(-(-1.0 + 0.03*age + 0.5*edu + 0.02*exp - 0.3*fem)))) ? 1 : 0 
                for (age,edu,exp,fem) in zip(data.age, edu_numeric, data.experience, data.female)]

# Fit logistic model
logit_model = glm(@formula(manager ~ age + education + experience + female), 
                  data, Binomial(), LogitLink())

# Effects on probability scale (most interpretable)
prob_effects = population_margins(logit_model, data; 
                                type=:effects, 
                                scale=:response)
println("Effects on probability of management position:")
println(DataFrame(prob_effects))

# Gender gap analysis across education levels
gender_grid = cartesian_grid(education=["HS", "College", "Graduate"], female=[0, 1])
gender_gap = profile_margins(logit_model, data, gender_grid;
    type=:predictions, scale=:response)
println("Gender gap in management probability by education:")
println(DataFrame(gender_gap))
```

## Elasticity Analysis

### Basic Elasticity Computation

```julia
# Population average elasticities
elasticities = population_margins(wage_model, data; 
                                type=:effects, 
                                measure=:elasticity,
                                vars=[:age, :experience])
println("Population average elasticities:")
println(DataFrame(elasticities))

# Elasticities at different education levels
edu_grid = cartesian_grid(education=["HS", "College", "Graduate"]) 
edu_elasticities = profile_margins(wage_model, data, edu_grid;
    type=:effects, measure=:elasticity, vars=[:age, :experience])
println("Elasticities by education level:")
println(DataFrame(edu_elasticities))
```

### Semi-Elasticities

```julia
# Semi-elasticity: % change in wages per unit change in unemployment
unemployment_semi = population_margins(wage_model, data;
                                     measure=:semielasticity_dyex,
                                     vars=[:unemployment_rate])
println("Unemployment semi-elasticity (% wage change per point):")
println(DataFrame(unemployment_semi))
```

## Advanced Features

### Advanced Grouping and Stratification

```julia
# Basic categorical grouping
urban_analysis = population_margins(wage_model, data; 
                                  type=:effects, 
                                  groups=:urban)

# Cross-tabulated grouping
education_urban = population_margins(wage_model, data; 
                                   type=:effects, 
                                   groups=[:education, :urban])

# Hierarchical grouping: education → urban within each education level
nested_analysis = population_margins(wage_model, data;
                                   type=:effects,
                                   groups=:education => :urban)

# Continuous binning: age quartiles
age_quartiles = population_margins(wage_model, data;
                                 type=:effects,
                                 groups=(:age, 4))

# Custom thresholds for policy analysis
income_thresholds = population_margins(wage_model, data;
                                     type=:effects,
                                     groups=(:log_wage, [2.0, 2.5, 3.0]))

# Mixed categorical and continuous
complex_groups = population_margins(wage_model, data;
                                  type=:effects,
                                  groups=[:education, (:age, 4)])
```

### Counterfactual Scenario Analysis

#### Skip Rule Example: dydx(x) across x strata using a derived bin variable

```julia
using Statistics
using CategoricalArrays

# Suppose we want dydx(age) across age strata without holding age fixed or using it as the grouping key directly.
# Create an "age_bin" column (quartiles), then group by that derived column:
edges = quantile(data.age, 0:0.25:1.0)
labels = ["Q1", "Q2", "Q3", "Q4"]
data.age_bin = cut(data.age, edges; labels=labels, extend=true)

age_effects_by_bin = population_margins(wage_model, data;
    type=:effects,
    vars=[:age],
    groups=:age_bin)

DataFrame(age_effects_by_bin)
```

```julia
# Policy scenarios: unemployment rate effects
recession_scenarios = population_margins(wage_model, data;
                                       type=:effects,
                                       scenarios=(:unemployment_rate => [3.0, 6.0, 12.0]))

# Combined grouping and scenarios
education_recession = population_margins(wage_model, data;
                                       type=:effects,
                                       groups=:education,
                                       scenarios=(:unemployment_rate => [3.0, 12.0]))

# Multi-variable scenarios
complex_policy = population_margins(wage_model, data;
                                  type=:effects,
                                  scenarios=(:urban => [0, 1], 
                                               :unemployment_rate => [3.0, 9.0]))
```

### Robust Standard Errors

```julia
using CovarianceMatrices

# Heteroskedasticity-robust standard errors (HC1)
robust_effects = population_margins(wage_model, data; vcov=HC1(), type=:effects)
println("Robust standard errors:")
println(DataFrame(robust_effects))
```

### Performance Comparison

```julia
using BenchmarkTools

# Profile margins: O(1) constant time
println("Profile margins performance (constant time):")
@btime profile_margins($wage_model, $data, means_grid($data); type=:effects)

# Population margins: O(n) scaling  
println("Population margins performance (scales with n):")
@btime population_margins($wage_model, $data; type=:effects)

# Complex scenario analysis (still O(1) for profiles)
complex_scenarios = cartesian_grid(age=[25, 35, 45, 55],
                                   education=["HS", "College", "Graduate"],
                                   urban=[0, 1])
println("Complex scenario performance (24 profiles, still O(1)):")
@btime profile_margins($wage_model, $data, $complex_scenarios; type=:effects)
```

## Stata Migration Examples

Direct equivalency for economists familiar with Stata:

```julia
# Stata: margins, dydx(*)
stata_ame = population_margins(wage_model, data; type=:effects)

# Stata: margins, at(means) dydx(*)  
stata_mem = profile_margins(wage_model, data, means_grid(data); type=:effects)

# Stata: margins, at(age=(25 35 45) education=(1 2 3))
stata_grid = cartesian_grid(age=[25, 35, 45], education=["HS", "College", "Graduate"]) 
stata_scenarios = profile_margins(wage_model, data, stata_grid; type=:effects)

# Stata: margins, over(female)
stata_subgroups = population_margins(wage_model, data; 
                                   type=:effects, 
                                   groups=:female)
```

## MixedModels.jl Examples

Minimal linear and generalized linear mixed models with population analysis.

```julia
# Illustrative example (not executed in docs CI): MixedModels integration
using Random
using DataFrames, CategoricalArrays, MixedModels, StatsModels, Margins

# Synthetic random-intercept dataset
Random.seed!(42)
n_groups = 20; n_per = 30; n = n_groups * n_per
group = repeat(1:n_groups, inner=n_per)
x = randn(n)
u = randn(n_groups)  # random intercepts
y = 1.0 .+ 0.5 .* x .+ u[group] .+ 0.2 .* randn(n)
df = DataFrame(y=y, x=x, group=categorical(string.(group)))

# Linear mixed model
lmm = fit(MixedModel, @formula(y ~ 1 + x + (1 | group)), df)

# Population AME for x (averaged across sample distribution)
ame_lmm = population_margins(lmm, df; type=:effects, vars=[:x])

# Generalized linear mixed model (binary outcome)
η = -0.5 .+ 1.2 .* x .+ u[group]
p = 1.0 ./ (1 .+ exp.(-η))
ybin = rand.(Bernoulli.(p))
df_bin = DataFrame(y=ybin, x=x, group=df.group)

glmm = GeneralizedLinearMixedModel(@formula(y ~ 1 + x + (1 | group)), df_bin, Binomial()) |> fit!

# Probability-scale effects
prob_effects_glmm = population_margins(glmm, df_bin; type=:effects, vars=[:x], scale=:response)
```

## Best Practices

### When to Use Population vs Profile

**Choose Population Analysis When**:
- Estimating true average effects across your sample
- Sample heterogeneity is important for policy
- External validity to similar populations is the goal
- Broad policy applications affecting diverse groups

**Choose Profile Analysis When**:
- Understanding specific, concrete scenarios  
- Communicating results to non-technical audiences
- Sample is relatively homogeneous
- Policy targets specific demographic profiles

### Performance Guidelines

```julia
# For large datasets (>100k observations)
# Profile margins remain fast regardless of size
large_data_profiles = profile_margins(model, large_data, means_grid(large_data); type=:effects)

# Population margins scale linearly - use selectively for very large data
key_population_effects = population_margins(model, large_data; 
                                          vars=[:key_variable], 
                                          type=:effects)
```

### Error Handling Best Practices

**Important:** Margins.jl follows an **error-first philosophy** - when statistical correctness cannot be guaranteed, the package errors explicitly rather than producing potentially invalid results. This ensures users are aware of problems rather than receiving plausible-but-wrong statistical output.

```julia
# GOOD: Let errors propagate to inform users of issues
function analyze_margins(model, data, vars)
    # Errors will propagate with clear messages
    result = population_margins(model, data; type=:effects, vars=vars)
    return DataFrame(result)
end

# GOOD: Validate inputs before computation
function safe_margins_analysis(model, data, vars)
    # Check that variables exist in data
    data_vars = Set(Symbol.(names(data)))
    missing_vars = setdiff(Set(vars), data_vars)

    if !isempty(missing_vars)
        error("Variables not found in data: $(collect(missing_vars))")
    end

    # Let computation errors propagate naturally
    return population_margins(model, data; type=:effects, vars=vars)
end

# BAD: Silent fallbacks violate error-first philosophy
# function bad_margins_analysis(model, data)
#     try
#         return population_margins(model, data; backend=:ad)
#     catch e
#         @warn "AD failed, using FD"  # User doesn't know why AD failed!
#         return population_margins(model, data; backend=:fd)  # May produce different results!
#     end
# end
#
# Why this is bad:
# 1. Silently switches backends without user awareness
# 2. May hide underlying data quality issues
# 3. Results from AD vs FD may differ slightly
# 4. Violates principle: "Error out rather than approximate"
```

**Guideline:** If you encounter errors during marginal effects computation, investigate and fix the root cause rather than implementing silent fallbacks. Common issues include:
- Domain errors (use variables that stay positive for log/sqrt)
- Missing variables (validate inputs before computation)
- Model convergence issues (check model fit quality)
- Data quality problems (check for NaN/Inf values)

---

*These examples demonstrate the full range of Margins.jl capabilities. For detailed API documentation, see [API Reference](api.md). For performance optimization, see [Performance Guide](performance.md).*
