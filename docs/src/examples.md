# Examples

*Comprehensive workflow examples and profile specification patterns*

## Quick Start Example

```julia
using Margins, DataFrames, GLM

# Generate sample data
n = 1000
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
mem_result = profile_margins(model, df; at=:means, type=:effects)
DataFrame(mem_result)
```

## Profile Specification Patterns

Margins.jl provides multiple ways to specify evaluation profiles for `profile_margins()`, each optimized for different use cases.

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
predictions = profile_margins(model, reference_grid; type=:predictions)

# Effects at specific points  
effects = profile_margins(model, reference_grid; type=:effects, vars=[:x1, :x2])
```

### 2. Cartesian Product Specification

For systematic scenario construction, use dictionaries to specify value combinations:

```julia
# Systematic scenario grid (Cartesian product)
scenarios = Dict(
    :x1 => [-1.0, 0.0, 1.0],
    :x2 => [10, 20], 
    :group => ["A", "B"]
)
# Creates 3×2×2 = 12 evaluation points

# Effects across all scenarios
scenario_effects = profile_margins(model, df; 
    at=scenarios, 
    type=:effects, 
    vars=[:x1]
)
DataFrame(scenario_effects)
```

### 3. At Sample Means (Most Common)

For representative case analysis:

```julia
# Effects at sample means - most interpretable approach
means_effects = profile_margins(model, df; at=:means, type=:effects)

# Predictions at sample means
means_predictions = profile_margins(model, df; at=:means, type=:predictions)
```

### 4. Explicit Profile Lists

For irregular or custom evaluation points:

```julia
# Explicit profile specification
custom_profiles = [
    Dict(:x1 => -1.0, :x2 => 10, :group => "A"),
    Dict(:x1 =>  0.0, :x2 => 15, :group => "B"), 
    Dict(:x1 =>  1.0, :x2 => 20, :group => "A")
]

results = profile_margins(model, df; at=custom_profiles, type=:effects)
```

### 5. Categorical Mixtures for Policy Analysis

For realistic population scenarios using categorical mixtures:

```julia
using CategoricalArrays

# Realistic policy scenarios with population composition
policy_scenario = profile_margins(model, df;
    at=Dict(:group => mix("A" => 0.5, "B" => 0.3, "C" => 0.2)),
    type=:predictions
)

# Multiple policy scenarios
policy_effects = profile_margins(model, df;
    at=Dict(
        :x1 => [0, 1],  # Policy intervention levels
        :group => mix("A" => 0.6, "B" => 0.4)  # Target population
    ),
    type=:effects
)
```

## Economic Analysis Workflow

### Wage Determination Analysis

Complete econometric workflow using human capital theory:

```julia
using GLM, CategoricalArrays, Random

# Generate realistic econometric dataset
Random.seed!(42)
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
mem_results = profile_margins(wage_model, data; at=:means, type=:effects)
println("Effects for typical person:")
println(DataFrame(mem_results))

# Policy scenarios: education and unemployment effects
policy_analysis = profile_margins(wage_model, data;
    at=Dict(
        :education => ["HS", "College", "Graduate"],
        :unemployment_rate => [3.0, 6.0, 9.0]  # Recession scenarios
    ),
    type=:predictions
)
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
                                target=:mu)
println("Effects on probability of management position:")
println(DataFrame(prob_effects))

# Gender gap analysis across education levels
gender_gap = profile_margins(logit_model, data;
    at=Dict(
        :education => ["HS", "College", "Graduate"],
        :female => [0, 1]
    ),
    type=:predictions,
    target=:mu
)
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
edu_elasticities = profile_margins(wage_model, data;
    at=Dict(:education => ["HS", "College", "Graduate"]),
    type=:effects,
    measure=:elasticity,
    vars=[:age, :experience]
)
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

```julia
# Policy scenarios: unemployment rate effects
recession_scenarios = population_margins(wage_model, data;
                                       type=:effects,
                                       scenarios=Dict(:unemployment_rate => [3.0, 6.0, 12.0]))

# Combined grouping and scenarios
education_recession = population_margins(wage_model, data;
                                       type=:effects,
                                       groups=:education,
                                       scenarios=Dict(:unemployment_rate => [3.0, 12.0]))

# Multi-variable scenarios
complex_policy = population_margins(wage_model, data;
                                  type=:effects,
                                  scenarios=Dict(:urban => [0, 1], 
                                               :unemployment_rate => [3.0, 9.0]))
```

### Robust Standard Errors

```julia
using CovarianceMatrices

# Heteroskedasticity-robust standard errors
robust_model = glm(@formula(log_wage ~ age + education + experience + female), 
                   data, Normal(), vcov=HC1())

robust_effects = population_margins(robust_model, data; type=:effects)
println("Robust standard errors:")
println(DataFrame(robust_effects))
```

### Performance Comparison

```julia
using BenchmarkTools

# Profile margins: O(1) constant time
println("Profile margins performance (constant time):")
@btime profile_margins($wage_model, $data; at=:means, type=:effects)

# Population margins: O(n) scaling  
println("Population margins performance (scales with n):")
@btime population_margins($wage_model, $data; type=:effects)

# Complex scenario analysis (still O(1) for profiles)
complex_scenarios = Dict(
    :age => [25, 35, 45, 55],
    :education => ["HS", "College", "Graduate"],
    :urban => [0, 1]
)
println("Complex scenario performance (24 profiles, still O(1)):")
@btime profile_margins($wage_model, $data; at=$complex_scenarios, type=:effects)
```

## Stata Migration Examples

Direct equivalency for economists familiar with Stata:

```julia
# Stata: margins, dydx(*)
stata_ame = population_margins(wage_model, data; type=:effects)

# Stata: margins, at(means) dydx(*)  
stata_mem = profile_margins(wage_model, data; at=:means, type=:effects)

# Stata: margins, at(age=(25 35 45) education=(1 2 3))
stata_scenarios = profile_margins(wage_model, data;
    at=Dict(:age => [25, 35, 45], :education => ["HS", "College", "Graduate"]),
    type=:effects
)

# Stata: margins, over(female)
stata_subgroups = population_margins(wage_model, data; 
                                   type=:effects, 
                                   groups=:female)
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
large_data_profiles = profile_margins(model, large_data; at=:means, type=:effects)

# Population margins scale linearly - use selectively for very large data
key_population_effects = population_margins(model, large_data; 
                                          vars=[:key_variable], 
                                          type=:effects)
```

### Error Handling

```julia
# Graceful error handling for production workflows
function robust_margins_analysis(model, data)
    try
        # Try high-accuracy automatic differentiation
        return population_margins(model, data; backend=:ad, type=:effects)
    catch e
        @warn "AD backend failed, using finite differences" exception=e
        return population_margins(model, data; backend=:fd, type=:effects)
    end
end
```

---

*These examples demonstrate the full range of Margins.jl capabilities. For detailed API documentation, see [API Reference](api.md). For performance optimization, see [Performance Guide](performance.md).*