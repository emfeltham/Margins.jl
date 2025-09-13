# Population Grouping Framework

*Comprehensive hierarchical analysis for stratified marginal effects*

## Conceptual Foundation

Margins.jl implements a population-based grouping framework that computes **average marginal effects (AME)** and **average adjusted predictions (AAP)** within stratified subgroups of the observed data.

### Core Design Principles

#### Population-Based Analysis
All operations maintain population averaging semantics - computing effects by averaging across actual or modified populations, not evaluating at synthetic representative points.

#### Orthogonal Parameters
Three independent dimensions combine multiplicatively:
- **`vars`**: Which variables to compute marginal effects for
- **`groups`**: How to stratify the analysis (data structure) 
- **`scenarios`**: What counterfactual scenarios to consider (data modification)

#### Single Fundamental Operation
All grouping reduces to: **stratify data into subgroups, compute population margins within each subgroup**.

## Basic Grouping Patterns

### Simple Categorical Grouping

Compute effects separately within each category of a grouping variable:

```julia
using Margins, DataFrames, GLM

# Effects by education level
education_effects = population_margins(model, data; 
                                     type=:effects, 
                                     groups=:education)

# Results: separate effects for each education category
DataFrame(education_effects)
```

### Cross-Tabulated Grouping

Analyze effects across combinations of multiple categorical variables:

```julia
# Effects by education × gender combinations
demographic_effects = population_margins(model, data;
                                        type=:effects,
                                        groups=[:education, :gender])

# Results: effects for (HS,Male), (HS,Female), (College,Male), (College,Female), etc.
```

## Advanced Hierarchical Grouping

### Nested Grouping with `=>` Operator

The `=>` operator creates hierarchical nesting where the right side is computed within each level of the left side:

```julia
# Region first, then education within each region
nested_effects = population_margins(model, data;
                                  type=:effects,
                                  groups=:region => :education)

# Results: (North,HS), (North,College), (South,HS), (South,College)
```

### Deep Hierarchical Nesting

Multiple levels of nesting support complex organizational structures:

```julia
# Three-level hierarchy: country → region → education
deep_hierarchy = population_margins(model, data;
                                  type=:effects,
                                  groups=:country = (:region => :education))

# Four-level hierarchy: sector → company → department → position
organizational = population_margins(model, data;
                                  type=:effects, 
                                  groups=:sector = (:company = (:department = :position)))
```

### Parallel Grouping Within Hierarchy

Complex patterns combining hierarchical and cross-tabulated structures:

```julia
# Region first, then education×gender cross-tab within each region
parallel_nested = population_margins(model, data;
                                   type=:effects,
                                   groups=:region = [:education, :gender])

# Region first, then separate analyses for education levels AND income quartiles
mixed_parallel = population_margins(model, data;
                                  type=:effects,
                                  groups=:region = [:education, (:income, 4)])
```

## Continuous Variable Binning

### Quantile-Based Binning

Automatic binning using quantiles with professional statistical terminology:

```julia
# Quartile analysis (Q1, Q2, Q3, Q4)
income_quartiles = population_margins(model, data;
                                    type=:effects,
                                    groups=(:income, 4))

# Tertile analysis (T1, T2, T3) 
score_tertiles = population_margins(model, data;
                                  type=:effects,
                                  groups=(:test_score, 3))

# Quintile analysis (P1, P2, P3, P4, P5)
wealth_quintiles = population_margins(model, data;
                                    type=:effects,
                                    groups=(:wealth, 5))
```

### Custom Threshold Binning

Policy-relevant thresholds using mathematical interval notation:

```julia
# Income brackets for tax policy analysis
tax_brackets = population_margins(model, data;
                                type=:effects,
                                groups=(:income, [25000, 50000, 75000]))

# Results: ["< 25000", "[25000, 50000)", "[50000, 75000)", ">= 75000"]

# Poverty line analysis
poverty_analysis = population_margins(model, data;
                                    type=:effects,
                                    groups=(:income, [federal_poverty_line]))

# Results: ["< 12880", ">= 12880"] (using 2023 federal poverty guideline)
```

### Mixed Categorical and Continuous Grouping

Combine categorical variables with binned continuous variables:

```julia
# Education levels × income quartiles
education_income = population_margins(model, data;
                                    type=:effects,
                                    groups=[:education, (:income, 4)])

# Results: (HS,Q1), (HS,Q2), (HS,Q3), (HS,Q4), (College,Q1), etc.

# Geographic region × age quintiles × gender
complex_demographics = population_margins(model, data;
                                        type=:effects,
                                        groups=[:region, (:age, 5), :gender])
```

## Counterfactual Scenario Analysis

### Policy Scenario Framework

The `scenarios` parameter modifies variable values for the entire population, creating counterfactual analyses:

```julia
# Binary treatment analysis
treatment_effects = population_margins(model, data;
                                     type=:effects,
                                     scenarios=(:treatment = [0, 1]))

# Multi-level policy scenarios
policy_scenarios = population_margins(model, data;
                                    type=:effects,
                                    scenarios=(:policy_level = ["none", "moderate", "aggressive"]))
```

### Multi-Variable Scenarios

Cartesian product expansion for complex policy analysis:

```julia
# Treatment × policy combinations
comprehensive_policy = population_margins(model, data;
                                        type=:effects,
                                        scenarios=(:treatment = [0, 1], 
                                                      :policy = ["current", "reform"]))

# Results: 4 scenarios (2×2 combinations)

# Three-dimensional policy space
complex_scenarios = population_margins(model, data;
                                     type=:effects,
                                     scenarios=(:treatment = [0, 1],
                                                   :funding = [0.8, 1.0, 1.2],
                                                   :regulation = ["light", "standard", "strict"]))

# Results: 18 scenarios (2×3×3 combinations)
```

## Combined Groups and Scenarios

### Comprehensive Policy Analysis

Groups and scenarios combine multiplicatively for complete analytical coverage:

```julia
# Demographics × policy scenarios
full_analysis = population_margins(model, data;
                                 type=:effects,
                                 groups=[:education, :region],
                                 scenarios=(:treatment = [0, 1]))

# Results: Each education×region combination under both treatment scenarios
```

### Advanced Applications

```julia
# Healthcare policy evaluation
healthcare_comprehensive = population_margins(health_model, health_data;
    type=:effects,
    groups=:state = (:urban_rural = [:insurance_type, (:income, 3)]),
    scenarios=(:aca_expansion = [0, 1], :medicaid_funding = [0.8, 1.2])
)

# Results: State × Urban/Rural × (Insurance×Income-Tertiles) × ACA×Medicaid scenarios
# Total combinations: 4 states × 2 urban/rural × 12 insurance×income × 4 policy scenarios = 384 results
```

## Real-World Applications

### Economic Policy Analysis

```julia
# Tax policy impact across income distribution
tax_analysis = population_margins(tax_model, tax_data;
    type=:effects,
    vars=[:after_tax_income],
    groups=(:pre_tax_income, 5),  # Income quintiles
    scenarios=(:tax_rate = [0.15, 0.25, 0.35], :deduction_cap = [10000, 25000])
)

# Labor market analysis with unemployment scenarios  
labor_analysis = population_margins(employment_model, labor_data;
    type=:effects,
    groups=[:education, (:experience, 4)],  # Education × experience quartiles
    scenarios=(:unemployment_rate = [3.0, 6.0, 9.0])  # Economic conditions
)
```

### Healthcare Research

```julia
# Treatment effectiveness across patient subgroups
clinical_analysis = population_margins(treatment_model, patient_data;
    type=:effects,
    groups=:hospital = [:condition_severity, (:age, 4)],
    scenarios=(:treatment_protocol = ["standard", "intensive"],
                   :resource_level = ["constrained", "adequate"])
)
```

### Educational Policy

```julia
# Educational intervention analysis
education_policy = population_margins(achievement_model, student_data;
    type=:effects,
    groups=:district = (:school_type = [:socioeconomic_status, (:baseline_score, 3)]),
    scenarios=(:intervention = [0, 1], :funding_increase = [0.0, 0.1, 0.2])
)
```

## Performance Characteristics

### Computational Complexity

Population grouping maintains efficient O(n) scaling within each subgroup:

```julia
using BenchmarkTools

# Simple grouping: O(n/k) per group for k groups
@btime population_margins($model, $data; groups=:education)

# Complex hierarchical grouping: O(n/k) per final subgroup
@btime population_margins($model, $data; groups=:region = (:education = :gender))

# With scenarios: same O(n/k) complexity repeated for each scenario
@btime population_margins($model, $data; groups=:education, scenarios=(:treatment = [0, 1]))
```

### Memory Efficiency

The grouping framework avoids data duplication through efficient indexing:

- **Subgroup filtering**: Uses DataFrame indexing, not data copying
- **Scenario modification**: Temporary overrides without permanent data changes  
- **Result aggregation**: Minimal memory footprint for result compilation

### Large Dataset Considerations

```julia
# For datasets >100k observations with many groups
# Consider selective analysis of key variables
key_analysis = population_margins(model, large_data;
                                type=:effects,
                                vars=[:primary_outcome],  # Limit variables
                                groups=(:income, 4))      # Manageable grouping

# Complex patterns still feasible for large n
complex_large = population_margins(model, large_data;
                                 type=:effects,
                                 groups=:region = [:education, (:income, 4)])
```

## Best Practices

### When to Use Different Grouping Patterns

**Simple Grouping** (`groups=:var`):
- Single dimension analysis
- Clear categorical divisions
- Straightforward interpretation needs

**Cross-Tabulation** (`groups=[:var1, :var2]`):
- Interaction effects important
- Policy targets multiple demographics simultaneously
- Comprehensive coverage needed

**Hierarchical Grouping** (`groups=:var1 = :var2`):
- Natural organizational structure exists
- Context matters (e.g., regions have different education systems)
- Nested decision-making processes

**Continuous Binning** (`groups=(:var, n)`):
- Policy-relevant thresholds exist
- Distribution-based analysis needed
- Quantile-based interpretation valuable

### Avoiding Common Pitfalls

#### Combination Explosion
```julia
# Dangerous: could create 1000s of combinations
# groups=[:var1, :var2, :var3, (:var4, 10), (:var5, 5)]

# Better: use hierarchical structure
groups=:var1 = [:var2, (:var4, 4)]
```

#### Empty Subgroups
```julia
# The framework automatically detects and errors on empty subgroups
# to maintain statistical validity
```

#### Skip Rule: vars also in groups or scenarios
- For population analysis, computing the effect of a variable while simultaneously holding it fixed (via `scenarios`) or using it to define subgroups (via `groups`) is contradictory.
- To preserve statistical correctness and interpretability, `population_margins` skips variables that appear in `vars` if they also appear in `groups` or `scenarios`.
- Practical guidance:
  - If you need Stata-style `dydx(x) over(x)`, consider whether you intend profile analysis instead. Use `profile_margins(..., at=...)` to evaluate derivatives at specific values of `x`.
  - If you want effects within strata of `x`, group by a coarser or external variable, or compute effects for other variables while stratifying by `x`.
  - If you want counterfactual predictions as `x` changes, use `type=:predictions` with `scenarios=(:x = [...])` and omit `x` from `vars`.

#### Interpretation Complexity
```julia
# For presentation, consider simpler patterns:
presentation_analysis = population_margins(model, data;
                                         groups=:education,
                                         scenarios=(:policy = [0, 1]))

# For comprehensive analysis, use full complexity:
research_analysis = population_margins(model, data;
                                     groups=:region = [:education, (:income, 4)],
                                     scenarios=(:policy = [0, 1], :funding = [0.8, 1.2]))
```

---

*The population grouping framework enables sophisticated econometric analysis while maintaining computational efficiency and statistical rigor. For implementation details, see the complete specification in `POP_GROUPING.md`. For performance optimization, see [Performance Guide](performance.md).*
