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
                                  groups=:country => (:region => :education))

# Four-level hierarchy: sector → company → department → position
organizational = population_margins(model, data;
                                  type=:effects, 
                                  groups=:sector => (:company => (:department => :position)))
```

### Parallel Grouping Within Hierarchy

Complex patterns combining hierarchical and cross-tabulated structures:

```julia
# Region first, then education×gender cross-tab within each region
parallel_nested = population_margins(model, data;
                                   type=:effects,
                                   groups=(:region => [:education, :gender]))

# Region first, then separate analyses for education levels AND income quartiles
mixed_parallel = population_margins(model, data;
                                  type=:effects,
                                  groups=(:region => [:education, (:income, 4)]))
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

### Bin Membership

Every bin is half-open — `[lower, upper)` — except the last, which is closed so the maximum
is included. Read the labels literally: a row whose income is exactly `25000` falls in
`"[25000, 50000)"`, not in `"< 25000"`. The bins therefore **partition the sample exactly**:
the reported `n` values always sum to the number of rows, and no observation is counted twice
or dropped.

### A Numeric Grouping Variable Needs Explicit Binning

`groups` accepts a bare `Symbol` only for a categorical variable, where the levels are the
groups. A numeric column does not say how it should be grouped, so it must be binned
explicitly:

```julia
# ERROR: MarginsError - :income is numeric, so this does not specify grouping
population_margins(model, data; type=:effects, vars=[:x], groups=:income)

# Say which binning you want
population_margins(model, data; type=:effects, vars=[:x], groups=(:income, 4))
population_margins(model, data; type=:effects, vars=[:x], groups=(:income, [25000, 50000]))

# Or group by levels, by making the variable categorical
data.income_band = categorical(data.income_band)
population_margins(model, data; type=:effects, vars=[:x], groups=:income_band)
```

This matters most for integer-valued columns such as counts: `groups=:nkids` on a 0–3 child
count is ambiguous between "four groups, one per level" and "quartiles of nkids", and the two
give different answers. Stata's `over()` and R marginaleffects' `by()` require a discrete
grouping variable for the same reason.

### Ties and Empty Subgroups

Quantile binning fails when a variable has enough tied values that two quantile boundaries
coincide, because the bin between them would contain no rows:

```julia
# income has a mass of zeros, so the 0th and 1st quartile boundaries are both 0.0
population_margins(model, data; type=:effects, vars=[:x], groups=(:income, 4))
# ERROR: MarginsError: Cannot form 4 quantile bins for 'income' ...
#        Use fewer bins, give explicit thresholds, or group by a categorical column.
```

Subgroups with no observations are **omitted from the results** rather than reported as
`0.0 ± 0.0`. This is routine for cross-tabulations, where not every level combination is
observed — `groups=[:region, :education]` returns a row only for the combinations that exist,
so `nrow` may be fewer than the product of the level counts.

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

See [Population Scenarios](population_scenarios.md) for detailed semantics and implementation notes on scenarios in population analysis.

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
    groups=(:state => (:urban_rural => [:insurance_type, (:income, 3)])),
    scenarios=(:aca_expansion = [0, 1], :medicaid_funding = [0.8, 1.2])
)

# Results: State × Urban/Rural × (Insurance×Income-Tertiles) × ACA×Medicaid scenarios
# Total combinations: 4 states × 2 urban/rural × 12 insurance×income × 4 policy scenarios = 384 results
```

## Important: Conflict Rule for Statistical Validity

**Critical Constraint:** For population analysis, computing the effect of a variable while simultaneously holding it fixed is contradictory and statistically meaningless.

### The Conflict Rule

The rule is about what a specification *does*, not about which names appear twice:

> A specification that **pins** a variable to a single value blocks that variable's own
> effect. A specification that **bins** it into ranges does not.

`scenarios` always pins — it overrides the variable's value for every row, leaving no
variation. Grouping by a categorical also pins: inside each subgroup the variable holds one
level. But grouping a numeric variable into quantiles or thresholds only *selects rows*; the
variable still varies inside each bin, so its average derivative there is well defined.

```julia
# REJECTED: g is pinned to one level inside each subgroup
population_margins(model, data; type=:effects, vars=[:g], groups=:g)

# REJECTED: scenarios override x for every row
population_margins(model, data; type=:effects, vars=[:x], scenarios=(x = [0, 1],))

# ALLOWED: x still varies within each quartile of x
population_margins(model, data; type=:effects, vars=[:x], groups=(:income, 4))
population_margins(model, data; type=:effects, vars=[:income], groups=(:income, 4))
```

The last line is the direct translation of Stata's `margins, dydx(income) over(incomebin)`:
the average marginal effect of income, computed within each quartile of income.

A variable cannot be binned by `groups` and overridden by `scenarios` at the same time — the
bins are built from the observed values that the override then discards — so that combination
is rejected as well.

`population_margins()` **rejects** an explicit `vars` entry that is pinned by `groups` or
`scenarios`, with an error explaining what to request instead.

```julia
# Example: x appears in both vars and scenarios
population_margins(model, data;
    type=:effects,
    vars=[:x, :z],            # Request effects for x and z
    scenarios=(x = [0, 1],)   # But fix x at specific values
)
# ERROR: ArgumentError: Invalid parameter combination: Variables x appear in
#        both 'vars' and 'scenarios'. ... (the message lists what to do instead)
```

The rule fires on **any** overlap, not only when it would leave nothing to compute: asking
for `vars=[:x, :z]` alongside `scenarios=(x = [0, 1],)` errors rather than quietly returning
only `z`. The same holds for `groups`.

### Why This Rule Exists

**Conceptual Problem:**
- **Marginal effect** asks: "What happens when x changes naturally?"
- **Scenario/Group** says: "Hold x fixed at specific values" or "Stratify by x levels"
- These two concepts are **mutually exclusive**

**Examples of Invalid Requests:**
```julia
# INVALID: "What's the effect of income while holding income fixed?"
population_margins(model, data;
    vars=[:income],                        # Effect of income changing
    scenarios=(income = [30000, 50000],)   # But income is fixed
)
# → ArgumentError

# INVALID: "What's the effect of education within education groups?"
population_margins(model, data;
    vars=[:education],         # Effect of education changing
    groups=:education          # But stratified by education levels
)
# → ArgumentError  (education is categorical: each subgroup holds one level)

# VALID: "What's the effect of income within income quartiles?"
population_margins(model, data;
    vars=[:income],            # Effect of income changing
    groups=(:income, 4)        # Within quartiles of income - income still varies inside each
)
# → one row per quartile
```

### Practical Alternatives

#### Alternative 1: Profile Analysis (for Stata users)

If you want Stata-style `dydx(x) over(x)` (derivative of x at different values of x), use **profile analysis**:

```julia
# Instead of: population_margins(model, data; vars=[:x], groups=:x)  # INVALID
# Use profile margins:
result = profile_margins(model, data, cartesian_grid(x=[10, 20, 30, 40]);
    type=:effects,
    vars=[:x]
)
# Computes marginal effect of x AT each specific value of x
```

#### Alternative 2: Effects Within Strata

If you want effects within strata of x, group by a **different variable** or compute effects of **other variables**:

```julia
# GOOD: Effects of z within education groups
result = population_margins(model, data;
    type=:effects,
    vars=[:z],              # Effect of z (not education)
    groups=:education       # Stratified by education
)

# GOOD: Effects within income quintiles
result = population_margins(model, data;
    type=:effects,
    vars=[:treatment],      # Effect of treatment (not income)
    groups=(:income, 5)     # Within income quintiles
)
```

#### Alternative 3: Counterfactual Predictions

If you want to see how outcomes change as x varies, use **predictions with scenarios**:

```julia
# Instead of: population_margins(model, data; vars=[:x], scenarios=(:x = [...]))
# Use predictions:
result = population_margins(model, data;
    type=:predictions,         # Not effects!
    scenarios=(:x = [10, 20, 30, 40])
)
# Shows predicted outcomes at each value of x
```

### Auto-Detected `vars`

The conflict rule applies to variables you name explicitly. When `vars` is omitted,
`population_margins()` detects them from the model, and a detected variable that a
`scenarios`/`groups` specification consumes is **skipped silently** — erroring there would
make a perfectly reasonable call fail because of a variable the user never asked about:

```julia
# x is auto-detected, then consumed by the grouping; g is still reported
result = population_margins(model, data; type=:effects, groups=(:x, 4))
unique(DataFrame(result).variable)   # ["g"] - x was skipped
```

If the specification consumes *every* auto-detected variable there is nothing left to
report, and the call raises a `MarginsError` saying so rather than returning an empty result:

```julia
# A model whose only variable is x, grouped by bins of x
population_margins(model, data; type=:effects, groups=(:x, 4))
# ERROR: MarginsError: No effects could be computed: every requested variable (x) is
#        also used by 'scenarios'/'groups' (x) ...
```

## Performance

### Computational Complexity

Population grouping maintains efficient O(n) scaling within each subgroup:

```julia
using BenchmarkTools

# Simple grouping: O(n/k) per group for k groups
@btime population_margins($model, $data; groups=:education)

# Complex hierarchical grouping: O(n/k) per final subgroup
@btime population_margins($model, $data; groups=(:region => (:education => :gender)))

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
                                 groups=(:region => [:education, (:income, 4)]))
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

**Hierarchical Grouping** (`groups=:var1 => :var2`):
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
groups=:var1 => [:var2, (:var4, 4)]
```

#### Empty Subgroups
```julia
# The framework automatically detects and errors on empty subgroups
# to maintain statistical validity
```

#### Conflict Rule Reference

See the dedicated section **"Important: Conflict Rule for Statistical Validity"** above for complete documentation on how `population_margins()` handles variables that appear in both `vars` and `groups`/`scenarios`.

#### Interpretation Complexity
```julia
# For presentation, consider simpler patterns:
presentation_analysis = population_margins(model, data;
                                         groups=:education,
                                         scenarios=(:policy = [0, 1]))

# For comprehensive analysis, use full complexity:
research_analysis = population_margins(model, data;
                                     groups=(:region => [:education, (:income, 4)]),
                                     scenarios=(:policy = [0, 1], :funding = [0.8, 1.2]))
```

---

*The population grouping framework enables sophisticated econometric analysis while maintaining computational efficiency and statistical rigor. For related details on scenarios and reference grids, see [Reference Grids](reference_grids.md) and for performance optimization, see [Performance Guide](performance.md).*
