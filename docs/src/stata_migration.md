# Stata Migration Guide

*Complete translation reference for economists migrating from Stata's `margins` command*

## Basic Command Translation

### Core Margins Commands

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins` | `population_margins(model, data; type=:predictions)` | Average adjusted predictions |
| `margins, dydx(*)` | `population_margins(model, data; type=:effects)` | Average marginal effects (AME) |
| `margins, at(means)` | `profile_margins(model, data, means_grid(data); type=:predictions)` | Predictions at sample means |
| `margins, at(means) dydx(*)` | `profile_margins(model, data, means_grid(data); type=:effects)` | Marginal effects at means (MEM) |
| `margins, dydx(*) atmeans` | `profile_margins(model, data, means_grid(data); type=:effects)` | Alternative MEM syntax |

### Variable Selection

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins, dydx(x1 x2)` | `population_margins(model, data; type=:effects, vars=[:x1, :x2])` | Specific variables only |
| `margins, dydx(_continuous)` | `population_margins(model, data; type=:effects)` | All continuous variables (automatic) |
| `margins, eyex(x1)` | `population_margins(model, data; type=:effects, vars=[:x1], measure=:elasticity)` | Elasticities |

## Grouping and Stratification

### Basic Grouping

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins education` | `population_margins(model, data; groups=:education)` | Group predictions |
| `margins education, dydx(*)` | `population_margins(model, data; type=:effects, groups=:education)` | Group effects |
| `margins, over(education)` | `population_margins(model, data; groups=:education)` | Alternative syntax |
| `margins education gender` | `population_margins(model, data; groups=[:education, :gender])` | Cross-tabulation |
| `margins education#gender` | `population_margins(model, data; groups=[:education, :gender])` | Interaction syntax |

### Nested Analysis

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `by region: margins education` | `population_margins(model, data; groups=:region => :education)` | Nested grouping |
| `margins education, over(region)` | `population_margins(model, data; groups=[:education, :region])` | Cross-tabulation alternative |

## Scenario Analysis (`at()` Specification)

### Basic Scenarios

| Stata Command | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins, at(x=0)` | `profile_margins(model, data, cartesian_grid(x=[0]); type=:predictions)` | Single scenario |
| `margins, at(x=(0 1 2))` | `profile_margins(model, data, cartesian_grid(x=[0, 1, 2]); type=:predictions)` | Multiple values |
| `margins, at(x=0 y=1)` | `profile_margins(model, data, cartesian_grid(x=[0], y=[1]); type=:predictions)` | Multiple variables |

### Population-Level Counterfactuals

**Key Difference**: Stata's `at()` creates evaluation points, while Margins.jl's `scenarios` creates population counterfactuals.

| Stata Approach | Margins.jl Population Approach | Notes |
|-----------------|-------------------------------|-------|
| `margins, at(treatment=(0 1))` | `population_margins(model, data; scenarios=(treatment=[0, 1]))` | Everyone untreated vs everyone treated |
| `margins education, at(policy=(0 1))` | `population_margins(model, data; groups=:education, scenarios=(policy=[0, 1]))` | Policy effects by education |

### Profile vs Population Interpretation

```julia
# Stata: margins, at(treatment=(0 1))
# → Effects at two evaluation points

# Margins.jl Profile Equivalent
profile_results = profile_margins(model, data, 
    cartesian_grid(treatment=[0, 1]);
    type=:effects)

# Margins.jl Population Alternative (often more relevant)  
population_results = population_margins(model, data;
    scenarios=(treatment=[0, 1]),
    type=:effects)
```

### Skip Rule: `dydx(x)` with `over(x)` (and scenarios)

Unlike Stata, `population_margins` intentionally skips computing the effect of a variable when that same variable appears in `groups` (Stata `over()`) or in `scenarios` (Stata `at()`). This avoids the contradiction of “compute the effect of x while holding x fixed” or “using x both as an effect variable and a grouping key.”

Recommended translations:

```julia
# 1) Stata: margins, dydx(x) over(x)
# → Profile-style alternative: evaluate derivatives at specific x values
mem_like = profile_margins(model, data,
    cartesian_grid(x=[-2.0, 0.0, 2.0]);
    type=:effects,
    vars=[:x])

# 2) Population stratification by x without contradiction:
#    Create a derived bin variable and group by it, not by :x directly
df.x_bin = cut(df.x, 4)  # quartiles via user code; or use groups=(:x, 4)
by_xbins = population_margins(model, df;
    type=:effects,
    vars=[:x],
    groups=:x_bin)  # allowed since groups variable ≠ :x

# 3) Effects of other variables within x strata (population approach)
effects_in_xbins = population_margins(model, data;
    type=:effects,
    vars=[:z, :w],
    groups=(:x, 4))

# 4) Counterfactual predictions as x changes (not effects of x)
preds_under_x = population_margins(model, data;
    type=:predictions,
    scenarios=(x=[-2.0, 0.0, 2.0]))
```

See also: “Skip Rule” note in the Population Grouping docs for rationale and guidance.

#### Short example: grouping by `x_bin` to compute `dydx(x)` across strata

```julia
using Random
using DataFrames, CategoricalArrays
using Statistics  # for quantile
using GLM
using Margins

Random.seed!(42)
n = 500
df = DataFrame(
    y = rand(Bool, n),
    x = randn(n),
    z = randn(n)
)

# Fit a simple model
m = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())

# Create quartile bins for x as a separate column "x_bin"
edges = quantile(df.x, 0:0.25:1.0)
labels = ["Q1", "Q2", "Q3", "Q4"]
df.x_bin = cut(df.x, edges; labels=labels, extend=true)

# Now compute population AME of x within x_bin strata (no contradiction)
res = population_margins(m, df;
    type=:effects,
    vars=[:x],
    groups=:x_bin)

DataFrame(res)  # Shows dydx(x) by Q1..Q4
```

## Combined Grouping and Scenarios

### Complex Analysis Patterns

| Stata Pattern | Margins.jl Equivalent | Notes |
|---------------|----------------------|-------|
| `margins education, at(treatment=(0 1))` | `population_margins(model, data; groups=:education, scenarios=(treatment=[0, 1]))` | Group × scenario analysis |
| Multiple `margins` commands | Single comprehensive call | More efficient in Julia |

## Advanced Patterns Beyond Stata

Margins.jl extends far beyond Stata's capabilities with features unavailable in Stata:

### Continuous Variable Binning

```stata
* Stata approach (manual and cumbersome):
gen income_q = .
_pctile income, nq(4)
replace income_q = 1 if income <= r(r1)
replace income_q = 2 if income > r(r1) & income <= r(r2)
replace income_q = 3 if income > r(r2) & income <= r(r3)
replace income_q = 4 if income > r(r3)
margins income_q
```

```julia
# Julia approach (automatic):
population_margins(model, data; groups=(:income, 4))  # Automatic Q1-Q4 quartiles
```

### Custom Policy Thresholds

```stata
* Stata approach:
gen income_bracket = .
replace income_bracket = 1 if income < 25000
replace income_bracket = 2 if income >= 25000 & income < 50000
replace income_bracket = 3 if income >= 50000 & income < 75000
replace income_bracket = 4 if income >= 75000
margins income_bracket
```

```julia
# Julia approach:
population_margins(model, data; groups=(:income, [25000, 50000, 75000]))
```

### Hierarchical Grouping

```stata
* Stata approach (requires multiple commands or complex by groups):
by region: margins education
* No native support for deep nesting
```

```julia
# Julia approach (native hierarchical support):
population_margins(model, data; groups=:region => :education)
population_margins(model, data; groups=:country => (:region => :education))  # Deep nesting
```

### Multi-Variable Scenarios

```stata
* Stata approach (requires multiple separate commands):
margins, at(treatment=0 policy=0)
margins, at(treatment=0 policy=1) 
margins, at(treatment=1 policy=0)
margins, at(treatment=1 policy=1)
```

```julia
# Julia approach (automatic Cartesian product):
population_margins(model, data; scenarios=(treatment=[0, 1], policy=[0, 1]))
```

Note: `scenarios` in Julia are population‑level counterfactuals (everyone receives each setting in turn). For Stata’s point‑evaluation semantics of `at()`, use `profile_margins(model, data, reference_grid)` with a grid builder (e.g., `means_grid`, `cartesian_grid`) or an explicit DataFrame.

## Complete Workflow Examples

### Example 1: Education Policy Analysis

#### Stata Workflow
```stata
* Fit model
logit outcome education income female urban policy_treatment

* Basic effects
margins, dydx(*)

* Effects by education
margins education, dydx(income)

* Policy scenarios (multiple commands required)
margins education, at(policy_treatment=0)
margins education, at(policy_treatment=1)

* Manual difference calculation needed for treatment effects
```

#### Julia Workflow
```julia
# Fit model  
model = glm(@formula(outcome ~ education + income + female + urban + policy_treatment),
            data, Binomial(), LogitLink())

# Basic effects
basic_effects = population_margins(model, data; type=:effects)

# Effects by education
education_effects = population_margins(model, data; 
                                     type=:effects, 
                                     vars=[:income],
                                     groups=:education)

# Policy scenarios (automatic treatment effect calculation)
policy_analysis = population_margins(model, data;
                                   type=:effects,
                                   groups=:education,
                                   scenarios=(:policy_treatment => [0, 1]))

# All results readily available as DataFrames
DataFrame(policy_analysis)
```

### Example 2: Complex Demographic Analysis

#### Stata Approach (Cumbersome)
```stata
* Multiple manual commands needed:
margins education, over(region)
margins gender, over(region)  
margins education#gender, over(region)

* Income quartiles require manual creation:
xtile income_q4 = income, nq(4)
margins education, over(income_q4)

* No native support for hierarchical analysis
```

#### Julia Approach (Comprehensive)
```julia
# Single comprehensive analysis
comprehensive_results = population_margins(model, data;
    type=:effects,
    groups=:region => [:education, :gender, (:income, 4)]
)

# Results: Region × (Education + Gender + Income-Quartiles) automatically computed
# Professional Q1-Q4 labeling included
DataFrame(comprehensive_results)
```

## Performance Comparisons

### Computational Advantages

| Aspect | Stata | Margins.jl |
|--------|-------|------------|
| **Complex grouping** | Multiple manual commands | Single comprehensive call |
| **Scenario analysis** | Manual looping/multiple commands | Automatic Cartesian products |
| **Large datasets** | Memory limitations | Efficient O(n) scaling |
| **Custom thresholds** | Manual variable creation | Automatic binning with labels |
| **Hierarchical analysis** | Limited native support | Unlimited nesting depth |

### Stata Command Count Reduction

```julia
# This single Julia command:
result = population_margins(model, data;
    groups=:region => [:education, (:income, 4)],
    scenarios=(treatment=[0, 1], policy=["old", "new"])
)

# Replaces ~30 individual Stata margins commands:
# 4 regions × 3 education × 4 income × 2 treatment × 2 policy = 192 combinations
# Plus manual variable creation, looping, and results compilation
```

## Migration Best Practices

### Workflow Translation Strategy

1. **Start with basic commands**: Translate simple `margins` and `margins, dydx(*)` first
2. **Identify grouping patterns**: Map Stata `over()` and `by:` to Julia `groups`  
3. **Distinguish scenarios vs profiles**: Decide whether Stata `at()` should become Julia `scenarios` or `at`
4. **Leverage advanced features**: Use continuous binning and hierarchical grouping where beneficial
5. **Consolidate analyses**: Combine multiple Stata commands into single Julia calls

### Common Translation Patterns

```julia
# Pattern 1: Simple margins → population_margins
# margins → population_margins(model, data; type=:predictions)

# Pattern 2: Effects by groups → groups parameter  
# margins education, dydx(*) → population_margins(model, data; type=:effects, groups=:education)

# Pattern 3: Multiple at() values → scenarios or profile grids
# margins, at(x=(0 1 2)) → profile_margins(model, data, cartesian_grid(x=[0, 1, 2]))
# OR population_margins(model, data; scenarios=(x=[0, 1, 2]))  # for counterfactuals

# Pattern 4: Complex manual analysis → comprehensive single call
# Multiple Stata commands → single population_margins with groups + scenarios
```

### Verification Strategy

```julia
# Verify translation accuracy by comparing key results:
# 1. Basic AME should match Stata margins, dydx(*)
# 2. Group means should match Stata margins groupvar
# 3. Scenario analysis should match Stata at() where comparable
```

## Julia Advantages for Economists

### Research Productivity

- **Fewer commands**: Complex analyses require single function calls
- **Automatic labeling**: Professional statistical terminology (Q1-Q4, T1-T3, P1-P5)
- **Integrated workflows**: Results immediately available as DataFrames for further analysis
- **Reproducible research**: Single script replaces multiple Stata command sequences

### Advanced Capabilities

- **Hierarchical analysis**: Organizational structures represented naturally
- **Policy thresholds**: Custom economically-relevant cutpoints without manual coding
- **Scenario space exploration**: Automatic Cartesian product expansion
- **Large dataset support**: Memory-efficient computation for modern data sizes

### Statistical Rigor  

- **Zero tolerance for statistical errors**: Package errors rather than producing invalid results
- **Bootstrap validation**: All standard errors validated against bootstrap estimates  
- **Delta method implementation**: Proper covariance matrix handling throughout
- **Publication quality**: Results meet econometric publication standards

---

*This guide provides complete translation patterns for migrating from Stata to Margins.jl. The Julia approach often simplifies complex analyses while providing more sophisticated capabilities. For detailed examples, see [Examples](examples.md). For comprehensive grouping documentation, see [Population Grouping Framework](grouping.md).*
