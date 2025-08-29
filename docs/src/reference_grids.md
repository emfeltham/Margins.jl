# Reference Grid Specification

Reference grids define the covariate scenarios where marginal effects and predictions are evaluated in profile-based analysis. Margins.jl provides a flexible, unified system for specifying reference grids through two main approaches:

1. **Dict-based specification** via the `at` parameter
2. **Table-based specification** via direct DataFrame input

## Overview

### Dict-Based Approach

Specify scenarios using the `at` parameter with various Dict formats:

```julia
profile_margins(model, data; at=scenarios, type=:effects, ...)
```

### Table-Based Approach

Specify scenarios directly as a DataFrame for maximum control:

```julia
reference_grid = DataFrame(
    x1 = [0.0, 1.0, 0.0, 1.0],
    x2 = [0.5, 0.5, -0.5, -0.5],
    treated = [true, false, true, false]
)
profile_margins(model, data, reference_grid; type=:effects, ...)
```

Both approaches use a unified architecture that flows through `_build_profiles()` or direct conversion → `FormulaCompiler.create_scenario()`.

## Dict-Based Specification Methods

### 1. Cartesian Product Grids

Create all combinations of specified values across variables:

```julia
# 3×2 = 6 scenarios: all combinations of x and z values
at = Dict(:x => [-1, 0, 1], :z => [-0.5, 0.5])

# Results in scenarios:
# (-1, -0.5), (-1, 0.5), (0, -0.5), (0, 0.5), (1, -0.5), (1, 0.5)
```

### 2. Summary Statistics

Use built-in statistical summaries:

```julia
at = Dict(
    :x => :mean,     # Mean of x
    :z => :median,   # Median of z  
    :w => :p75       # 75th percentile of w
)

# Supported statistics: :mean, :median, :p10, :p25, :p50, :p75, :p90, etc.
```

### 3. String-Based Ranges

Stata-style numlist notation for sequences:

```julia
at = Dict(
    :x => "10(5)30",    # [10, 15, 20, 25, 30] - start(step)end
    :z => "0(0.1)0.5"   # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
)

# Also supports comma/space-separated values
at = Dict(:x => "1, 2.5, 4, 6.5")
```

### 4. Multiple Profile Blocks

Concatenate different scenario sets:

```julia
at = [
    Dict(:x => [0], :z => [-1, 1]),      # 2 scenarios: (0,-1), (0,1)
    Dict(:x => [1, 2], :z => [0])        # 2 scenarios: (1,0), (2,0)  
]
# Total: 4 scenarios
```

### 5. Global + Specific Settings

Apply defaults to all numeric variables, then override specific ones:

```julia
at = Dict(
    :all => :mean,                    # Set all Real columns to their means
    :x => [-2, -1, 0, 1, 2],         # Override x with specific grid
    :treatment => [0, 1],             # Override categorical variable
    :income => :p25                   # Override with 25th percentile
)
```

The `:all` key applies to all `Real` columns in the data, then specific variable settings take precedence.

## Special Cases

### Built-in Shortcuts

```julia
# Representative values at sample means
at = :means

# Per-row evaluation (empty grid)
at = :none
```

### Boolean Variables

Boolean variables can accept fractional values for population composition:

```julia
at = Dict(
    :male => [0.0, 0.47, 1.0],      # 0%, 47%, 100% male scenarios
    :urban => 0.65                   # 65% urban scenario
)
```

This allows modeling scenarios with different population compositions rather than just individual-level contrasts.

### Categorical Mixtures

For categorical variables, Margins.jl supports **categorical mixtures** that specify population composition scenarios using the `mix()` function. This extends fractional specification beyond boolean variables to multi-level categorical variables.

#### Basic Categorical Mixture Syntax

```julia
# Specify population composition for education levels
edu_mix = mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2)

at = Dict(
    :education => edu_mix,
    :age => [25, 50, 65],
    :income => :mean
)

result = profile_margins(model, data; at=at, type=:predictions)
```

#### Key Features

- **Sum-to-1 validation**: Mixture weights must sum to 1.0 (throws `ArgumentError` if not)
- **Level validation**: All mixture levels must exist in the original categorical variable
- **Weighted contrasts**: Uses true weighted combination of contrast matrix rows, not approximations
- **Integration**: Works seamlessly with existing numeric profile specifications

#### Multiple Categorical Mixtures

Combine multiple categorical mixtures in the same scenario:

```julia
# Population composition for multiple categorical variables
scenarios = Dict(
    :education => mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2),
    :region => mix("urban" => 0.65, "rural" => 0.35),
    :employed => mix(true => 0.75, false => 0.25),  # Boolean mixture
    :age => 40,  # Fixed continuous value
    :income => 50000
)

result = profile_margins(model, data; at=scenarios, type=:predictions)
```

#### Mathematical Foundation

Categorical mixtures compute **true weighted combinations** of contrast matrix rows rather than using single discrete levels:

```julia
# Standard categorical specification (single level)
at = Dict(:education => "college")  # Uses one contrast matrix row

# Categorical mixture (weighted combination)  
at = Dict(:education => mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2))
# Computes: 0.3 * contrast_matrix[hs_row, :] + 0.5 * contrast_matrix[col_row, :] + 0.2 * contrast_matrix[grad_row, :]
```

This enables population-level analysis where you want to understand effects or predictions for a **mixed population** with specific demographic composition, rather than just individual-level scenarios.

#### Boolean Mixtures

Boolean variables support mixture specification using either boolean values or strings:

```julia
# Both equivalent for boolean variables
employment_mix1 = mix(true => 0.7, false => 0.3)          # 70% employed
employment_mix2 = mix("true" => 0.7, "false" => 0.3)      # Same result

# Usage in profiles
at = Dict(:employed => employment_mix1, :age => :mean)
```

#### Use Cases

1. **Population Policy Analysis**: Model effects of policies on populations with specific demographic compositions
2. **Counterfactual Analysis**: "What if our customer base was 40% college-educated instead of 60%?"
3. **Market Research**: Predictions for target demographics with known composition
4. **Clinical Trials**: Effects in populations with specific comorbidity patterns

#### Error Handling

```julia
# These will throw ArgumentError
mix("high_school" => 0.3, "college" => 0.8)    # Weights don't sum to 1.0
mix("high_school" => 0.3, "phd" => 0.7)        # "phd" not in data levels
mix()                                           # Empty mixture
```

#### Validation and Accuracy

Categorical mixtures are mathematically validated to ensure exact weighted combinations:

```julia
# Manual verification
hs_pred = profile_margins(model, data; at=Dict(:education => "high_school"), type=:predictions)
col_pred = profile_margins(model, data; at=Dict(:education => "college"), type=:predictions)  
grad_pred = profile_margins(model, data; at=Dict(:education => "graduate"), type=:predictions)

expected = 0.3 * hs_pred + 0.5 * col_pred + 0.2 * grad_pred

# Mixture computation
edu_mix = mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2)
actual = profile_margins(model, data; at=Dict(:education => edu_mix), type=:predictions)

# Results are identical within floating-point precision
@assert abs(expected - actual) < 1e-12
```

### Mixed Specifications

Combine different methods within a single specification:

```julia
at = Dict(
    :x => [-2, 0, 2],               # Explicit values
    :z => :mean,                    # Summary statistic
    :w => "10(5)25",                # Range notation
    :treatment => [0, 1]            # Categorical levels
)
```

## Implementation Architecture

The reference grid system is architecturally unified:

1. **Single Source**: All specifications processed by `_build_profiles()` in `profiles.jl`
2. **Consistent Output**: Returns `Vector{Dict{Symbol,Any}}` of scenarios
3. **FormulaCompiler Integration**: Scenarios passed to `create_scenario()` for evaluation
4. **Type Safety**: Automatic handling of variable types and conversions

## Usage Examples

### Basic Profile Analysis

```julia
# Evaluate effects at low/medium/high values
scenarios = Dict(:education => [8, 12, 16], :experience => :mean)
results = profile_margins(
    model, data; at = scenarios, type = :effects, vars = [:education])
```

### Treatment Effect Analysis

```julia
# Compare treatment effects across demographic groups
scenarios = [
    Dict(:treatment => [0, 1], :age => :p25, :urban => 1.0),    # Young urban
    Dict(:treatment => [0, 1], :age => :p75, :urban => 0.0)     # Older rural
]
results = profile_margins(
    model, data; at = scenarios, type = :effects, vars = [:treatment]
)
```

### Prediction Profiles

```julia
# Generate prediction surface
grid = Dict(
    :x1 => "-2(0.5)2",      # -2.0, -1.5, -1.0, ..., 1.5, 2.0
    :x2 => [-1, 0, 1]       # Three levels
)
predictions = profile_margins(
    model, data; at = grid, type = :predictions, scale = :response
)
```

## Table-Based Specification

For maximum control and complex scenarios that are difficult to express through Dict syntax, use the table-based approach by passing a DataFrame directly as the third argument:

### Basic Table-Based Usage

```julia
# Define exact covariate combinations
reference_grid = DataFrame(
    x1 = [0.0, 1.0, 0.0, 1.0, 2.0],
    x2 = [0.5, 0.5, -0.5, -0.5, 0.0],
    treated = [true, false, true, false, false]
)

# Compute effects at these exact combinations
results = profile_margins(model, data, reference_grid; type=:effects, vars=[:x1])

# Compute predictions at these exact combinations
predictions = profile_margins(model, data, reference_grid; type=:predictions, scale=:response)
```

### Advanced Table-Based Examples

```julia
# Complex interaction analysis
interaction_grid = DataFrame(
    age = [25, 25, 65, 65],
    income = [30000, 80000, 30000, 80000],
    treatment = [false, false, true, true],
    region = ["urban", "rural", "urban", "rural"]
)

# Non-uniform spacing for dose-response analysis
dose_response_grid = DataFrame(
    dose = [0.0, 0.1, 0.5, 2.0, 10.0],  # Log-spaced doses
    age = [40.0, 40.0, 40.0, 40.0, 40.0],  # Fixed age
    weight = [70.0, 70.0, 70.0, 70.0, 70.0]  # Fixed weight
)

# Factorial design with unbalanced scenarios
factorial_grid = DataFrame(
    factor_a = [1, 1, 2, 2, 3],
    factor_b = [1, 2, 1, 2, 1],  # Unbalanced - no (3,2) combination
    covariate = [0.0, 0.5, -0.5, 1.0, 0.25]
)
```

### Advantages of Table-Based Specification

1. **Exact Control**: Specify any combination of covariate values, including irregular grids
2. **Complex Designs**: Handle factorial, response surface, or custom experimental designs
3. **Unbalanced Scenarios**: Include only the specific combinations you need
4. **External Integration**: Import reference grids from spreadsheets, databases, or other analyses
5. **Reproducible Research**: Store and version control exact evaluation points

### Equivalence Between Approaches

The two approaches are equivalent when the table represents a full Cartesian product:

```julia
# Dict approach (Cartesian product)
dict_spec = Dict(:x1 => [0.0, 1.0], :x2 => [0.5, -0.5])

# Equivalent table approach
table_spec = DataFrame(
    x1 = [0.0, 0.0, 1.0, 1.0],
    x2 = [0.5, -0.5, 0.5, -0.5]
)

# These produce identical results
result1 = profile_margins(model, data; at=dict_spec, type=:effects, vars=[:x1])
result2 = profile_margins(model, data, table_spec; type=:effects, vars=[:x1])
```

### Current Limitations

- **Continuous variables only**: Table-based approach currently supports only continuous marginal effects
- **No categorical contrasts**: Categorical variable effects not yet implemented for table-based specification
- **No grouping**: Same limitation as Dict-based approach for `over`/`by` parameters

## Choosing Between Approaches

**Use Dict-based specification when:**
- Creating regular grids (Cartesian products)
- Using summary statistics (means, quantiles)
- Working with simple factorial designs
- Need categorical variable contrasts

**Use table-based specification when:**
- Requiring exact control over covariate combinations
- Working with irregular or unbalanced designs
- Importing scenarios from external sources
- Analyzing complex interaction patterns
- Need to exclude specific covariate combinations

This unified reference grid system provides the flexibility to specify complex evaluation scenarios while maintaining consistent, predictable behavior across all profile-based computations.
