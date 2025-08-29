# Reference Grid Specification

Reference grids define the covariate scenarios where marginal effects and predictions are evaluated in profile-based analysis. Margins.jl provides a flexible, unified system for specifying reference grids through the `at` parameter in `profile_margins()`.

## Overview

When computing profile margins, you specify scenarios using the `at` parameter:

```julia
profile_margins(model, data; at=scenarios, type=:effects, ...)
```

The reference grid system supports multiple specification methods, all processed through a unified architecture that flows through `_build_profiles()` → `FormulaCompiler.create_scenario()`.

## Specification Methods

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

This unified reference grid system provides the flexibility to specify complex evaluation scenarios while maintaining consistent, predictable behavior across all profile-based computations.
