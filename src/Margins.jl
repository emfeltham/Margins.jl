"""
    Margins.jl

A Julia package for computing marginal effects with Stata-like functionality.
Built on FormulaCompiler.jl for high-performance zero-allocation evaluation.

# 2×2 Framework

**Population vs Profile**:
- `population_margins()`: Effects/predictions averaged over observed data (AME/AAP)
- `profile_margins()`: Effects/predictions at specific evaluation points (MEM/APM)

**Effects vs Predictions**:
- `type=:effects`: Marginal effects (derivatives for continuous, contrasts for categorical)
- `type=:predictions`: Adjusted predictions (fitted values)

# Core API

- `population_margins(model, data; type, vars, scale, backend, scenarios, groups, measure, vcov)`: Population-level analysis
- `profile_margins(model, data, reference_grid; type, vars, scale, backend, measure, vcov)`: Profile-level analysis  
- `MarginsResult`: Result container with Tables.jl interface

# Examples

```julia
using Margins, DataFrames, GLM

# Fit model
model = lm(@formula(y ~ x1 + x2), data)

# Population average marginal effects (AME)
result = population_margins(model, data; type=:effects)

# Marginal effects at sample means (MEM)
result = profile_margins(model, data, means_grid(data); type=:effects)

# Convert to DataFrame
df = DataFrame(result)
```
"""
module Margins

# Dependencies
using Tables, DataFrames, StatsModels, GLM
using FormulaCompiler
using FormulaCompiler: CategoricalMixture, MixtureWithLevels, create_balanced_mixture, validate_mixture_against_data, mixture_to_scenario_value
import FormulaCompiler: mix, _get_baseline_level
using LinearAlgebra: dot
using Statistics: mean
using CategoricalArrays
using Printf: @sprintf
using Distributions: Normal, cdf, quantile

# Version info
const VERSION = v"2.0.0"

# Core API - Clean 2×2 framework  
export population_margins, profile_margins, MarginsResult

# Display configuration
export set_display_digits, get_display_digits, set_profile_digits, get_profile_digits

# Categorical mixture utilities
# Re-export FormulaCompiler's native mixture functionality
export CategoricalMixture, mix

# Reference grid builders (NEW - AsBalanced support)
export means_grid, balanced_grid, cartesian_grid, quantile_grid, hierarchical_grid, complete_reference_grid

# Include all submodules in dependency order
include("types.jl")
include("core/validation.jl")
include("core/margins_validation.jl")
include("computation/predictions.jl")
include("computation/statistics.jl")
include("engine/measures.jl")  # Measure transformation utilities
include("engine/core.jl")
include("engine/utilities.jl") 
include("engine/caching.jl")  # Unified caching system
include("population/core.jl")
include("population/contexts.jl")
include("population/effects.jl")
include("profile/core.jl")
include("profile/refgrids.jl")
include("profile/contrasts.jl")

end # module