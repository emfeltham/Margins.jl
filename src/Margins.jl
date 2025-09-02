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

- `population_margins(model, data; kwargs...)`: Population-level analysis
- `profile_margins(model, data; at=:means, kwargs...)`: Profile-level analysis  
- `MarginsResult`: Result container with Tables.jl interface

# Examples

```julia
using Margins, DataFrames, GLM

# Fit model
model = lm(@formula(y ~ x1 + x2), data)

# Population average marginal effects (AME)
result = population_margins(model, data; type=:effects)

# Marginal effects at sample means (MEM)
result = profile_margins(model, data; type=:effects, at=:means)

# Convert to DataFrame
df = DataFrame(result)
```
"""
module Margins

# Dependencies
using Tables, DataFrames, StatsModels, GLM
using FormulaCompiler
using LinearAlgebra: dot
using Statistics: mean
using CategoricalArrays

# Version info
const VERSION = v"2.0.0"

# Core API - Clean 2×2 framework  
export population_margins, profile_margins, MarginsResult

# Include all submodules in dependency order
include("types.jl")
include("core/validation.jl")
include("computation/predictions.jl")
include("features/categorical_mixtures.jl")
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