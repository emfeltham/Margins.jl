# Margins.jl

module Margins

using Statistics
using LinearAlgebra
using Tables, DataFrames
using StatsModels, StatsBase, GLM, CategoricalArrays
using FormulaCompiler
using CovarianceMatrices
using Distributions

# Public API exports - Clean two-function design
export population_margins, profile_margins, MarginsResult
# Categorical mixture support
export mix, CategoricalMixture

# Core Infrastructure
include("core/utilities.jl")          # General utility functions
include("core/grouping.jl")           # Grouping and stratification utilities
include("core/results.jl")            # Result types and display
include("core/profiles.jl")           # Profile grid building and processing
include("core/link.jl")               # Link function utilities

# Computation Engine
include("computation/engine.jl")       # FormulaCompiler integration
include("computation/continuous.jl")   # Continuous marginal effects (AME/MEM/MER)
include("computation/categorical.jl")  # Categorical contrasts and discrete changes
include("computation/predictions.jl")  # Adjusted predictions (APE/APM/APR)

# Advanced Features
include("features/categorical_mixtures.jl")  # Categorical mixture support
include("features/averaging.jl")             # Proper delta method averaging for profiles

# API Layer
include("api/common.jl")              # Shared API utilities and helpers
include("api/population.jl")          # Population margins API (AME/APE)
include("api/profile.jl")             # Profile margins API (MEM/MER/APM/APR)

end # module
