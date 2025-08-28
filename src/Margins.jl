# Margins.jl

module Margins

using Statistics
using LinearAlgebra
using Tables, DataFrames
using StatsModels, GLM, CategoricalArrays
using FormulaCompiler
using CovarianceMatrices

# Public API exports - Clean two-function design
export population_margins, profile_margins, MarginsResult

# Internal modules
include("results.jl")     # Result type and builders
include("link.jl")        # Link utilities
include("engine_fc.jl")   # Build compiled + evaluator + model info
include("profiles.jl")    # at = :means and profile grids
include("predictions.jl") # APE/APM/APR computations
include("compute_continuous.jl")  # AME/MEM/MER continuous effects
include("compute_categorical.jl") # Categorical contrasts (stubs initially)
include("api.jl")         # User-facing API and dispatch

end # module
