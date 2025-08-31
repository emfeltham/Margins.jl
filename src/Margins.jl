module Margins

# Dependencies
using Tables, DataFrames, StatsModels, GLM
using FormulaCompiler
using LinearAlgebra: dot
using Statistics: mean
using CategoricalArrays

# Version info
const VERSION = v"2.0.0"

# Main exports - Clean 2×2 framework
export population_margins, profile_margins, MarginsResult

# Advanced exports
export mix  # Categorical mixture constructor
export population_margins!, profile_margins!  # In-place versions (future)

# Include all submodules in dependency order
include("types.jl")
include("features/categorical_mixtures.jl")  # Include before utilities that use it
include("engine/core.jl")
include("engine/utilities.jl")
include("engine/caching.jl")
include("population/core.jl")
include("population/contexts.jl")
include("population/effects.jl")
include("profile/core.jl")
include("profile/refgrids.jl")
include("profile/contrasts.jl")

end # module