using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

include("test_glm_basic.jl")
include("test_profiles.jl")
include("test_grouping.jl")
include("test_contrasts.jl")
include("test_vcov.jl")
include("test_errors.jl")
try
    include("test_mixedmodels.jl")
catch e
    @warn "Skipping MixedModels tests: $e"
end
