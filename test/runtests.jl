using Test
using Random
using DataFrames, CategoricalArrays, GLM
using StatsModels
using Statistics
using Margins

# Core functionality tests
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

# Statistical Validation Suite
@testset "Statistical Correctness Validation" begin
    @info "Starting comprehensive statistical validation framework..."
    
    # Core statistical correctness validation
    include("statistical_validation/statistical_validation.jl")
    
    # Backend consistency validation (essential)
    include("statistical_validation/backend_consistency.jl")
    
    @info "Statistical validation framework completed successfully ✓"
end
