# runtests.jl

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using StatsModels
using Statistics
using Margins

# Core functionality tests
@testset "Core Functionality" begin
    include("test_glm_basic.jl")
    include("test_profiles.jl")
    include("test_grouping.jl")
    include("test_contrasts.jl")
    include("test_vcov.jl")
    include("test_errors.jl")
    include("test_automatic_variable_detection.jl")
    # include("test_mixedmodels.jl") where is this?
end

# Statistical Validation Suite
@testset "Statistical Correctness" begin
    @info "Starting comprehensive statistical validation framework..."
    # Core statistical correctness validation
    include("statistical_validation/statistical_validation.jl")
    # Backend consistency validation (essential)
    include("statistical_validation/backend_consistency.jl")
    @info "Statistical validation framework completed successfully ✓"
end
