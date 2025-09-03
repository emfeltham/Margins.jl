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
    # Essential quick statistical validation (always run)
    include("statistical_validation/backend_consistency.jl")
    
    # Comprehensive statistical validation (9 tiers, ~60 seconds)
    # Run comprehensive tests if explicitly requested via environment variable
    if get(ENV, "MARGINS_COMPREHENSIVE_TESTS", "false") == "true"
        @info "Running comprehensive statistical validation framework (9 tiers)..."
        @info "This includes: analytical SE validation, bootstrap validation, robust SE integration, and specialized edge cases"
        include("statistical_validation/statistical_validation.jl")
        @info "Comprehensive statistical validation completed successfully ✓"
    else
        @info "Quick statistical validation completed ✓"
        @info "Run with MARGINS_COMPREHENSIVE_TESTS=true for full 9-tier validation suite"
    end
end
