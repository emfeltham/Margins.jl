# runtests.jl

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using StatsModels
using Statistics
using Margins
using MixedModels
using RDatasets
using Distributions

# Core functionality tests
@testset "Core Functionality" begin
    include("test_glm_basic.jl")
    include("test_profiles.jl")
    include("test_grouping.jl")
    include("test_contrasts.jl")
    include("test_vcov.jl")
    include("test_errors.jl")
    include("test_automatic_variable_detection.jl")
    include("test_mixedmodels.jl")
end

# Advanced Features (Phase 1 additions)
@testset "Advanced Features" begin
    include("test_elasticities.jl")
    include("test_categorical_mixtures.jl") 
    include("test_bool_profiles.jl")
    include("test_table_profiles.jl")
    include("test_prediction_scales.jl")
end

# Performance (critical for regression prevention)
@testset "Performance" begin
    include("test_performance.jl")
    include("test_zero_allocations.jl")
end

# Statistical validation tests
@testset "Statistical Correctness" begin
    include("statistical_validation/backend_consistency.jl")
    include("statistical_validation/statistical_validation.jl")
end
