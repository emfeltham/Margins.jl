# runtests.jl
# julia --project="." -e "import Pkg; Pkg.test()" > test/test.txt 2>&1

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using StatsModels
using Statistics
using Margins
using MixedModels
using RDatasets
using Distributions

# Load testing utilities centrally to prevent method definition warnings
include("statistical_validation/testing_utilities.jl")
include("statistical_validation/bootstrap_se_validation.jl") 
include("statistical_validation/analytical_se_validation.jl")

@testset "Core Functionality" begin
    include("core/test_glm_basic.jl")
    include("core/test_profiles.jl")
    include("core/test_grouping.jl")
    include("core/test_contrasts.jl")
    include("core/test_vcov.jl")
    include("core/test_errors.jl")
    include("core/test_automatic_variable_detection.jl")
    include("core/test_mixedmodels.jl")
    include("core/test_weights.jl")
end

@testset "Advanced Features" begin
    include("features/test_elasticities.jl")
    include("features/test_categorical_mixtures.jl") 
    include("features/test_bool_profiles.jl")
    include("features/test_table_profiles.jl")
    include("features/test_prediction_scales.jl")
end

@testset "Performance" begin
    include("performance/test_performance.jl")
    include("performance/test_zero_allocations.jl")
end

@testset "Statistical Correctness" begin
    include("statistical_validation/backend_consistency.jl")
    include("statistical_validation/statistical_validation.jl")
    include("statistical_validation/robust_se_validation.jl")
end
