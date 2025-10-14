# runtests.jl
# julia --project="." -e "import Pkg; Pkg.test()" > test/runtests.txt 2>&1

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using StatsModels
using Statistics
using Margins
using MixedModels
using RDatasets
using Distributions

# Clear engine cache to ensure fresh engines (prevents stale cache issues)
Margins.clear_engine_cache!()

# Load testing utilities centrally to prevent method definition warnings
include("statistical_validation/testing_utilities.jl")
include("statistical_validation/bootstrap_se_validation.jl")
include("statistical_validation/analytical_se_validation.jl")

@testset "Core Functionality" begin
    include("core/test_glm_basic.jl")
    include("core/test_profiles.jl")
    include("core/test_grouping.jl")
    include("core/test_column_naming.jl")
    include("core/test_contrasts.jl")
    include("core/test_contrast_dataframe.jl")
    include("core/test_vcov.jl")
    include("core/test_errors.jl")
    include("core/test_automatic_variable_detection.jl")
    include("core/test_mixedmodels.jl")
    include("core/test_weights.jl")
    include("statistical_validation/robust_se_validation.jl")
end

@testset "Advanced Features" begin
    include("features/test_elasticities.jl")
    include("features/test_categorical_mixtures.jl")
    include("features/test_bool_profiles.jl")
    include("features/test_table_profiles.jl")
    include("features/test_prediction_scales.jl")
    include("features/test_hierarchical_grids.jl")
    include("regression/test_bool_mixture_skip.jl")
end

@testset "Performance" begin
    include("performance/test_allocation_scaling.jl")
    include("performance/test_performance.jl")
    include("test_categorical_batch_zero_alloc.jl")
    include("test_categorical_kernel_zero_alloc.jl")
    include("test_per_row_allocations.jl")
end

@testset "Statistical Correctness" begin
    include("statistical_validation/backend_consistency.jl")
    include("statistical_validation/statistical_validation.jl")
    include("statistical_validation/incompatible_formula_se_validation.jl")
end

@testset "Computational Primitives" begin
    include("primitives/test_derivatives.jl")
    include("primitives/test_links.jl")
    include("primitives/test_derivatives_log_profile_regression.jl")
    include("primitives/test_marginal_effects_allocations.jl")
    include("helpers/test_mixture_utilities.jl")
end

@testset "Inference Methods" begin
    include("inference/test_delta_method_glm.jl")
    include("inference/test_delta_method_mixedmodels.jl")
end

@testset "Profiling and Allocations" begin
    include("performance/profiling/test_contrast_gradient_allocations.jl")
end

@testset "Validation Tests" begin
    include("validation/test_contrast_invariance.jl")
    include("validation/test_manual_counterfactual_validation.jl")
    include("validation/test_ci_and_n_logic.jl")
    include("validation/test_population_scenarios_groups.jl")
    include("validation/test_gradient_correctness.jl")
end
