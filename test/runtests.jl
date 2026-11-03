# runtests.jl
# julia --project="." -e "import Pkg; Pkg.test()" > test/runtests.txt 2>&1

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using StatsModels
using Statistics
using Margins
using MixedModels
using FixedEffectModels
using RDatasets
using Distributions

# Run-environment header and a per-testset clock line (wall, CPU, time the machine was asleep)
include("support/suite_clock.jl")
print_suite_environment()

# Start from an empty FormulaCompiler compiled-formula cache (Margins keeps no cache of its own)
Margins.clear_engine_cache!()

# Load testing utilities centrally to prevent method definition warnings
include("statistical_validation/testing_utilities.jl")
include("statistical_validation/analytical_se_validation.jl")

suite_phase("Core Functionality")
@testset "Core Functionality" begin
    include("core/test_output_schema.jl")
    include("core/test_contrast_dataframe.jl")
    include("core/test_typed_results.jl")      # result.inference / result.context (B9, B10)
    include("core/test_errors.jl")
    include("core/test_automatic_variable_detection.jl")
    include("core/test_mixedmodels.jl")
    include("core/test_weights.jl")
    include("core/test_model_weights.jl")
    include("core/test_request_layer.jl")
    include("core/test_model_interface.jl")   # model-class interface: df rule, offsets, FixedEffectModels
    include("core/test_derived_columns.jl")   # SetTo/Shift counterfactuals and the derived-column warning
    include("core/test_ordering.jl")          # deterministic grid, groups and scenarios order (step 9)
    include("core/test_rank_deficient.jl")    # dropped (collinear) coefficients fixed at 0 (ISSUES #10)
    include("core/test_profile_fill_weights.jl") # profile fill-ins weighted by the model's estimation weights
    include("core/test_public_api.jl")        # API freeze: exported names, keywords, fields, result columns
    include("statistical_validation/robust_se_validation.jl")
end

suite_phase("Advanced Features")
@testset "Advanced Features" begin
    include("features/test_elasticities.jl")
    include("features/test_categorical_mixtures.jl")
    include("features/test_bool_profiles.jl")
    include("features/test_table_profiles.jl")
    include("features/test_prediction_intervals.jl")
    include("features/test_hierarchical_grids.jl")
    include("features/test_string_categorical_conversion.jl")
    include("features/test_balanced_grid.jl")
    include("features/test_typical_values.jl")   # weighted typical values, grid builders' weights= (step 9)
    include("features/test_fixed_effect_models.jl")
    include("regression/test_bool_mixture_skip.jl")
    include("regression/test_unused_categorical_levels.jl")
    include("regression/test_group_binning.jl")
    include("regression/test_profile_multi_categorical.jl")
    include("regression/test_zero_weight_contexts.jl")
    include("regression/test_missing_eltype_columns.jl")
    include("regression/test_mixedmodels_link_without_main.jl")
    include("regression/test_pvalues_and_version.jl")
    include("regression/test_categorical_pool_alignment.jl")
    include("regression/test_negbin_link_derivative.jl")
    include("regression/test_integer_coded_categorical.jl")
    include("regression/test_profile_measure_quotient_rule.jl")
    include("regression/test_int_column_conversion.jl")
    include("regression/test_stale_results.jl")
    include("regression/test_hierarchical_partial_typical.jl")
    include("regression/test_mixture_cache.jl")
    include("second_differences/sd_result_level.jl")      # second_differences(result, …), engines, row selection, public API
    include("second_differences/sd_model_level.jl")       # second_differences(model, data, …): forms, at=, levels, profile, weights
    include("second_differences/sd_inference.jl")         # vcov forms, df rule, CIs/alpha, p_adjust, gradients, structural zeros
    include("second_differences/sd_observed_groups.jl")   # at=:observed, groups=, backend=:ad vs :fd
    include("second_differences/sd_categorical_focal.jl") # categorical / Bool focal variables, contrast=, interaction-only variables
    include("test_vcov_matrix.jl")
    include("test_result_metadata_contract.jl")
    include("test_unified_api.jl")                  # mixed variable types: joint call equals separate calls
end

suite_phase("Performance and Allocations")
@testset "Performance and Allocations" begin
    include("performance/test_allocation_scaling.jl")
    include("test_categorical_kernel_zero_alloc.jl")
    include("test_per_row_allocations.jl")
    include("performance/test_ame_fd_loop_zero_alloc.jl")
    include("performance/test_population_aap_loop_zero_alloc.jl")
end

suite_phase("Statistical Correctness")
@testset "Statistical Correctness" begin
    include("statistical_validation/backend_consistency.jl")
    include("statistical_validation/statistical_validation.jl")
    include("statistical_validation/incompatible_formula_se_validation.jl")
    include("statistical_validation/function_composition_tests.jl")
end

suite_phase("Computational Primitives")
@testset "Computational Primitives" begin
    include("primitives/test_derivatives.jl")
    include("primitives/test_links.jl")
    include("primitives/test_prediction_kernel.jl")
    include("helpers/test_mixture_utilities.jl")
    include("primitives/test_formulacompiler_public_api.jl")  # no FC internals; baseline from contrasts
end

suite_phase("Validation Tests")
@testset "Validation Tests" begin
    include("validation/test_contrast_invariance.jl")
    include("validation/test_ci_and_n_logic.jl")
    include("validation/test_population_scenarios_groups.jl")
    include("validation/test_gradient_correctness.jl")
    include("validation/test_link_scale.jl")
    include("validation/test_measure_contexts.jl")
    include("validation/test_elasticity_unit_definitions.jl")
    include("validation/test_second_differences_r_oracle.jl")
    include("validation/test_oracle_3_0.jl")                      # R: at=:observed SD, model weights, offsets, fixest
    include("validation/test_second_differences_mixed_fem.jl")    # SD on LMM, GLMM, FixedEffectModels
    include("validation/test_engine_data_binding.jl")
    include("validation/test_no_engine_cache.jl")
    include("validation/test_concurrent_calls.jl")
    include("validation/test_profile_categorical_contrasts.jl")
    # Step 0 of the R6/R7 refactor: independent oracle and golden-snapshot equivalence
    include("validation/test_feature_matrix_oracle.jl")
    include("validation/test_golden_equivalence.jl")
end
suite_phase(nothing)
