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

# Start from an empty FormulaCompiler compiled-formula cache (Margins keeps no cache of its own)
Margins.clear_engine_cache!()

# Load testing utilities centrally to prevent method definition warnings
include("statistical_validation/testing_utilities.jl")
include("statistical_validation/bootstrap_se_validation.jl")
include("statistical_validation/analytical_se_validation.jl")

@testset "Core Functionality" begin
    include("core/test_grouping.jl")
    include("core/test_column_naming.jl")
    include("core/test_output_schema.jl")
    include("core/test_contrasts.jl")
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
    include("statistical_validation/robust_se_validation.jl")
end

@testset "Advanced Features" begin
    include("features/test_elasticities.jl")
    include("features/test_categorical_mixtures.jl")
    include("features/test_bool_profiles.jl")
    include("features/test_table_profiles.jl")
    include("features/test_prediction_scales.jl")
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
    include("test_second_differences.jl")
    include("test_second_differences_at.jl")
    include("test_second_differences_regressions.jl")
    include("test_second_differences_model_interface.jl")
    include("test_second_differences_api.jl")
    include("test_second_differences_structural.jl")
    include("test_second_differences_review.jl")
    include("test_vcov_matrix.jl")
    include("test_result_metadata_contract.jl")
    include("test_second_differences_observed.jl")
    include("test_second_differences_groups.jl")    # groups=, backend=, interaction-only variables (step 8)
    include("test_second_diff_bool_fix.jl")         # categorical Bool focal variable, contrast found automatically
    include("test_unified_api.jl")                  # mixed variable types: joint call equals separate calls
    include("test_profile_pairwise.jl")
end

@testset "Performance and Allocations" begin
    include("performance/test_allocation_scaling.jl")
    include("performance/test_performance.jl")
    include("performance/profiling/test_contrast_gradient_allocations.jl")
    include("test_categorical_batch_zero_alloc.jl")
    include("test_categorical_kernel_zero_alloc.jl")
    include("test_per_row_allocations.jl")
    include("performance/test_ame_fd_loop_zero_alloc.jl")
    include("performance/test_population_aap_loop_zero_alloc.jl")
end

@testset "Statistical Correctness" begin
    include("statistical_validation/backend_consistency.jl")
    include("statistical_validation/statistical_validation.jl")
    include("statistical_validation/incompatible_formula_se_validation.jl")
    include("statistical_validation/elasticity_se_validation.jl")
    include("statistical_validation/bootstrap_validation_tests.jl")
    include("statistical_validation/robust_se_tests.jl")
    include("statistical_validation/specialized_se_tests.jl")
    include("statistical_validation/analytical_elasticity_se_validation.jl")
    include("statistical_validation/population_context_bootstrap_validation.jl")
    include("statistical_validation/backend_reliability_tests.jl")
    include("statistical_validation/function_composition_tests.jl")
end

@testset "Computational Primitives" begin
    include("primitives/test_derivatives.jl")
    include("primitives/test_links.jl")
    include("primitives/test_prediction_kernel.jl")
    include("primitives/test_derivatives_log_profile_regression.jl")
    include("primitives/test_marginal_effects_allocations.jl")
    include("helpers/test_mixture_utilities.jl")
    include("primitives/test_formulacompiler_public_api.jl")  # no FC internals; baseline from contrasts
end

@testset "Validation Tests" begin
    include("validation/test_contrast_invariance.jl")
    include("validation/test_manual_counterfactual_validation.jl")
    include("validation/test_ci_and_n_logic.jl")
    include("validation/test_population_scenarios_groups.jl")
    include("validation/test_gradient_correctness.jl")
    include("validation/test_link_scale.jl")
    include("validation/test_measure_contexts.jl")
    include("validation/test_elasticity_unit_definitions.jl")
    include("validation/test_second_differences_r_oracle.jl")
    include("validation/test_engine_data_binding.jl")
    include("validation/test_no_engine_cache.jl")
    include("validation/test_concurrent_calls.jl")
    include("validation/test_profile_categorical_contrasts.jl")
    # Step 0 of the R6/R7 refactor: independent oracle and golden-snapshot equivalence
    include("validation/test_feature_matrix_oracle.jl")
    include("validation/test_golden_equivalence.jl")
end
