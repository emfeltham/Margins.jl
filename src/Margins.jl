"""
    Margins.jl

A Julia package for computing marginal effects with Stata-like functionality.
Built on jl for high-performance zero-allocation evaluation.

# 2×2 Framework

**Population vs Profile**:
- `population_margins()`: Effects/predictions averaged over observed data (AME/AAP)
- `profile_margins()`: Effects/predictions at specific evaluation points (MEM/APM)

**Effects vs Predictions**:
- `type=:effects`: Marginal effects (derivatives for continuous, contrasts for categorical)
- `type=:predictions`: Adjusted predictions (fitted values)

# Core API

- `population_margins(model, data; type, vars, scale, backend, scenarios, groups, measure, vcov)`: Population-level analysis
- `profile_margins(model, data, reference_grid; type, vars, scale, backend, measure, vcov)`: Profile-level analysis  
- `EffectsResult`, `PredictionsResult`: Type-specific result containers with Tables.jl interface

# Examples

```julia
using Margins, DataFrames, GLM

# Fit model
model = lm(@formula(y ~ x1 + x2), data)

# Population average marginal effects (AME)
result = population_margins(model, data; type=:effects)

# Marginal effects at sample means (MEM)
result = profile_margins(model, data, means_grid(data); type=:effects)

# Convert to DataFrame
df = DataFrame(result)
```
"""
module Margins

# Dependencies
using Tables, DataFrames, StatsModels, GLM
using FormulaCompiler
# Only FormulaCompiler's exported (public) API is used
using FormulaCompiler:
    CategoricalMixture, MixtureWithLevels, validate_mixture_against_data,
    mixture_to_scenario_value,
    AbstractDerivativeEvaluator, derivativeevaluator,
    derivative_variables, derivative_backend,
    CategoricalCounterfactualVector,  # Needed by continuous_variables
    fixed_effects_form,
    ContrastEvaluator, contrast_modelrow!, contrast_gradient!, contrast_endpoints,
    validate_mixture_weights, validate_mixture_levels,
    categorical_levels, categorical_variables, baseline_level,
    get_or_compile_formula, with_private_scratch, extract_variable_names,
    mix,
    _dmu_deta, _d2mu_deta2,  # Link function derivatives (marginal_effects_mu! test oracle)
    link_derivatives         # (μ, dμ/dη, d²μ/dη²) for the prediction kernel
using LinearAlgebra: dot, mul!, BLAS
using LinearAlgebra
using Statistics: mean, median
using CategoricalArrays
using Printf: @sprintf
using Distributions: Normal, TDist, cdf, ccdf, quantile
using Dates: now
import StatsBase  # weight types of a fitted model (_weights_from_storage)
import CovarianceMatrices  # robust/clustered `vcov` estimators (`_evaluate_vcov`)

# Version info
const VERSION = pkgversion(@__MODULE__)  # read from Project.toml, so it cannot drift

# Core API - Clean 2×2 framework
export population_margins, profile_margins, MarginsResult, EffectsResult, PredictionsResult, ContrastResult
export Inference, ResultContext  # the types of `result.inference` and `result.context`

# Display configuration
export set_display_digits, get_display_digits, set_profile_digits, get_profile_digits

# Statistical utilities
export contrast

# Categorical mixture utilities
# Re-export FormulaCompiler's native mixture functionality
export CategoricalMixture, mix
export create_balanced_mixture, create_mixture_column, expand_mixture_grid
export validate_mixture_weights, validate_mixture_levels  # FormulaCompiler's, re-exported

# Reference grid builders (NEW - AsBalanced support)
export means_grid, balanced_grid, cartesian_grid, quantile_grid, hierarchical_grid, complete_reference_grid

# Include all submodules in dependency order
include("types.jl")
include("metadata.jl")   # the documented contract of `result.metadata` and its accessors

# Statistical inference modules
include("inference/derivative_utilities.jl")             # _matrix_multiply_eta!, continuous_variables
include("inference/marginal_effects.jl")                 # marginal_effects_eta!/_mu! (row-level; internal test oracles)
include("inference/delta_method.jl")                     # delta_method_se
include("inference/mixture_utilities.jl")                # create_mixture_column, expand_mixture_grid, create_balanced_mixture

include("core/data_conversion.jl")  # Data type conversion utilities
include("core/model_interface.jl")  # model-class interface: _link, _coef, _inference_df, _offset, ...
include("core/validation.jl")
include("core/model_weights.jl")  # weights=:model default: _model_weights, _resolve_weights
include("core/margins_validation.jl")
include("core/request.jl")  # MarginsRequest, _normalize_request: every entry point's options, validated once
include("core/typical_values.jl")
include("computation/scenarios.jl")  # ContrastPair and scenario utilities
include("computation/predictions.jl")
include("computation/continuous_accumulator.jl")  # Measure, MEWorkspace, accumulate_continuous!
include("computation/statistics.jl")
include("computation/result_formatting.jl")  # Result DataFrame builders
include("engine/core.jl")  # Defines MarginsEngine
include("core/engine_validation.jl")  # Uses MarginsEngine, must come after engine/core.jl
include("core/variable_detection.jl")  # Uses MarginsEngine, must come after engine/core.jl
# core/buffer_management.jl DELETED (2026-08): ComputationBuffers/ProfileBuffers and their
# accessors were never adopted; all computation uses the engine's own pre-allocated buffers.
include("kernels/categorical.jl")  # Phase 4: Categorical kernel using FC primitives
include("population/core.jl")
include("population/contexts.jl")
include("population/continuous_effects.jl")
include("population/categorical_effects.jl")
include("population/effects.jl")
include("profile/core.jl")
include("profile/reference_grids.jl")

# Second differences module
include("second_differences/contrasts.jl")
include("second_differences/at_point.jl")
include("second_differences/utilities.jl")
include("second_differences/model_interface.jl")

export second_differences

# Second-difference functions of Margins 2.x that were removed in 3.0, with their replacements.
# Calling one gives an UndefVarError; this hint says what to use instead.
const REMOVED_SECOND_DIFFERENCE_FUNCTIONS = Dict(
    :second_differences_at => "second_differences(model, data, variables, modifier; at=…) " *
        "(the vcov argument is now the keyword `vcov=`)",
    :second_differences_pairwise => "second_differences(result, variables, modifier; contrast=…)",
    :second_differences_all_contrasts => "second_differences(result, variables, modifier)",
    :second_differences_table => "second_differences(result, variables, modifier)",
    :second_difference => "second_differences(result, variables, modifier; include_gradients=true) " *
        "(it returns a DataFrame row, not a NamedTuple)",
)

# The engine cache and its controls, removed in 3.0 (every call builds its own engine; only
# FormulaCompiler's compiled-formula cache remains), with what to do instead.
const REMOVED_CACHE_NAMES = Dict(
    :get_cache_stats => "there is no engine cache to inspect",
    :set_engine_cache_capacity! => "there is no engine cache to bound",
    :engine_cache_capacity => "there is no engine cache to bound",
    :get_or_build_engine => "use `Margins.build_engine` (same arguments without the second, " *
        "`HasDerivatives`/`NoDerivatives`)",
    :ENGINE_CACHE => "there is no engine cache; `Margins.clear_engine_cache!()` still clears " *
        "FormulaCompiler's compiled-formula cache",
    :VALIDATED_MODELS_CACHE => "validation runs on every call, so there is nothing to clear",
)

function __init__()
    if isdefined(Base, :Experimental) && isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(UndefVarError) do io, exc
            replacement = get(REMOVED_SECOND_DIFFERENCE_FUNCTIONS, exc.var, nothing)
            if !isnothing(replacement)
                print(io, "\n`", exc.var, "` was removed in Margins 3.0. Use ", replacement,
                      ". See the CHANGELOG and the Second Differences documentation for the migration.")
                return
            end
            replacement = get(REMOVED_CACHE_NAMES, exc.var, nothing)
            isnothing(replacement) && return
            print(io, "\n`Margins.", exc.var, "` was removed in Margins 3.0 with the engine cache: ",
                  replacement, ". See the CHANGELOG.")
        end
    end
end

end # module
