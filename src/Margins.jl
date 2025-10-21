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
using FormulaCompiler:
    CategoricalMixture, MixtureWithLevels, validate_mixture_against_data,
    mixture_to_scenario_value,
    AbstractDerivativeEvaluator, derivativeevaluator, ADEvaluator, FDEvaluator,
    LoadOp, ContrastOp,
    CategoricalCounterfactualVector,  # Needed by continuous_variables
    fixed_effects_form,
    ContrastEvaluator, contrast_modelrow!,
    get_or_compile_formula,
    mix,
    _dmu_deta, _d2mu_deta2  # Link function derivatives for marginal_effects_mu!
# Import (not using) to allow method extension
import FormulaCompiler: _get_baseline_level
using LinearAlgebra: dot, mul!, BLAS
using LinearAlgebra
using Statistics: mean, median
using CategoricalArrays
using Printf: @sprintf
using Distributions: Normal, cdf, quantile
using Dates: now

# Version info
const VERSION = v"2.0.0"

# Core API - Clean 2×2 framework
export population_margins, profile_margins, MarginsResult, EffectsResult, PredictionsResult, ContrastResult

# Display configuration
export set_display_digits, get_display_digits, set_profile_digits, get_profile_digits

# Statistical utilities
export contrast

# Categorical mixture utilities
# Re-export FormulaCompiler's native mixture functionality
export CategoricalMixture, mix
export create_balanced_mixture, create_mixture_column, expand_mixture_grid
export validate_mixture_weights, validate_mixture_levels

# Reference grid builders (NEW - AsBalanced support)
export means_grid, balanced_grid, cartesian_grid, quantile_grid, hierarchical_grid, complete_reference_grid

# Include all submodules in dependency order
include("types.jl")

# Statistical inference modules
include("inference/derivative_utilities.jl")             # _matrix_multiply_eta!, continuous_variables
include("inference/marginal_effects.jl")                 # marginal_effects_eta!, marginal_effects_mu! (wrapper functions with dispatch)
include("inference/marginal_effects_automatic_diff.jl")  # marginal_effects_eta!, marginal_effects_mu! (AD backend)
include("inference/marginal_effects_finite_diff.jl")     # marginal_effects_eta!, marginal_effects_mu! (FD backend)
include("inference/delta_method.jl")                     # delta_method_se
include("inference/contrast_gradient.jl")                # contrast_gradient!, contrast_gradient (migrated from FormulaCompiler)
include("inference/mixture_utilities.jl")                # create_mixture_column, expand_mixture_grid, create_balanced_mixture

include("core/data_conversion.jl")  # Data type conversion utilities
include("core/validation.jl")
include("core/margins_validation.jl")
include("core/typical_values.jl")
include("computation/scenarios.jl")  # ContrastPair and scenario utilities
include("computation/marginal_effects.jl")  # Marginal effects computation kernels
include("computation/predictions.jl")
include("computation/statistics.jl")
include("computation/result_formatting.jl")  # Result DataFrame builders
include("population/effects_buffers.jl")  # Must come before engine/core.jl
include("engine/measures.jl")  # Measure transformation utilities
include("engine/core.jl")  # Defines MarginsEngine
include("core/engine_validation.jl")  # Uses MarginsEngine, must come after engine/core.jl
include("core/variable_detection.jl")  # Uses MarginsEngine, must come after engine/core.jl
include("core/buffer_management.jl")  # Buffer extraction utilities
include("engine/caching.jl")  # Unified caching system
include("kernels/categorical.jl")  # Phase 4: Categorical kernel using FC primitives
include("population/core.jl")
include("population/contexts.jl")
include("population/continuous_effects.jl")
include("population/categorical_effects.jl")
include("population/effects.jl")
include("profile/core.jl")
include("profile/reference_grids.jl")
include("profile/categorical_effects.jl")  # Phase 4 migration: ContrastEvaluator-based categorical effects
# profile/contrasts.jl DELETED (2025-10-01): Obsolete DataScenario code replaced by ContrastEvaluator in profile/core.jl

# Second differences module
include("second_differences/contrasts.jl")
include("second_differences/at_point.jl")
include("second_differences/utilities.jl")

export second_differences, second_difference, second_differences_all_contrasts,
       second_differences_table, second_differences_pairwise,
       second_differences_at

end # module
