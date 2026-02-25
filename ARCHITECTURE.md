# ARCHITECTURE.md: Margins.jl System Design

This document describes the implemented architecture of Margins.jl (v2.1.x).

## Overview

Margins.jl implements a **2×2 framework** for marginal effects computation, built on FormulaCompiler.jl's zero-allocation foundation. The architecture prioritizes statistical correctness, performance, and proper FormulaCompiler integration.

## The 2×2 Framework

```
                   │  Effects      │  Predictions
                   │  (∂y/∂x)      │  (fitted values)
───────────────────┼───────────────┼─────────────────
Population         │  AME          │  AAP
(across sample)    │  (avg marg    │  (avg adjusted
                   │   effect)     │   prediction)
───────────────────┼───────────────┼─────────────────
Profile            │  MEM          │  APM
(at specific pts)  │  (marg effect │  (adjusted pred
                   │   at mean)    │   at mean)
```

**Acronym Definitions:**
- **AME**: Average Marginal Effect (derivative for continuous, baseline contrast for categorical/bool)
- **AAP**: Average Adjusted Prediction (population average of fitted values)
- **MEM**: Marginal Effect at the Mean (derivative for continuous, row-specific baseline contrast for categorical)
- **APM**: Adjusted Prediction at the Mean (fitted value at representative point)

**User Interface:**
```julia
# Population analysis
population_margins(model, data; type=:effects)                                   # AME
population_margins(model, data; type=:predictions)                               # AAP
population_margins(model, data; type=:effects, groups=:region)                   # AME by subgroup
population_margins(model, data; type=:effects, scenarios=(x2=[1, 2, 3],))       # AME at counterfactuals

# Profile analysis
profile_margins(model, data, means_grid(data); type=:effects)                   # MEM
profile_margins(model, data, cartesian_grid(x1=[0,1,2]); type=:effects)         # MER
profile_margins(model, data, reference_grid; type=:predictions)                 # APM/APR
```

## File Organization

```
src/
├── Margins.jl                         # Module definition, exports
├── types.jl                           # EffectsResult, PredictionsResult, ContrastResult
├── engine/                            # Engine layer
│   ├── core.jl                        # MarginsEngine construction and management
│   ├── caching.jl                     # Engine caching system
│   └── measures.jl                    # Measure conversions (elasticities)
├── kernels/                           # Computational kernels
│   └── categorical.jl                 # Categorical AME kernel (ContrastEvaluator-based)
├── population/                        # Population margins implementation
│   ├── core.jl                        # Main population_margins() entry point
│   ├── contexts.jl                    # Context effects (scenarios/groups)
│   ├── continuous_effects.jl          # Continuous AME computation
│   ├── categorical_effects.jl         # Categorical AME computation
│   ├── effects.jl                     # Effect accumulation
│   └── effects_buffers.jl             # Buffer management
├── profile/                           # Profile margins implementation
│   ├── core.jl                        # Main profile_margins() entry point
│   ├── continuous_effects.jl          # MEM computation
│   ├── categorical_effects.jl         # Profile categorical effects
│   └── reference_grids.jl            # Reference grid builders (means_grid, cartesian_grid, etc.)
├── computation/                       # Shared computation utilities
│   ├── statistics.jl                  # Statistical computations
│   ├── result_formatting.jl           # Result formatting and presentation
│   ├── predictions.jl                 # Prediction accumulation
│   ├── scenarios.jl                   # ContrastPair utilities
│   └── marginal_effects.jl           # ME computation utilities
├── inference/                         # Statistical inference
│   ├── delta_method.jl               # Delta-method SE computation
│   ├── marginal_effects.jl           # ME inference wrappers (dispatch hub)
│   ├── marginal_effects_automatic_diff.jl  # AD backend
│   ├── marginal_effects_finite_diff.jl     # FD backend
│   ├── derivative_utilities.jl       # _matrix_multiply_eta!, continuous_variables
│   ├── contrast_gradient.jl          # Contrast gradient computation
│   └── mixture_utilities.jl          # Categorical mixture support (mix(), CategoricalMixture)
├── core/                              # Core utilities
│   ├── variable_detection.jl          # Variable type detection
│   ├── validation.jl                  # Input validation
│   ├── margins_validation.jl          # Margins-specific validation
│   ├── engine_validation.jl           # Engine validation
│   ├── buffer_management.jl           # Buffer allocation
│   ├── data_conversion.jl            # Data type conversions
│   └── typical_values.jl             # Representative value computation
├── second_differences/               # Second differences (interaction effects)
│   ├── contrasts.jl                  # second_differences(), pairwise, all_contrasts
│   ├── at_point.jl                   # second_differences_at() (profile-based local derivatives)
│   └── utilities.jl                  # Shared utilities for second differences
└── features/                          # Advanced features
    └── averaging.jl                   # Profile averaging with delta-method SEs
```

## Core Components

### MarginsEngine

The central data structure wrapping FormulaCompiler components with pre-allocated buffers.

```julia
struct MarginsEngine{L<:GLM.Link, U<:MarginsUsage, D<:DerivativeSupport, C<:UnifiedCompiled}
    # FormulaCompiler components (pre-compiled)
    compiled::C
    de::Union{AbstractDerivativeEvaluator, Nothing}

    # Variable index mapping (O(1) lookups instead of O(n) linear search)
    var_index_map::Dict{Symbol, Int}

    # Categorical contrast evaluator
    contrast::Union{ContrastEvaluator, Nothing}

    # Core buffers (zero runtime allocation)
    g_buf::Vector{Float64}              # Marginal effects results
    gβ_accumulator::Vector{Float64}     # AME gradient accumulation
    η_buf::Vector{Float64}              # Linear predictor computation
    row_buf::Vector{Float64}            # Design matrix row

    # Batch operation buffers
    batch_ame_values::Vector{Float64}   # Averaged marginal effects accumulator
    batch_gradients::Matrix{Float64}    # Parameter gradients (n_vars × n_params)
    batch_var_indices::Vector{Int}      # Variable indices scratch
    deta_dx_buf::Vector{Float64}        # Marginal effects on linear predictor scale
    cont_var_indices_buf::Vector{Int}   # Continuous variable index mapping

    # Continuous effects buffers
    g_all_buf::Vector{Float64}          # All continuous variable marginal effects
    Gβ_all_buf::Matrix{Float64}         # All continuous variable parameter gradients

    # Categorical effects buffers (zero-allocation via ContrastEvaluator)
    contrast_buf::Vector{Float64}
    contrast_grad_buf::Vector{Float64}
    contrast_grad_accum::Vector{Float64}

    # Persistent EffectsBuffers for zero-allocation population_margins_raw!
    effects_buffers::EffectsBuffers

    # Cached variable metadata
    continuous_vars::Vector{Symbol}
    categorical_vars::Vector{Symbol}

    # Model parameters
    model::Any
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple
end
```

**Type Parameters:**
- `L <: GLM.Link`: Link function type (IdentityLink, LogitLink, etc.)
- `U <: MarginsUsage`: Usage pattern (PopulationUsage or ProfileUsage) — controls buffer sizing
- `D <: DerivativeSupport`: Whether derivatives are needed (HasDerivatives or NoDerivatives)
- `C <: UnifiedCompiled`: Compiled formula type from FormulaCompiler

**Key Properties:**
- Single compilation per model+data combination
- All buffers pre-allocated — zero allocation in hot paths
- Cached by data signature for reuse across calls
- Separates population vs profile usage for optimal buffer sizing

### Result Types

**`EffectsResult`**: Marginal effects (AME, MEM) — includes `variables`, `terms`, `gradients`, `profile_values`, `group_values`
**`PredictionsResult`**: Adjusted predictions (AAP, APM) — streamlined without variable/contrast fields
**`ContrastResult`**: Categorical contrasts with structured output

All implement the Tables.jl interface for `DataFrame` conversion. Effects support multiple formats: `:standard`, `:compact`, `:confidence`, `:profile`, `:stata`.

## Data Flow Architecture

### Population Margins Flow

```
population_margins(model, data; type=:effects, scenarios=..., groups=...)
    ↓
Tables.columntable(data)                    # Single conversion
    ↓
build_engine(PopulationUsage, ...)          # Compilation + caching
    ↓
┌─ If scenarios/groups specified:
│      contexts.jl dispatches per-context:
│          stratify data → compute AME within each subgroup/scenario
└─ Else: direct AME computation
    ↓
┌─ Continuous: accumulate_ame_gradient!()   # Zero-alloc gradient accumulation
└─ Categorical: categorical_contrast_ame!() # Zero-alloc via ContrastEvaluator
    ↓
delta_method_se(gradient, Σ)               # Standard errors
    ↓
EffectsResult or PredictionsResult          # Type-safe result
```

### Profile Margins Flow

```
profile_margins(model, data, reference_grid; type=:effects)
    ↓
Tables.columntable(reference_grid)           # Convert grid
    ↓
build_engine(ProfileUsage, ...)              # Compilation + caching
    ↓
For each row in reference_grid:
    ├─ Continuous: marginal_effects_eta!()   # Derivative at profile
    └─ Categorical: row-specific baseline contrast
    ↓
delta_method_se(gradient, Σ)                # Standard errors per row
    ↓
EffectsResult or PredictionsResult
```

Profile margins are **O(1) with respect to dataset size** — they evaluate only at the reference grid rows, not the full dataset.

### Second Differences Flow

```
# Discrete approach (population-based):
population_margins(model, data; scenarios=(z=[0,1],), type=:effects)
    → EffectsResult with AMEs at each modifier level
        → second_differences(ames, :x, :z, vcov(model))
            → Delta-method SE from gradient differences

# Local derivative approach (profile-based):
second_differences_at(model, data, :x, :z, vcov(model); at=40)
    → Finite difference: AME(z=40+δ) - AME(z=40-δ) / 2δ
    → Delta-method SE from gradient information
```

## Backend Selection

Both `:ad` (automatic differentiation) and `:fd` (finite differences) backends achieve zero allocation after warmup. **`:ad` is the default for both population and profile analysis.**

| Property | `:ad` (Default) | `:fd` |
|----------|----------------|-------|
| Accuracy | Machine precision (exact) | Numerical approximation |
| Allocation | Zero after warmup | Zero after warmup |
| Domain safety | Handles log(), sqrt(), 1/x | Requires well-behaved domain |
| Use case | Recommended default | Simple formulas, legacy |

There are **no silent fallbacks** — if the requested backend fails, the error propagates.

## `scenarios` vs `groups` Parameters

**`scenarios`** — Counterfactual analysis (Stata's `at()`):
- Override variable values for entire population
- "What if everyone had income=50k?"
- Creates Cartesian product of specified values

**`groups`** — Subgroup analysis (Stata's `over()`):
- Stratify by observed data groups
- "What is the effect within each region?"
- Supports categorical variables, continuous binning (quartiles/custom thresholds), and hierarchical nesting

**Skip rule**: When a variable appears in both `vars` and `scenarios`/`groups`, that variable's own effect is skipped (computing ∂y/∂x while holding x fixed is contradictory).

**Combined usage**: Groups and scenarios combine multiplicatively — each subgroup × each scenario.

## Categorical Variable Behavior

**Population (traditional baseline contrasts):**
- E[ŷ|level] - E[ŷ|baseline] averaged across sample

**Profile (row-specific baseline contrasts):**
- Each reference grid row specifies both covariate values AND categorical level
- Contrast between that row's level and baseline, evaluated at that row's exact covariate profile
- Row with (x=2, cat=B, z=20) computes ŷ(x=2,cat=B,z=20) - ŷ(x=2,cat=A,z=20)

## Performance Architecture

### Zero-Allocation Targets

| Path | Allocation | Complexity |
|------|-----------|------------|
| Population AME (continuous) | 0 bytes | O(n) per variable |
| Population AME (categorical) | 0 bytes | O(n) per contrast |
| Profile effects | 0 bytes | O(k) where k = grid rows |
| Profile predictions | 0 bytes | O(k) |
| Second differences | Negligible | O(1) (vector ops on pre-computed AMEs) |

### Memory Management

All buffers pre-allocated in `MarginsEngine` and reused:
- `g_buf`, `gβ_accumulator`: Core effect computation
- `η_buf`, `row_buf`: Formula evaluation
- `batch_*` buffers: Multi-variable batch operations
- `contrast_*` buffers: Categorical ContrastEvaluator operations
- `effects_buffers`: Persistent container for `population_margins_raw!`

### Caching Strategy

Compiled engines are cached by model+data signature:
```julia
const COMPILED_CACHE = Dict{UInt64, MarginsEngine}()
# cache_key = hash(model, keys(data_nt), vars)
# Automatic invalidation via hash-based keys
```

## FormulaCompiler Integration

**Key APIs Used:**
```julia
# Compilation
FormulaCompiler.compile_formula(model, data_nt)
FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars)
FormulaCompiler.continuous_variables(compiled, data_nt)

# Zero-allocation evaluation
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link, backend)
FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend)
FormulaCompiler.marginal_effects_mu!(g_buf, de, β, row; link, backend)

# Gradient computation and standard errors
FormulaCompiler.me_eta_grad_beta!(gβ_temp, de, β, row, var)
FormulaCompiler.me_mu_grad_beta!(gβ_temp, de, β, row, var; link)
FormulaCompiler.delta_method_se(gradient, Σ)
```

## Robust Standard Errors

Integration with CovarianceMatrices.jl via the `vcov` parameter:
```julia
population_margins(model, data; vcov=HC1())              # Heteroskedasticity-robust
population_margins(model, data; vcov=Clustered(:firm_id)) # Clustered
population_margins(model, data; vcov=HAC(Bartlett()))      # HAC
population_margins(model, data; vcov=my_vcov_function)     # Custom function
```

The `vcov` parameter accepts CovarianceMatrices.jl estimator instances, functions returning a matrix, or uses `GLM.vcov` by default.

## Testing Architecture

```
test/
├── core/                           # Core functionality (GLM basic, column naming, etc.)
├── features/                       # Advanced features (elasticities, mixtures, etc.)
├── performance/                    # Allocation and performance regression tests
├── statistical_validation/         # Bootstrap SE validation, analytical SE, backend consistency
├── primitives/                     # Low-level primitive tests
├── inference/                      # Inference method tests
├── validation/                     # Cross-validation (contrast invariance, R comparison)
└── r_compare/                      # R marginaleffects comparison benchmarks
```

**Key testing principles:**
- Zero-allocation guarantees enforced via `@allocated` checks
- Bootstrap validation for all standard errors
- AD vs FD backend consistency
- Contrast coding invariance (dummy/effects/helmert)
- R marginaleffects cross-validation

## Design Decisions

### Reference Grids over Scenario Overrides for Profiles
- Memory efficiency: minimal synthetic data vs full data copying
- Clarity: clean evaluation points vs complex data correlations
- Performance: single compilation per grid structure

### Error-First Philosophy
- Statistical correctness errors propagate — no silent fallbacks
- If `:ad` is requested and fails, the error surfaces (no silent switch to `:fd`)
- Invalid statistical requests (e.g., effect of x while holding x fixed) are handled by the skip rule, not silently

### Aggressive Caching
- FormulaCompiler compilation is expensive (milliseconds)
- Users often call margins on same model+data
- Small memory cost for major performance gain via `COMPILED_CACHE`
